"""十轮逐次更新预算。每次真实拟合从在用权重出发，独立确认后交出当前轮无标签预测。"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import (
    BASE, PROFILES, ROOT, LocalStore, _trainer, atomic_json, coverage_rows, delivered,
    digest, official, read, reference_folders, remaining_cycles, remaining_minutes, summary,
)

OUT = BASE/"update-budget"
SCHEDULES = {"unit":("unit",False), "fixed":("fixed",False), "adaptive":("adaptive",False),
             "fixed-rotate":("fixed",True), "adaptive-rotate":("adaptive",True)}


def run_path(model, epochs, schedule, rotate, batches, samples):
    bank, request, issued, log = 0, 1 if schedule!="fixed" else 4, {}, []
    for index, batch in enumerate(batches):
        bank += 1
        spent = bank if index==len(batches)-1 else request if bank>=request else 0
        event = {"round":index+1, "requested":request, "epochs":spent, "adopted":False}
        by_role = {role:sorted(i for i,r in samples.items() if r["role"]==role and r["available_at"]==index)
                   for role in ("explore","adopt","serve")}
        if any(not ids for ids in by_role.values()):
            raise ValueError("当前轮缺少合法探索、确认或服务测点")
        if spent:
            child = deepcopy(model)
            previous_steps = child.meta["steps"]
            start = perf_counter_ns()
            for j in range(spent):
                x,y = epochs[j if rotate else 0]
                child.fit(x,y,1)
            event.update(fit_seconds=(perf_counter_ns()-start)/1e9, steps=child.meta["steps"]-previous_steps)
            bank -= spent
            for role in ("explore","adopt"):
                ids = by_role[role]
                x = np.array([samples[i]["values"] for i in ids])
                y = np.array([samples[i]["label"] for i in ids])
                before,after = [float(np.sqrt(np.mean((m.predict(x)-y)**2))) for m in (model,child)]
                event[role] = [before, after]
                if after+1e-12>=before:
                    break
                if role=="adopt":
                    model = child
                    event["adopted"] = True
            if schedule=="adaptive":
                request = 1 if event["adopted"] else 4
        ids = by_role["serve"]
        if set(ids)&issued.keys() or any(samples[i]["label"] is not None for i in ids):
            raise ValueError("服务样本已交接或带有答案")
        values = model.predict(np.array([samples[i]["values"] for i in ids]))
        if values.shape!=(len(ids),) or not np.isfinite(values).all():
            raise ValueError("无效服务预测")
        issued.update(zip(ids,values.tolist(),strict=True))
        event["issued_sha256"] = digest([ids,values.tolist()])
        if schedule=="unit" and event["adopted"] != (batch["outcome"]=="adopted"):
            raise ValueError(f"一遍对照在第 {index+1} 轮未能复现真实采用")
        log.append(event)
    if bank or sum(r["epochs"] for r in log)!=len(batches):
        raise ValueError("十轮梯度预算不符")
    return issued,log


def compare(cohort):
    task = {"phm08":"engine","pronostia":"bearing"}.get(cohort,cohort)
    package, checksum, _ = official(task)
    fixed, mature = reference_folders(cohort,"mature")
    samples = read(fixed/"state.json")["samples"]
    batches = read(fixed/"run.json")["rounds"]
    if len(batches)!=10:
        raise ValueError("本协议须完整十轮")
    fit = {i for i,s in samples.items() if s["role"]=="fit"}
    rows = [r for r in package.train if r.sample_id in fit]
    if len(rows)!=len(fit):
        raise ValueError("训练池身份不符")
    epochs = [(np.array([r.values for r in seq]),np.array([r.label for r in seq]))
              for seq in coverage_rows(task,rows,package.train)]
    folder = OUT/cohort
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    with LocalStore(folder).exclusive(), TemporaryDirectory(prefix="budget-replay-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),PROFILES[task]) as trainer:
        path = Path(tmp)/"parent.npz"
        path.write_bytes(package.checkpoint)
        streams, history = {}, {}
        for name,(rule,rotate) in SCHEDULES.items():
            streams[name],history[name] = run_path(trainer.Model.load(path), epochs, rule, rotate, batches, samples)
        ids = sorted(streams["unit"])
        historical = delivered(fixed)
        if set(ids)!=set(historical) or not np.allclose([streams["unit"][i] for i in ids],
                [historical[i] for i in ids],rtol=1e-12,atol=1e-8):
            raise ValueError("每轮一遍未复现真实交接，停止新策略计分")
        sealed = folder/"issued.json"
        atomic_json(sealed,{"package_sha256":checksum,"fit_sha256":digest(sorted(fit)),
                            "predictions":streams,"history":history})
        original = BASE/"original"/task if task==cohort else fixed.parent/"original"
        old = BASE/"selected"/task if task==cohort else fixed.parent/"project"
        streams.update(parent=delivered(original),old=delivered(old),mature=delivered(mature))
        answers = {r.sample_id:r.label for r in package.validation}
        if cohort=="phm08":
            answers = remaining_cycles()
        if cohort=="pronostia":
            answers = remaining_minutes("full-test-manifest.json")
        statistics = summary(task,answers,streams,{i:samples[i]["unit_id"] for i in ids})
        atomic_json(folder/"scores.json",{"arms":statistics,"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),
                    "cost":{k:{"epochs":sum(r["epochs"] for r in v),"steps":sum(r.get("steps",0) for r in v),
                               "attempts":sum(r["epochs"]>0 for r in v),"fit_seconds":sum(r.get("fit_seconds",0) for r in v)}
                            for k,v in history.items()}})
        print(cohort,{k:round(v["score"],4) for k,v in statistics.items()})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("bearing","engine","pronostia","phm08"))
    compare(parser.parse_args().cohort)
