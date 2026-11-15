"""在官方检查点的隐藏表示上更新输出层，用真实十轮反馈检验微调的采用与交接。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import (BASE, ROOT, PROFILES, _trainer, atomic_json, delivered, digest,
                                      official, read, reference_folders, remaining_cycles, remaining_minutes, summary)
from compare_proposal_frequency import TASKS
from harness.runtime.storage import LocalStore
from harness.models.readout_training import fit_readout

DEST = BASE/"readout-ridge"


def compare(cohort):
    task = TASKS[cohort]
    package, checksum, _ = official(task)
    fixed, paper = reference_folders(cohort,"mature")
    samples = read(fixed/"state.json")["samples"]
    ids = {i for i,r in samples.items() if r["role"]=="fit"}
    fit_rows = [r for r in package.train if r.sample_id in ids]
    if {r.sample_id for r in fit_rows} != ids:
        raise ValueError("训练行须完全来自官方训练集")
    x = np.array([r.values for r in fit_rows])
    y = np.array([r.label for r in fit_rows])
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    out = DEST/cohort
    output, history, builds = {}, {}, {}
    with LocalStore(out/"lock").exclusive(), TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),PROFILES[task]) as trainer:
        parent = out/"models"/"initial.npz"
        parent.parent.mkdir(exist_ok=True, parents=True)
        parent.write_bytes(package.checkpoint)
        for arm in ("parent","bias","ridge"):
            live = parent
            stream, events, trained = {}, [], {}
            for r in range(10):
                roles = {role:[i for i,s in samples.items() if s["role"]==role and s["available_at"]==r]
                         for role in ("explore","adopt","serve")}
                if not all(roles.values()) or any(samples[i]["label"] is not None for i in roles["serve"]):
                    raise ValueError("当轮缺观察或服务输入带答案")
                current = trainer.Model.load(live)
                adopted, checks = False, {}
                if arm != "parent":
                    if live.name not in trained:
                        start = process_time()
                        candidate = trainer.Model.load(live)
                        fit_readout(candidate,x,y,arm)
                        path = out/"models"/f"{arm}-r{r+1}.npz"
                        candidate.save(path)
                        trained[live.name] = {"path":path.name, "cpu":process_time()-start,
                                              "parent_sha256":sha256(live.read_bytes()).hexdigest(),
                                              "sha256":sha256(path.read_bytes()).hexdigest()}
                    child = out/"models"/trained[live.name]["path"]
                    candidate = trainer.Model.load(child)
                    for role in ("explore","adopt"):
                        rows = [samples[i] for i in roles[role]]
                        data, truth = np.array([v["values"] for v in rows]),np.array([v["label"] for v in rows])
                        before, after = [float(np.sqrt(np.mean((m.predict(data)-truth)**2))) for m in (current,candidate)]
                        checks[role] = [before,after]
                        if after >= before-1e-12:
                            break
                    else:
                        live, current, adopted = child,candidate,True
                service = roles["serve"]
                pred = current.predict(np.asarray([samples[i]["values"] for i in service]))
                if not np.isfinite(pred).all() or stream.keys() & set(service):
                    raise ValueError("交接出现非有限预测或重复测点")
                stream.update(zip(service,map(float,pred),strict=True))
                events.append({"round":r+1,"selected":live.name,"adopted":adopted,"checks":checks})
            output[arm], history[arm], builds[arm] = stream, events, trained
        old = BASE/"selected"/task if cohort==task else fixed.parent/"project"
        first = BASE/"live-first"/("public" if cohort==task else "")/cohort
        for arm,folder in (("default",old),("mature",paper),("first",first)):
            output[arm] = delivered(folder)
        sealed = out/"issued.json"
        atomic_json(sealed,{"source":checksum,"fit_ids":digest(sorted(ids)),"predictions":output,
                            "history":history,"builds":builds})
        predictions = read(sealed)["predictions"]
        truth = {r.sample_id:r.label for r in package.validation}
        if cohort=="phm08":
            truth = remaining_cycles()
        elif cohort=="pronostia":
            truth = remaining_minutes("full-test-manifest.json")
        stats = summary(task,truth,predictions,{i:s["unit_id"] for i,s in samples.items()})
        atomic_json(out/"scores.json", {"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":stats})
        print(cohort, {k:round(v["score"],3) for k,v in stats.items()},flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    choices = ("bearing","pronostia","engine","phm08")
    parser.add_argument("cohort", choices=(*choices,"all"))
    value = parser.parse_args().cohort
    for cohort in choices if value=="all" else [value]:
        compare(cohort)
