"""把已实际执行的三条维护轨迹接到完全相同的交付工具，封存预测后按源答案评分。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import source_window
from compare_error_slice import RAW, read_cmapss, test_answers
from compare_fixed_tree_references import OUT, paths
from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, official, read, summary
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.observer_errors import ObserverErrors
from harness.models.source_combinations import SourceCombinations
from score_unseen_bearing_windows import window_answers, window_inputs


DEST = BASE/"whole-maintenance"
INTEGRATED = OUT/"setting-reference/automatic/integrated"
PROTOCOL = ROOT/"engineering/configs/experiments/whole-maintenance-comparison.json"
ARMS = ("project","fixed","paper-adapted")


def trajectory(task):
    current, other = paths(task,task)
    folders = dict(zip(ARMS,(current,other["fixed"],other["paper"]),strict=True))
    runs, identity, resources = {}, {}, {}
    settings = {}
    for arm,folder in folders.items():
        state, run = read(folder/"state.json"), read(folder/"run.json")
        root, = (m for m in state["models"].values() if m["parent_id"] is None)
        store = ModelArtifacts(folder/"models")
        root_header,_ = store.read(root["artifact"])
        if (state["config"]["max_trials"],state["config"]["max_plan_calls"],run["completed_rounds"])!=(1,1,10):
            raise ValueError("维护轨迹与事前上限不符")
        identity[arm] = (digest(state["samples"]),root_header["sha256"])
        settings[arm] = digest(state["config"])
        rounds = []
        for event in run["rounds"]:
            payload = read(folder/event["submitted"])["payload"]
            if event["selected"]!=payload["model_id"] or event["batch"]!=payload["batch_id"]:
                raise ValueError("在用权重没有对应当轮交接")
            header,blob = store.read(event["model_artifact"])
            rounds.append((header["sha256"],blob,payload["predictions"]))
        runs[arm] = rounds
        costs = []
        for job in (folder/"jobs").glob("*/state.json"):
            record=read(job)
            if record.get("status")=="completed":
                costs.append((read(job.with_name("input.json"))["request"]["kind"],record["cpu_seconds"]))
        resources[arm]={"driver":run["driver"],"source_run":str(folder.relative_to(ROOT)),
                        "receipt_sha256":sha256((folder/"run.json").read_bytes()).hexdigest(),
                        "completed_cpu_seconds":sum(v for _,v in costs),
                        "physical_continuations":sum(k=="continue" for k,_ in costs)}
    if len(set(settings.values())) != 1 or len(set(identity.values())) != 1:
        raise ValueError("各臂的源训练、实时角色、回标时钟或根权重不同")
    return runs, read(current/"state.json")["samples"], identity["project"], resources


def public_batches(samples):
    owners={}
    batches=[]
    for j in range(10):
        explore,confirm,rows=source_window(samples,j)
        if any(row["label"] is not None for row in rows):
            raise ValueError("服务行携带答案")
        owners.update((r["sample_id"],r["unit_id"] or r["group"]) for r in rows)
        batches.append((j,[r["sample_id"] for r in rows],[r["values"] for r in rows],explore+confirm))
    return batches,owners


def new_engines():
    tracks={s:read_cmapss(RAW,subset=s,partition="test") for s in ("FD002","FD003","FD004")}
    rounds=defaultdict(list)
    devices, gap = {},{}
    for subset,units in tracks.items():
        for t in units:
            for j in range(10):
                cycle=9+20*j
                if cycle>=len(t.values):
                    continue
                key=f"{subset}/{t.unit}/{cycle}"
                x=t.values[cycle-1]
                rounds[j].append((key,[*x[3:],*x[:3]]))
                devices[key]=t.unit_id
                gap[key]=len(t.values)-cycle
    for suffix in ("","-offset6","-offset11","-offset16"):
        old=BASE/"live-first"/("cross-stream"+suffix)/"issued.json"
        if read(old)["predictions"]["fixed"].keys() & devices.keys():
            raise ValueError("新增测点曾进入已登记的错位测量")
    def truth():
        end={k:v for s,units in tracks.items() for k,v in test_answers(RAW,s,units).items()}
        return {k:end[u]+gap[k] for k,u in devices.items()}
    return [(j,[k for k,_ in rows],[x for _,x in rows],[]) for j,rows in sorted(rounds.items())],devices,truth


def fresh_bearing():
    rounds,metadata,ends=window_inputs(54)
    return ([(j,[k for k,_ in rows],[x for _,x in rows],[]) for j,rows in sorted(rounds.items())],
            {k:v[0] for k,v in metadata.items()},lambda:window_answers(metadata,ends))


def late_gas():
    data=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    batches,truth,owners=[],{},{}
    for b in data:
        if b.batch_number<3:
            continue
        keys=[r.sample_id for r in b.records]
        feedback=[{"sample_id":r.sample_id,"values":r.values,"label":r.label} for r in b.records]
        batches.append((3,keys,[r.values for r in b.records],feedback))
        owners.update(dict.fromkeys(keys,str(b.batch_number)))
        truth.update((r.sample_id,r.label) for r in b.records)
    return batches,owners,lambda:truth


def integrated_predictions(task):
    folder=INTEGRATED/task
    r=read(folder/"run.json")
    if "accepted" in r:
        files=[folder/"accepted"/(e["request"]+".json") for e in r["accepted"]]
    else:
        files=[folder/e["submitted"] for e in r["rounds"]]
    return {k:v for f in files for k,v in read(f)["payload"]["predictions"].items()}


def evaluate(task, suite, inputs, runs, identity, resources, fit, trainer, work):
    batches,owners,truth=inputs
    streams=defaultdict(dict)
    trace=[]
    repair=SourceCombinations([r.values for r in fit]) if task=="engine" else None
    bank=GasFeedbackBank(fit) if task=="gas" else None
    observers={a:ObserverErrors(bank.names,6) for a in ARMS} if bank else {}
    seen=set()
    cls=pair_model(trainer.Model) if task=="gas" else trainer.Model
    for ordinal,(j,ids,values,pool) in enumerate(batches,1):
        if seen.intersection(ids) or len(ids)!=len(set(ids)):
            raise ValueError("同一服务样本重复发放")
        seen.update(ids)
        x=np.asarray(values)
        choices=sorted(pool,key=lambda r:digest([17,r["sample_id"]]))[:32] if bank else []
        event={"round":ordinal,"source_round":j+1,"ids":ids,"returned_after":[r["sample_id"] for r in choices],"weights":{}}
        for a in ARMS:
            sha,blob,native=runs[a][j]
            path=work/"published.npz"
            path.write_bytes(blob)
            model=cls.load(path)
            if bank:
                votes=bank.predict(model,x)
                raw=votes["current"]
                shared=observers[a].predict(votes,"uniform")
            else:
                raw=model.predict(x)
                shared=model.predict(repair.transform(x)) if repair is not None else raw
            if suite=="public" and (set(ids)!=native.keys() or not np.allclose(raw,[native[k] for k in ids],rtol=0,atol=1e-7)):
                raise ValueError("原生轨迹未复现历史已发预测")
            event["weights"][a]=sha
            for arm,v in ((a,shared),(a+"-raw",raw)):
                streams[arm].update(zip(ids,v.tolist(),strict=True))
        # 三组的本轮服务都已计算，先固化发放再给各臂开放该轮选中的标签。
        out=DEST/task/suite
        part=out/f"round-{ordinal:02}.json"
        atomic_json(part,{"event":event,"predictions":{a:{k:v[k] for k in ids} for a,v in streams.items()}})
        trace.append(sha256(part.read_bytes()).hexdigest())
        if choices:
            for a in ARMS:
                path.write_bytes(runs[a][j][1])
                observed=bank.predict(cls.load(path),np.asarray([r["values"] for r in choices]))
                observers[a].update([r["label"] for r in choices],observed)
    y=truth()
    p=defaultdict(dict)
    for n,h in enumerate(trace,1):
        f=out/f"round-{n:02}.json"
        if sha256(f.read_bytes()).hexdigest()!=h:
            raise ValueError("发放文件身份变化")
        for a,chunk in read(f)["predictions"].items():
            p[a].update(chunk)
    if suite=="public":
        actual=integrated_predictions(task)
        if actual.keys()!=p["project"].keys() or max(abs(v-actual[k]) for k,v in p["project"].items())>1e-7:
            raise ValueError("项目组没有复现当前三任务交付入口")
    table=summary(task,y,{"parent":p["fixed"],**p},owners)
    del table["parent"]
    if task=="gas":
        for a,stream in p.items():
            recalls=[sum(stream[k]==c for k in seen if y[k]==c)/sum(y[k]==c for k in seen)
                     for c in sorted({y[k] for k in seen})]
            table[a]["min_recall"]=min(recalls)
    dominant=all(table["project"]["score"]>=table[a]["score"]+.01 and
                 table["project"]["min_recall"]>=table[a]["min_recall"]-.02 for a in ARMS[1:]) if bank else all(
                     table["project"]["score"]<=.99*table[a]["score"] and table["project"]["worst_group"]<=1.01*table[a]["worst_group"] for a in ARMS[1:])
    result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"sample_history_and_root":identity,
            "source_runs":resources,"round_sha256":trace,"rows":len(seen),"groups":len(set(owners.values())),
            "different_from_fixed":sum(p["project"][i]!=p["fixed"][i] for i in seen),
            "meets_gate":dominant,"arms":table}
    atomic_json(out/"scores.json",result)
    print(task,suite,len(seen),{a:round(v["score"],5) for a,v in table.items()},flush=True)


def main():
    for task in ("gas","bearing","engine"):
        package,_,batches=official(task)
        fit=[r for r in batches[0].samples if r.role=="fit"]
        runs,samples,identity,resources=trajectory(task)
        pub,owners=public_batches(samples)
        incoming={"public":(pub,owners,lambda:{r.sample_id:r.label for r in package.validation})}
        incoming["external"]={"gas":late_gas,"bearing":fresh_bearing,"engine":new_engines}[task]()
        archive,=(ROOT/"data/official"/task).glob("*.zip")
        with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),{"gas":"A","bearing":"B","engine":"C"}[task]) as trainer:
            for suite,data in incoming.items():
                evaluate(task,suite,data,runs,identity,resources,fit,trainer,Path(tmp))


if __name__=="__main__":
    main()
