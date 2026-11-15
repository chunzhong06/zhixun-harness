"""独立核对常规维护发布序列，在该序列上测试一次拟合的历史参照的使用时机。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import OTHER, archived, old_step, source_window, unlabelled
from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, metric, official, read, reference_folders, remaining_cycles
from compare_neighbor_routes import DEST
from harness.data.cmapss_input import condition_key
from harness.models.feedback_neighbors import CurrentNeighbors, route

ARMS=("current","always","confirm","last","by-condition","weight")


def schedule(cohort,mixed):
    state={c:read(reference_folders(c,"mature")[0]/"state.json")["samples"] for c in (cohort,OTHER[cohort])}
    if not mixed:
        return [("service",source_window(state[cohort],r)) for r in range(4,10)],state,{}
    blocks,truth=unlabelled("engine")
    seq=[("before",source_window(state[cohort],r)) for r in range(4,8)]
    seq.extend(("unlabelled",([],[],b)) for b in blocks)
    seq.extend(("new_feedback",source_window(state[OTHER[cohort]],r)) for r in range(4))
    seq.extend(("return",source_window(state[cohort],r)) for r in (8,9))
    return seq,state,truth


def run(cohort,mixed):
    seq,state,target_truth=schedule(cohort,mixed)
    pkg,checksum,batches=official("engine")
    fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    train=[r for r in pkg.train if r.sample_id in fit_ids]
    fit=np.array([r.values for r in train]),np.array([r.label for r in train])
    folder=BASE/"live-first"/("public" if cohort=="engine" else "")/cohort
    evidence=(read(DEST/cohort/"issued.json")["predictions"] if mixed else
              read(BASE/"engine-neighbor-updates"/cohort/"issued.json")["predictions"])
    archive,=(ROOT/"data/official/engine").glob("*.zip")
    history,stages,units,decisions=[],{},{},{}
    streams={k:{} for k in ARMS}
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
        model,origin,records=archived(folder,trainer,Path(tmp),"engine")
        constructor=perf_counter()
        lookup=CurrentNeighbors(model,origin,train)
        bank_seconds=perf_counter()-constructor
        for r,rec in enumerate(records[:4]):
            p=read(folder/rec["submitted"])["payload"]["predictions"]
            for arm in ARMS:
                streams[arm].update(p)
            for s in source_window(state[cohort],r)[2]:
                stages[s["sample_id"]],units[s["sample_id"]]="warmup",s["unit_id"]
        previous={"last":0.,"by-condition":[]}
        for stage,(explore,confirm,serve) in seq:
            start=perf_counter()
            model,_,update,pred=old_step("engine","fixed",model,fit,explore,confirm,serve,None)
            base_seconds=perf_counter()-start
            start=perf_counter()
            gate=route(lookup.observations(model,explore),lookup.observations(model,confirm),previous)
            previous=gate
            _,base,ref=lookup.observations(model,serve)
            ids=[r["sample_id"] for r in serve]
            if set(ids)&stages.keys() or len(ids)!=len(set(ids)) or any(r["label"] is not None for r in serve):
                raise ValueError("服务标签或编号有误")
            if not np.allclose(base,[pred[i] for i in ids],rtol=1e-12,atol=1e-10):
                raise ValueError("基础分支预测未复现")
            for arm in ARMS:
                if arm=="by-condition":
                    w=np.array([.5 if condition_key(r["values"][21:]) in gate[arm] else 0. for r in serve])
                else:
                    w=0. if arm=="current" else gate[arm]
                stream=base+w*(ref-base)
                streams[arm].update(zip(ids,map(float,stream),strict=True))
            stages.update(dict.fromkeys(ids,stage))
            units.update({r["sample_id"]:r["unit_id"] for r in serve})
            history.append({"stage":stage,"gate":gate,"base_build":update["physical_fit"],"base_seconds":base_seconds,
                            "additional_seconds":perf_counter()-start,"serve_rows":len(ids)})
            decisions.update({i:len(history)-1 for i in ids})
    delta=max(abs(v-evidence["current"][i]) for i,v in streams["current"].items())
    if delta>1e-7:
        raise ValueError(f"常规发布轨迹不同 {delta}")
    refs=("old","paper_once","refresh") if mixed else (("fixed","mature","simple") if cohort=="phm08" else ("fixed","mature"))
    streams.update({k:evidence[k] for k in refs})
    sub="dynamic-mixed" if mixed else "dynamic-public"
    out=DEST/cohort/sub
    p=out/"issued.json"
    atomic_json(p,{"source":checksum,"parent":origin,"training_ids":digest(sorted(fit_ids)),"stages":stages,
                   "history":[{k:v for k,v in r.items() if not k.endswith("_seconds")} for r in history],
                   "decisions":decisions,"predictions":streams})
    truth={r.sample_id:r.label for r in pkg.validation}
    if mixed or cohort=="phm08":
        truth.update(remaining_cycles())
    truth.update(target_truth)
    table={}
    for arm,pred in read(p)["predictions"].items():
        if pred.keys()!=stages.keys():
            raise ValueError("对照分片不同")
        devices=defaultdict(dict)
        for i,v in pred.items():
            devices[units[i]][i]=v
        table[arm]={"score":metric("engine",truth,pred),"worst_device":max(metric("engine",truth,g) for g in devices.values()),
                    "stages":{s:metric("engine",truth,{i:v for i,v in pred.items() if stages[i]==s}) for s in set(stages.values())},
                    "window_scores":[metric("engine",truth,{r["sample_id"]:pred[r["sample_id"]] for r in rows}) for _,(_,_,rows) in seq]}
    atomic_json(out/"scores.json",{"issued_sha256":sha256(p.read_bytes()).hexdigest(),"arms":table,"bank_seconds":bank_seconds,
                                   "regular_seconds":sum(r["base_seconds"] for r in history),
                                   "shared_extra_seconds":sum(r["additional_seconds"] for r in history),"builds":sum(r["base_build"] for r in history)})
    print(cohort,sub,{k:(round(v["score"],2),round(v["worst_device"],2)) for k,v in table.items()},flush=True)


if __name__=="__main__":
    for cohort in ("engine","phm08"):
        for mixed in (False,True):
            run(cohort,mixed)
