"""按公开维护时序从已交接父版逐棵续接残差树，封存测点预测后对照各个实际参照。"""
from hashlib import sha256
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from compare_calibration_expiry import archived
from compare_group_objectives import BASE, ROOT, PROFILES, _trainer, atomic_json, delivered, digest, metric, official, read, reference_folders, remaining_cycles, remaining_minutes, summary
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches

OUT = BASE/"residual-trees"


def update(task, parent, tree_prediction):
    value = np.maximum(0,parent+.5*tree_prediction)
    if task=="gas":
        return value/value.sum(axis=1,keepdims=True)
    return value


def labels(task,values):
    return values.argmax(axis=1)+1 if task=="gas" else values


def tree_spec(tree):
    t=tree.tree_
    return {k:getattr(t,k).tolist() for k in ("children_left","children_right","feature","threshold","value")}


def compare(cohort):
    task={"phm08":"engine","pronostia":"bearing"}.get(cohort,cohort)
    package,checksum,batches=official(task)
    fixed,paper=reference_folders(cohort,"mature")
    old=BASE/"selected"/task if cohort==task else fixed.parent/"project"
    samples=read(fixed/"state.json")["samples"]
    fit_ids={s.sample_id for s in batches[0].samples if s.role=="fit"}
    if {i for i,v in samples.items() if v["role"]=="fit"}!=fit_ids:
        raise ValueError("训练清单与同条件对照不同")
    rows=[r for r in package.train if r.sample_id in fit_ids]
    fit_x=np.asarray([r.values for r in rows])
    target=np.asarray([r.label for r in rows])
    if task=="gas":
        target=np.eye(6)[target.astype(int)-1]
    obs=[r for r in samples.values() if r["role"]!="fit"]
    ids=[r["sample_id"] for r in obs]
    all_x=np.array([r["values"] for r in obs])
    later=[]
    if task=="gas":
        later=[r for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
               if b.batch_number>=3 for r in b.records]
        ids.extend(r.sample_id for r in later)
        all_x=np.vstack((all_x,[r.values for r in later]))
    if len(set(ids))!=len(ids):
        raise ValueError("重复测点")
    index={k:j for j,k in enumerate(ids)}
    out=OUT/cohort
    streams,history,later_preds,costs,trees={},{},{},{},{}
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),PROFILES[task]) as trainer:
        start_round=0
        prefix={}
        if task=="gas":
            model,identity,records=archived(old,trainer,Path(tmp),"gas")
            start_round=4
            for r in records[:4]:
                payload=read(old/r["submitted"])["payload"]
                if payload["model_id"]!=r["selected"]:
                    raise ValueError("气体热身版本不符")
                prefix.update(payload["predictions"])
        else:
            p=Path(tmp)/"parent.npz"
            p.write_bytes(package.checkpoint)
            model=trainer.Model.load(p)
            identity=sha256(package.checkpoint).hexdigest()
        original_fit,original_pred=model.predict(fit_x),model.predict(all_x)
        for arm,depth in (("parent",0),("stump",1),("tree3",3)):
            fit_pred,active=original_fit,original_pred
            stream=dict(prefix)
            events,built=[],[]
            train_seconds=0.
            pending=None
            for r in range(start_round,10):
                roles={role:[i for i,z in samples.items() if z["role"]==role and z["available_at"]==r] for role in ("explore","adopt","serve")}
                if not all(roles.values()) or any(samples[i]["label"] is not None for i in roles["serve"]):
                    raise ValueError("缺当轮档案或服务测量含答案")
                event={"round":r+1,"adopted":False}
                if depth:
                    if pending is None:
                        start=perf_counter()
                        tree=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=max(5,ceil(.02*len(rows))),random_state=17)
                        tree.fit(fit_x,target-fit_pred)
                        train_seconds+=perf_counter()-start
                        built.append(tree_spec(tree))
                        pending=(update(task,fit_pred,tree.predict(fit_x)),update(task,active,tree.predict(all_x)))
                    for role in ("explore","adopt"):
                        members=roles[role]
                        y={i:samples[i]["label"] for i in members}
                        positions=[index[i] for i in members]
                        a,b=[metric(task,y,dict(zip(members,labels(task,p[positions]).tolist(),strict=True))) for p in (active,pending[1])]
                        event[role]=[a,b]
                        if (b-a)*(1 if task=="gas" else -1)<=1e-12:
                            break
                    else:
                        fit_pred,active=pending
                        pending=None
                        event["adopted"]=True
                values=labels(task,active[[index[i] for i in roles["serve"]]])
                stream.update(zip(roles["serve"],values.tolist(),strict=True))
                events.append(event)
            streams[arm],history[arm],trees[arm]=stream,events,built
            costs[arm]={"physical_fits":len(built),"tree_fit_seconds":train_seconds,"rows_per_fit":len(rows)}
            if later:
                later_preds[arm]=dict(zip([s.sample_id for s in later],labels(task,active[-len(later):]).tolist(),strict=True))
    paths={"old":old,"fixed":fixed,"mature":paper}
    if task=="engine":
        paths["current"]=BASE/"live-first"/("public" if cohort==task else "")/cohort
    if cohort!=task:
        paths["simple"]=fixed.parent/"simple"
    for arm,p in paths.items():
        streams[arm]=delivered(p)
    sealed=out/"issued.json"
    atomic_json(sealed,{"source":checksum,"initial_model":identity,"fit":digest(sorted(fit_ids)),"trees":trees,
                        "events":history,"predictions":streams,"future":later_preds})
    receipt=read(sealed)
    truth={s.sample_id:s.label for s in package.validation}
    if cohort=="pronostia":
        truth=remaining_minutes("full-test-manifest.json")
    elif cohort=="phm08":
        truth=remaining_cycles()
    stats=summary(task,truth,receipt["predictions"],{i:s["unit_id"] for i,s in samples.items()})
    for arm,pred in receipt["predictions"].items():
        stats[arm]["rounds"]=[metric(task,truth,{i:v for i,v in pred.items() if samples[i]["available_at"]==r}) for r in range(10)]
    later_scores={}
    if later:
        truth={s.sample_id:s.label for s in later}
        later_scores={a:metric(task,truth,p) for a,p in receipt["future"].items()}
    atomic_json(out/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":stats,"future":later_scores,"costs":costs})
    print(cohort,{k:round(s["score"],5) for k,s in stats.items()},later_scores,flush=True)


if __name__=="__main__":
    for cohort in ("gas","bearing","pronostia","engine","phm08"):
        compare(cohort)
