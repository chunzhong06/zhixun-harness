"""固定同一维护模型链，实测当批有无当地证据时使用浅树或热身锚点的差别。"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import OUT as MIXED, OTHER, archived,old_step,source_window,unlabelled
from compare_group_objectives import BASE,ROOT,_trainer,atomic_json,delivered,digest,metric,official,read,reference_folders,remaining_minutes,summary
from compare_residual_trees import OUT
from harness.models.bearing_stumps import stump_model


def score(model,rows):
    observed=np.asarray([s["label"] for s in rows])
    pred=model.predict(np.array([s["values"] for s in rows]))
    return float(np.sqrt(np.mean((observed-pred)**2)))


def alternate(anchor,fit,schedule,prefix,start_tree,base_type):
    # 常规分支须使用原训练器。对同一基权重的浅树训练只发生于树的轮次。
    base=deepcopy(anchor)
    base.__class__=base_type
    tree=None
    cached_base,cached_tree=None,None
    turn=0
    stream=dict(prefix)
    history=[]
    for stage,(explore,confirm,serve) in schedule:
        event={"stage":stage,"constructed":False,"adopted":False,"confirmed":False}
        if explore and confirm:
            tree_turn=(turn%2==0)==start_tree
            turn+=1
            event["scheduled"]="tree" if tree_turn else "regular"
            if tree_turn:
                if cached_tree is None:
                    cached_tree=deepcopy(tree if tree is not None else base)
                    cached_tree.__class__=type(anchor)
                    cached_tree.fit(*fit,1)
                    event["constructed"]=True
                incumbent=tree if tree is not None else base
                if all(score(cached_tree,r)<score(incumbent,r)-1e-12 for r in (explore,confirm)):
                    tree,cached_tree=cached_tree,None
                    event["adopted"]=True
            else:
                base,cached_base,attempt,_=old_step("bearing","old",base,fit,explore,confirm,serve,cached_base)
                event.update(constructed=attempt["physical_fit"],adopted=attempt["adopted"])
                if attempt["adopted"]:
                    tree,cached_tree=None,None
            event["confirmed"]=tree is not None and all(score(tree,r)<score(base,r)-1e-12 for r in (explore,confirm))
        active=tree if event["confirmed"] else base
        pred=active.predict(np.array([r["values"] for r in serve]))
        stream.update(zip((r["sample_id"] for r in serve),map(float,pred),strict=True))
        history.append(event)
    return stream,history


def run(cohort,mixed,alternating=False):
    fixed,_=reference_folders(cohort,"mature")
    old=BASE/"selected"/"bearing" if cohort=="bearing" else fixed.parent/"project"
    src=read(fixed/"state.json")["samples"]
    package,checksum,_=official("bearing")
    fit_ids={i for i,s in src.items() if s["role"]=="fit"}
    rows=[r for r in package.train if r.sample_id in fit_ids]
    x,y=np.array([r.values for r in rows]),np.array([r.label for r in rows])
    ref=read(MIXED/cohort/"issued.json") if mixed else None
    if mixed:
        blocks,target_truth=unlabelled("bearing")
        other=read(reference_folders(OTHER[cohort],"mature")[0]/"state.json")["samples"]
        schedule=[("before",source_window(src,r)) for r in range(4,8)]
        schedule.extend(("unlabelled",([],[],b)) for b in blocks)
        schedule.extend(("new_feedback",source_window(other,r)) for r in range(4))
        schedule.extend(("return",source_window(src,r)) for r in (8,9))
    else:
        target_truth={}
        schedule=[("service",source_window(src,r)) for r in range(4,10)]
    streams={"keep":{},"confirmed":{}}
    stages,units={},{}
    for r,rec in enumerate(read(old/"run.json")["rounds"][:4]):
        payload=read(old/rec["submitted"])["payload"]
        if payload["model_id"]!=rec["selected"]:
            raise ValueError("热身权重不符")
        for stream in streams.values():
            stream.update(payload["predictions"])
        for s in source_window(src,r)[2]:
            stages[s["sample_id"]],units[s["sample_id"]]="warmup",s["unit_id"]
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        anchor,anchor_sha,_=archived(old,trainer,Path(tmp),"bearing")
        anchor.__class__=stump_model(trainer.Model)
        if mixed and ref["prefix_sha256"]["old"]!=anchor_sha:
            raise ValueError("混合时序的锚点不符")
        current=anchor
        candidate=None
        events=[]
        if alternating:
            prefix=streams["keep"]
            streams={}
            for arm in ("regular-first","tree-first"):
                streams[arm],history=alternate(anchor,(x,y),schedule,prefix,arm=="tree-first",trainer.Model)
                events.extend({**e,"arm":arm} for e in history)
            for stage,(_,_,serve) in schedule:
                for row in serve:
                    key=row["sample_id"]
                    if key in stages or row["label"] is not None:
                        raise ValueError("服务输入重复或携带答案")
                    stages[key],units[key]=stage,row["unit_id"]
        for stage,(explore,confirm,serve) in ([] if alternating else schedule):
            ids=[s["sample_id"] for s in serve]
            if any(s["label"] is not None for s in serve) or stages.keys() & set(ids):
                raise ValueError("服务标签或编号不符")
            built=False
            adopted=False
            verified=False
            if explore and confirm:
                if candidate is None:
                    candidate=deepcopy(current)
                    candidate.fit(x,y,1)
                    built=True
                if all(score(candidate,rr)<score(current,rr)-1e-12 for rr in (explore,confirm)):
                    current,candidate=candidate,None
                    adopted=True
                verified=all(score(current,rr)<score(anchor,rr)-1e-12 for rr in (explore,confirm))
            for arm,predictor in (("keep",current),("confirmed",current if verified else anchor)):
                values=predictor.predict(np.array([s["values"] for s in serve]))
                streams[arm].update(zip(ids,map(float,values),strict=True))
            stages.update(dict.fromkeys(ids,stage))
            units.update({s["sample_id"]:s["unit_id"] for s in serve})
            events.append({"stage":stage,"constructed":built,"adopted":adopted,"confirmed":verified})
    if mixed:
        if stages!=ref["stages"]:
            raise ValueError("不同时序参照")
        streams.update({k:ref["streams"][k] for k in ("old","paper_once","refresh")})
    else:
        streams["old"]=delivered(old)
        streams["mature"]=delivered(reference_folders(cohort,"mature")[1])
        if cohort=="pronostia":
            streams["simple"]=delivered(fixed.parent/"simple")
    out=OUT/cohort/("mixed-feedback" if mixed else "public-feedback")
    if alternating:
        out=out.with_name(out.name+"-alternating")
    p=out/"issued.json"
    atomic_json(p,{"source":checksum,"anchor":anchor_sha,"fit":digest(sorted(fit_ids)),"history":events,"predictions":streams})
    streams=read(p)["predictions"]
    truth={r.sample_id:r.label for r in package.validation}
    truth.update(remaining_minutes("full-test-manifest.json"))
    truth.update(target_truth)
    stats=summary("bearing",truth,{"parent":streams["old"],**streams},units)
    for arm,s in stats.items():
        pred=streams["old" if arm=="parent" else arm]
        s["stages"]={stage:metric("bearing",truth,{i:v for i,v in pred.items() if stages[i]==stage}) for stage in set(stages.values())}
    counts=({arm:sum(e["constructed"] for e in events if e["arm"]==arm) for arm in ("regular-first","tree-first")}
            if alternating else sum(e["constructed"] for e in events))
    atomic_json(out/"scores.json",{"issued_sha256":sha256(p.read_bytes()).hexdigest(),"arms":stats,"builds":counts})
    print(cohort,mixed,{k:round(v["score"],2) for k,v in stats.items()})


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--alternating",action="store_true")
    args=parser.parse_args()
    for c in ("bearing","pronostia"):
        for mixed in (False,True):
            run(c,mixed,args.alternating)
