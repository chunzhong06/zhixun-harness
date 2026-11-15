"""训练资料内分开学习阈值和修正幅度；已计分外部观测仅用于冻结后的复核。"""
from hashlib import sha256
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from compare_group_objectives import BASE,ROOT,_trainer,atomic_json,delivered,digest,official,read,reference_folders,remaining_minutes,summary
from compare_residual_trees import OUT,tree_spec
from compare_calibration_expiry import source_window
from score_released_bearing_trees import tree_prediction
from score_unseen_bearing_windows import window_inputs,window_answers
from check_source_mean import transfer_truth


def candidate(arm,x,residual,units,sparse):
    if arm=="constant":
        value=float(np.median([residual[units==u].mean() for u in np.unique(units)]))
        return {"children_left":[-1],"children_right":[-1],"feature":[-2],"threshold":[-2.],"value":[[[value]]]}
    structural=np.ones(len(x),dtype=bool) if arm=="all-tree" else sparse
    min_leaf=max(5,ceil(.02*structural.sum()))
    tree=DecisionTreeRegressor(max_depth=1,min_samples_leaf=min_leaf,random_state=17)
    tree.fit(x[structural],residual[structural])
    spec=tree_spec(tree)
    if arm in ("original-tree","all-tree"):
        return spec
    estimate=~sparse
    leaf=tree.apply(x)
    for node in np.unique(leaf):
        mask=(leaf==node)&estimate
        if not mask.any():
            spec["value"][node][0][0]=0.
            continue
        v=(float(np.median([residual[mask&(units==u)].mean() for u in np.unique(units[mask])]))
           if arm=="held-units" else float(residual[mask].mean()))
        spec["value"][node][0][0]=v
    return spec


def run(cohort):
    pkg,checksum,batches=official("bearing")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=list(pkg.train)
    x=np.asarray([r.values for r in rows])
    y=np.asarray([r.label for r in rows])
    units=np.asarray([r.unit_id for r in rows])
    sparse=np.asarray([r.sample_id in ids for r in rows])
    fixed,paper=reference_folders(cohort,"mature")
    state=read(fixed/"state.json")["samples"]
    old=BASE/"selected/bearing" if cohort=="bearing" else fixed.parent/"project"
    source=[r for r in state.values() if r["role"]!="fit"]
    keys=[r["sample_id"] for r in source]
    xx=np.asarray([r["values"] for r in source])
    pos={k:j for j,k in enumerate(keys)}
    suites,labels,devices={},{},{}
    if cohort=="pronostia":
        for off in (8,16):
            windows,meta,ends=window_inputs(off)
            name=f"offset{off}"
            suites[name]=windows
            labels[name]=(meta,ends)
            devices[name]={i:v[0] for i,v in meta.items()}
        kaist=read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
        suites["kaist"]={9:[("KAIST/"+r["file"],r["features"]) for r in kaist if "features" in r]}
        devices["kaist"]=dict.fromkeys((i for i,_ in suites["kaist"][9]),"KAIST")
    pools={k:{} for k in ("development",*suites)}
    builds,events,cost={},{},{}
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        p=Path(tmp)/"original.npz"
        p.write_bytes(pkg.checkpoint)
        model=trainer.Model.load(p)
        for arm in ("original-tree","held-rows","held-units","all-tree","constant"):
            current_y,active=model.predict(x),model.predict(xx)
            history,orders,chain=[],[],[]
            pending=None
            spent=0.
            stream={}
            aux={s:{} for s in suites}
            for r in range(10):
                explore,confirm,serve=source_window(state,r)
                if pending is None:
                    start=perf_counter()
                    spec=candidate(arm,x,y-current_y,units,sparse)
                    spent+=perf_counter()-start
                    history.append(spec)
                    pending=(np.maximum(0,current_y+.5*tree_prediction(spec,x)),np.maximum(0,active+.5*tree_prediction(spec,xx)),spec)
                selected=False
                for group in (explore,confirm):
                    ij=[pos[v["sample_id"]] for v in group]
                    truth=np.asarray([v["label"] for v in group])
                    if np.sqrt(np.mean((pending[1][ij]-truth)**2))>=np.sqrt(np.mean((active[ij]-truth)**2))-1e-12:
                        break
                else:
                    current_y,active,spec=pending
                    chain.append(spec)
                    pending=None
                    selected=True
                orders.append(selected)
                for row in serve:
                    if row["label"] is not None:
                        raise ValueError("服务输入带有答案")
                    key=row["sample_id"]
                    stream[key]=float(active[pos[key]])
                for name,windows in suites.items():
                    if r not in windows:
                        continue
                    new_ids,data=zip(*windows[r])
                    data=np.asarray(data)
                    values=model.predict(data)
                    for tree in chain:
                        values=np.maximum(0,values+.5*tree_prediction(tree,data))
                    aux[name].update(zip(new_ids,map(float,values),strict=True))
            pools["development"][arm]=stream
            for name,values in aux.items():
                pools[name][arm]=values
            builds[arm],events[arm],cost[arm]=history,orders,{"fit_seconds":spent,"fits":len(history),
                       "structure_rows":0 if arm=="constant" else (len(rows) if arm=="all-tree" else int(sparse.sum())),
                       "value_rows":{"original-tree":int(sparse.sum()),"all-tree":len(rows),"constant":len(rows)}.get(arm,int((~sparse).sum()))}
    pools["development"].update(parent=delivered(old),mature=delivered(paper))
    if cohort=="pronostia":
        pools["development"]["simple"]=delivered(fixed.parent/"simple")
        reference=read(OUT/cohort/"transfer"/"issued.json")["predictions"]
        for name,k in (("offset8","unseen8"),("offset16","previous16"),("kaist","kaist")):
            for arm,oldarm in (("parent","old"),("mature","mature"),("simple","simple")):
                pools[name][arm]=reference[k][oldarm]
            if max(abs(v-reference[k]["stump"][i]) for i,v in pools[name]["original-tree"].items())>1e-8:
                raise ValueError("原浅树在移用时没有复现")
    dev=read(OUT/cohort/"issued.json")["predictions"]["stump"]
    if max(abs(v-dev[k]) for k,v in pools["development"]["original-tree"].items())>1e-8:
        raise ValueError("原浅树开发流不一致")
    dest=OUT/cohort/"leaf-estimation"
    sealed=dest/"issued.json"
    atomic_json(sealed,{"source":checksum,"sparse":digest(sorted(ids)),"predictions":pools,"trees":builds,"accepted":events})
    truth={r.sample_id:r.label for r in pkg.validation} if cohort=="bearing" else remaining_minutes("full-test-manifest.json")
    answers={"development":truth}
    devices["development"]={i:r["unit_id"] for i,r in state.items()}
    if cohort=="pronostia":
        answers.update({k:window_answers(*meta) for k,meta in labels.items()})
        answers["kaist"],_=transfer_truth("bearing")
    metrics={}
    for name,streams in read(sealed)["predictions"].items():
        metrics[name]=summary("bearing",answers[name],streams,devices[name])
        print(cohort,name,{k:round(s["score"],2) for k,s in metrics[name].items()})
    atomic_json(dest/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"suites":metrics,"cost":cost})


if __name__=="__main__":
    for c in ("bearing","pronostia"):
        run(c)
