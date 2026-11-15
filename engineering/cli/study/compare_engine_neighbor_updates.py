"""把按台检索的训练参照接到已交付发动机权重，冻结后测采用决定与真实服务分片。"""
from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import archived,source_window
from compare_group_objectives import BASE,ROOT,_trainer,atomic_json,delivered,digest,official,read,reference_folders,remaining_cycles,summary
from harness.models.device_neighbors import nearest

DEST=BASE/"engine-neighbor-updates"


def compare(cohort):
    task="engine"
    pkg,checksum,batches=official(task)
    ids={s.sample_id for s in batches[0].samples if s.role=="fit"}
    train=[r for r in pkg.train if r.sample_id in ids]
    x,y=np.asarray([r.values for r in train]),np.asarray([r.label for r in train])
    units=np.asarray([r.unit_id for r in train])
    fixed,paper=reference_folders(cohort,"mature")
    current=BASE/"live-first"/("public" if cohort==task else "")/cohort
    points=read(fixed/"state.json")["samples"]
    rows=[r for r in points.values() if r["role"]!="fit"]
    keys=[r["sample_id"] for r in rows]
    idx={i:j for j,i in enumerate(keys)}
    query=np.asarray([r["values"] for r in rows])
    m,scale=x[:,:21].mean(axis=0),np.maximum(x[:,:21].std(axis=0),1e-12)
    bank=np.clip((x[:,:21]-m)/scale,-10,10)
    query_z=np.clip((query[:,:21]-m)/scale,-10,10)
    paths={"fixed":fixed,"mature":paper,"current":current}
    if cohort!=task:
        paths["simple"]=fixed.parent/"simple"
    preds={a:delivered(p) for a,p in paths.items()}
    logs,artifacts,seconds={},{},{}
    archive,=(ROOT/"data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
        work=Path(tmp)
        p=work/"root.npz"
        p.write_bytes(pkg.checkpoint)
        root=trainer.Model.load(p)
        fourth,identity,records=archived(current,trainer,work,task)
        for step,model,weight in ((0,root,sha256(pkg.checkpoint).hexdigest()),(4,fourth,identity)):
            start=perf_counter()
            base=model.predict(query)
            bias=y-model.predict(x)
            _,target_nn=nearest(bank,y,units,query_z,np.zeros(len(x)),np.zeros(len(query)))
            _,residual_nn=nearest(bank,bias,units,query_z,np.zeros(len(x)),np.zeros(len(query)))
            seconds[step]=perf_counter()-start
            artifacts[step]={"parent":weight,"bank_digest":digest([m.tolist(),scale.tolist(),[r.sample_id for r in train]]),
                             "rows":len(train),"devices":len(set(units))}
            prefix={}
            for rec in records[:step]:
                ack=read(current/rec["submitted"])["payload"]
                if ack["model_id"]!=rec["selected"]:
                    raise ValueError("热身模型未实际交出")
                prefix.update(ack["predictions"])
            for name,trial in (("anchor",base),("blend",.5*(target_nn+base)),("residual",np.maximum(0,base+.5*residual_nn))):
                active=False
                stream=dict(prefix)
                history=[]
                for ordinal in range(step,10):
                    explore,confirm,serve=source_window(points,ordinal)
                    event={"batch":ordinal+1}
                    if name!="anchor" and not active:
                        for role,observed in (("explore",explore),("adopt",confirm)):
                            positions=[idx[r["sample_id"]] for r in observed]
                            yy=np.asarray([r["label"] for r in observed])
                            a,b=[float(np.sqrt(np.mean((v[positions]-yy)**2))) for v in (base,trial)]
                            event[role]=[a,b]
                            if b>=a-1e-12:
                                break
                        else:
                            active=True
                    selected=trial if active else base
                    for row in serve:
                        k=row["sample_id"]
                        if k in stream or row["label"] is not None:
                            raise ValueError("真实服务分片不符")
                        stream[k]=float(selected[idx[k]])
                    history.append({**event,"using":active})
                arm=f"{name}-after-{step}"
                preds[arm],logs[arm]=stream,history
        # 首轮只有一次候选名额。若该候选落选，回到自己的根权重续训，而非借用对照的父版。
        first=logs["blend-after-0"][0]
        if first["using"]:
            preds["choose-first"]=preds["blend-after-0"].copy()
            logs["choose-first"]=[{**v,"route":"neighbors"} for v in logs["blend-after-0"]]
        else:
            state=root
            stream,events={},[]
            for r in range(10):
                explore,confirm,serve=source_window(points,r)
                adopted=False
                if r>0:
                    trial=deepcopy(state)
                    trial.fit(x,y,1)
                    for group in (explore,confirm):
                        xx=np.array([s["values"] for s in group])
                        yy=np.array([s["label"] for s in group])
                        old,new=[float(np.sqrt(np.mean((v.predict(xx)-yy)**2))) for v in (state,trial)]
                        if new>=old-1e-12:
                            break
                    else:
                        state=trial
                        adopted=True
                xx=np.array([s["values"] for s in serve])
                stream.update(zip((s["sample_id"] for s in serve),map(float,state.predict(xx)),strict=True))
                events.append({"batch":r+1,"route":"neighbors-rejected" if r==0 else "fixed-update","adopted":adopted})
            preds["choose-first"],logs["choose-first"]=stream,events
    sealed=DEST/cohort/"issued.json"
    atomic_json(sealed,{"package":checksum,"artifacts":artifacts,"predictions":preds,"history":logs})
    truth={r.sample_id:r.label for r in pkg.validation} if cohort==task else remaining_cycles()
    issued=read(sealed)["predictions"]
    stats=summary(task,truth,{"parent":issued["current"],**issued},{i:s["unit_id"] for i,s in points.items()})
    atomic_json(DEST/cohort/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":stats,"fit_query_seconds":seconds})
    print(cohort,{k:round(v["score"],2) for k,v in stats.items()})


if __name__=="__main__":
    for name in ("engine","phm08"):
        compare(name)
