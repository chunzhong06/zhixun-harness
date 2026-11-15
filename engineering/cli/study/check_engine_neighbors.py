"""按设备留出，测单次传感器数据能否借历史工况和相似设备估计剩余寿命。"""
from hashlib import sha256
from math import ceil
from time import perf_counter

import numpy as np

from compare_engine_regimes import assign,design,fit
from compare_group_objectives import BASE,atomic_json,digest,official,read
from harness.models.device_neighbors import nearest

OUT=BASE/"engine-neighbor-holdout"


class ZeroReference:
    """同一零起点仅用于复用按工况拟合函数，目标和统计均来自外层训练设备。"""
    @staticmethod
    def predict(x):
        return np.zeros(len(x))


def main():
    package,checksum,batches=official("engine")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=list(package.train)
    groups=sorted({r.unit_id.split(":")[0] for r in rows})
    selected=set()
    for group in groups:
        units=sorted({r.unit_id for r in rows if r.unit_id.split(":")[0]==group},key=digest)
        selected.update(units[:ceil(.2*len(units))])
    train=[r for r in rows if r.unit_id not in selected and r.sample_id in ids]
    test=[r for r in rows if r.unit_id in selected]
    x,y=np.array([r.values for r in train]),np.array([r.label for r in train])
    units=np.array([r.unit_id for r in train])
    if set(units)&selected or len(groups)!=4:
        raise ValueError("设备分组错误")
    q=np.asarray([r.values for r in test])
    streams,elapsed={},{}
    for recipe in ("global","regime"):
        start=perf_counter()
        spec=fit(ZeroReference(),x,y,recipe)
        bx,qx=design(x,spec),design(q,spec)
        streams[recipe+"-ridge"]=np.maximum(0,qx@spec["coefficients"])
        keys,query_keys=np.zeros(len(x),dtype=int),np.zeros(len(q),dtype=int)
        if recipe=="regime":
            keys,query_keys=[assign(data[:,21:],spec["setting_mean"],spec["setting_scale"],spec["clusters"]) for data in (x,q)]
        a,b=nearest(bx[:,:-1],y,units,qx[:,:-1],keys,query_keys)
        streams[recipe+"-observations"],streams[recipe+"-devices"]=a,b
        elapsed[recipe]=perf_counter()-start
    if not all(np.isfinite(p).all() for p in streams.values()):
        raise ValueError("出现非有限预测")
    issued=OUT/"issued.json"
    atomic_json(issued,{"source":checksum,"fit_ids":digest([r.sample_id for r in train]),"held":sorted(selected),
                        "ids":[r.sample_id for r in test],"predictions":{k:v.tolist() for k,v in streams.items()}})
    saved=read(issued)
    truth=np.array([r.label for r in test])
    uid=np.array([r.unit_id for r in test])
    subset=np.array([u.split(":")[0] for u in uid])
    table={}
    for arm,p in saved["predictions"].items():
        errors=(np.asarray(p)-truth)**2
        dev={u:float(np.sqrt(errors[uid==u].mean())) for u in selected}
        table[arm]={"score":float(np.sqrt(errors.mean())),"subsets":{s:float(np.sqrt(errors[subset==s].mean())) for s in groups},
                    "worst_device":max(dev.values()),"devices":dev}
        print(arm,round(table[arm]["score"],2),{k:round(v,2) for k,v in table[arm]["subsets"].items()},round(max(dev.values()),2))
    atomic_json(OUT/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"train_devices":len(set(units)),
                                 "held_devices":len(selected),"held_rows":len(test),"elapsed_seconds":elapsed,"arms":table})


if __name__=="__main__":
    main()
