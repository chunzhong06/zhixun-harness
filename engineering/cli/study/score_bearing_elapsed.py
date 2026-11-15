"""用已知开工时间和源设备完整寿命估计剩余分钟，不读新设备的失效时间作决定。"""
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

from compare_fixed_tree_references import OUT
from compare_group_objectives import ROOT,official,read,atomic_json,remaining_minutes,summary
from score_tree_reference_transfer import suites


def estimates(times,age,current,median):
    if age<0:
        raise ValueError("当前测量早于首次测量")
    remaining=times[times>age]-age
    avg=float(np.mean(remaining)) if len(remaining) else current
    return {"age-mean":avg,"age-median":float(np.median(remaining)) if len(remaining) else current,
            "age-half":.5*(avg+current),"constant-then-current":median if age<=.5*times.max() else current}


def main():
    pkg,_,batches=official("bearing")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    ends=defaultdict(set)
    for r in pkg.train:
        if r.sample_id in ids:
            ends[r.unit_id].add(r.label+r.time_end-1)
    if len(ends)!=9 or any(len(e)!=1 for e in ends.values()):
        raise ValueError("源训练档案的实测时间与失效时间不一致")
    times=np.asarray([next(iter(v)) for v in ends.values()])
    median=float(np.median([r.label for r in pkg.train if r.sample_id in ids]))
    public={r.sample_id:r for r in pkg.validation}
    snapshots=read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
    raw=read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
    kaist_start=datetime.fromisoformat(raw[0]["time"])
    ims_start=datetime.strptime(Path(read(ROOT/"data/public/ims-bearings/sampled.json")["timestamps"][0]).name,"%Y.%m.%d.%H.%M.%S")
    others={name:truth for name,_,_,truth in suites("bearing",False) if name in ("kaist","ims")}
    data=OUT/"teacher-readout/bearing"
    out=OUT/"elapsed-bearing"
    result={}
    for name in ("roles","new-xjtu","pronostia","kaist","ims"):
        path=data/name/"issued.json"
        if sha256(path.read_bytes()).hexdigest()!=read(path.with_name("scores.json"))["issued_sha256"]:
            raise ValueError("已有预测与封存回执不符")
        source=read(path)["predictions"]
        p={k:dict(source[k]) for k in ("current","teacher-ridge")}
        p["tree"]=source["tree"] if "tree" in source else source["fixed-tree"]
        groups={}
        for k,v in p["current"].items():
            if name=="roles":
                r=public[k.split("/")[-1]]
                age=r.time_end-1
                unit=r.unit_id
            elif name=="new-xjtu":
                age=int(Path(k).stem)-1
                unit=Path(k).parent.name
            elif name=="pronostia":
                _,unit,n=k.split("/")
                age=(int(n)-1)/6
            elif name=="ims":
                _,unit,stamp=k.split("/")
                age=(datetime.strptime(stamp,"%Y.%m.%d.%H.%M.%S")-ims_start).total_seconds()/60
            else:
                unit="KAIST"
                stamp=datetime.strptime(Path(k).name,"LogFile_%Y-%m-%d-%H-%M-%S.csv")
                age=(stamp-kaist_start).total_seconds()/60
            groups[k]=unit
            for arm,value in {"median":median,**estimates(times,age,v,median)}.items():
                p.setdefault(arm,{})[k]=value
        sealed=out/name/"issued.json"
        atomic_json(sealed,{"source_lifetimes":times.tolist(),"predictions":p})
        saved=read(sealed)["predictions"]
        if name=="roles":
            truth={k:public[k.split("/")[-1]].label for k in groups}
        elif name=="new-xjtu":
            truth={k:snapshots["ends"][unit]-int(Path(k).stem) for k,unit in groups.items()}
        elif name=="pronostia":
            truth=remaining_minutes("full-test-manifest.json")
        else:
            truth=others[name]()
        t=summary("bearing",truth,{"parent":saved["current"],**saved},groups)
        atomic_json(sealed.with_name("scores.json"),{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":t})
        result[name]={k:[r["score"],r["worst_group"]] for k,r in t.items()}
        print(name,{k:round(r["score"],2) for k,r in t.items()})
    atomic_json(out/"scores.json",result)


if __name__=="__main__":
    main()
