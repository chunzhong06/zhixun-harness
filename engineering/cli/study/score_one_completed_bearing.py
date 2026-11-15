"""在相同现场用一台已终止的设备挑固定模型，然后给不同设备整段交付。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from compare_fixed_tree_references import OUT
from compare_group_objectives import ROOT,official,read,atomic_json,remaining_minutes,summary


def main():
    pkg,_,batches=official("bearing")
    training=[r.label for r in batches[0].samples if r.role=="fit"]
    public={r.sample_id:r for r in pkg.validation}
    external=read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
    options=("current","median","mean","tree","teacher-ridge")
    study=OUT/"one-completed-bearing"
    results={}
    for name in ("roles","pronostia","new-xjtu"):
        folder=OUT/"teacher-readout/bearing"/name
        file=folder/"issued.json"
        if sha256(file.read_bytes()).hexdigest()!=read(folder/"scores.json")["issued_sha256"]:
            raise ValueError("冻结对照身份不符")
        old=read(file)["predictions"]
        groups=defaultdict(list)
        pred={}
        sequence={}
        for key in old["current"]:
            if name=="roles":
                k=key.split("/")[-1]
                unit=public[k].unit_id
                when=public[k].time_end
            elif name=="pronostia":
                k=key
                _,unit,order=k.split("/")
                when=int(order)
            else:
                k=key
                unit=Path(k).parent.name
                when=int(Path(k).stem)
            values={"current":old["current"][key],"median":float(np.median(training)),"mean":float(np.mean(training)),
                    "tree":old["tree" if "tree" in old else "fixed-tree"][key],"teacher-ridge":old["teacher-ridge"][key]}
            if k in pred:
                if pred[k]!=values:
                    raise ValueError("同一行在角色轮换中存在不同预测，不能合并")
            else:
                pred[k]=values
                groups[unit].append(k)
                sequence[k]=when
        if name=="roles":
            answers={k:public[k].label for k in pred}
        elif name=="pronostia":
            answers=remaining_minutes("full-test-manifest.json")
        else:
            answers={k:external["ends"][Path(k).parent.name]-sequence[k] for k in pred}
        forecast={}
        groups_service={}
        ids={}
        events=[]
        for unit,history in sorted(groups.items()):
            ordered=sorted(history,key=sequence.get)
            for budget in ("five","all"):
                visible=ordered if budget=="all" else [ordered[i] for i in np.linspace(0,len(ordered)-1,min(5,len(ordered))).round().astype(int)]
                err={a:sum((pred[k][a]-answers[k])**2 for k in visible) for a in options}
                arms={a:a for a in options}
                for label,pool in (("select",options),("median-or-current",("current","median")),("tree-or-current",("current","tree"))):
                    arms[label]=min(pool,key=err.get)
                events.append({"completed":unit,"budget":budget,"visible":visible,"choices":arms})
                for service,keys in groups.items():
                    if service==unit:
                        continue
                    for k in keys:
                        key=f"{unit}::{k}"
                        groups_service[key]=service
                        ids[key]=k
                        for arm,choice in arms.items():
                            forecast.setdefault(f"{budget}-{arm}",{})[key]=pred[k][choice]
        sealed=study/name/"issued.json"
        atomic_json(sealed,{"events":events,"predictions":forecast})
        saved=read(sealed)["predictions"]
        truth={k:answers[i] for k,i in ids.items()}
        table=summary("bearing",truth,{"parent":saved["five-current"],**saved},groups_service)
        # 同一台被多个来源校准，另列各校准来源的最坏总体误差，显示领取哪个档案会得出坏决定。
        for arm,p in saved.items():
            by={u:float(np.sqrt(np.mean([(p[k]-truth[k])**2 for k in ids if k.split("::")[0]==u]))) for u in groups}
            table[arm]["per_calibration"]=by
            table[arm]["worst_calibration"]=max(by.values())
        atomic_json(sealed.with_name("scores.json"),{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table})
        results[name]={a:(r["score"],r["worst_group"]) for a,r in table.items()}
        print(name,{a:round(table[a]["score"],2) for a in ("five-current","five-median","five-tree","five-teacher-ridge","five-median-or-current","five-tree-or-current","five-select","all-select")})
    atomic_json(study/"scores.json",results)


if __name__=="__main__":
    main()
