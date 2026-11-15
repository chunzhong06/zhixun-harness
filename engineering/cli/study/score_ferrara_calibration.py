"""独立单径向台架的封存预测，同时区分单测量和带已知工况、机龄两种输入边界。"""
from dataclasses import replace
from hashlib import sha256
from time import process_time

import numpy as np
from sklearn.base import clone

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import atomic_json, official, read, remaining_minutes, summary
from compare_public_selection import match_bearing_window
from read_ferrara_windows import FOLDER
from score_completed_calibration import five, ridge
from unseen_sources import bearing_data

DEST = OUT/"ferrara"
LOAD = {"E1":4., "E2":4., "E3":4., "E4":3., "E5":4.7, "E6":5.}
DAYS = {"E1":17., "E2":6.9, "E3":8.3, "E4":20.9, "E5":6., "E6":1.8}


def age(device, number, transition=452):
    intervals = number-1
    if device == "E4":
        return 60*min(intervals,transition)+5*max(0,intervals-transition)
    return 5*intervals


def predict_clock(lives, loads, speeds, service_load, service_speed, elapsed):
    """历史时长均来自已失效设备；查询参数只有当前台的已知工况与已运行分钟。"""
    adjusted = lives*(speeds/service_speed)*(loads/service_load)**3
    arms = {"constant": np.mean(lives)/2, "constant-median": np.median(lives)/2,
            "countdown-mean": max(np.mean(lives)-elapsed,0), "countdown-median": max(np.median(lives)-elapsed,0),
            "load-linear": max(np.median(lives*(speeds/service_speed)*(loads/service_load))-elapsed,0),
            "load-cubic": max(np.median(adjusted)-elapsed,0)}
    for name, values in (("plain",lives),("cubic",adjusted)):
        remaining = values[values>elapsed]-elapsed
        arms[name+"-survive-zero"] = float(np.mean(remaining)) if len(remaining) else 0.
        arms[name+"-survive-median"] = float(np.mean(remaining)) if len(remaining) else float(np.median(values))
    return arms


def seal(out, predictions, get_truth, devices, metadata):
    f=out/"issued.json"
    atomic_json(f,{"predictions":predictions,"conditions":metadata})
    p=read(f)["predictions"]
    baseline=next(iter(p))
    truth=get_truth()
    table=summary("bearing",truth,{"parent":p[baseline],**p},devices)
    checks={}
    for group in ("E4","E5-E6"):
        keys=[k for k,v in devices.items() if (v=="E4" if group=="E4" else v in ("E5","E6"))]
        subset={a:{k:v[k] for k in keys} for a,v in p.items()}
        cohort={k:devices[k] for k in keys}
        checks[group]=summary("bearing",truth,{"parent":subset[baseline],**subset},cohort)
        if group=="E4":
            # 相同已发出的估计，最大差额追加到尚不可定位的遗漏时段。此组是目标口径敏感性。
            extra=1440*DAYS["E4"]-age("E4",metadata["E4_terminal"])
            checks["E4-plus-gap"]=summary("bearing",{k:truth[k]+extra for k in keys},{"parent":subset[baseline],**subset},cohort)
    atomic_json(out/"scores.json",{"arms":table,"separate":checks,"issued_sha256":sha256(f.read_bytes()).hexdigest()})
    return {k:[v["score"],v["worst_group"]] for k,v in table.items() if k!="parent"}


def time_trial(records, leave_one=False, transition=452):
    tests = sorted(records) if leave_one else ["E4","E5","E6"]
    predictions, groups, meta = {},{},{}
    meta["E4_terminal"]=records["E4"]["terminal_index"]
    for unit in tests:
        base=sorted(set(records)-{unit}) if leave_one else ["E1","E2","E3"]
        index=records[unit]["terminal_index"]
        lives=np.array([age(i,records[i]["terminal_index"],transition) for i in base],dtype=float)
        meta[unit]={"fit_units":base,"fit_minutes":lives.tolist(),"load_kN":LOAD[unit],"rpm":2400}
        for idx in sorted({*range(1,index+1,60),index}):
            key=f"{unit}/{idx}"
            groups[key]=unit
            p=predict_clock(lives,np.array([LOAD[i] for i in base]),np.full(len(base),2400.),LOAD[unit],2400.,age(unit,idx,transition))
            for k,v in p.items():
                predictions.setdefault(k,{})[key]=float(v)
    def truth():
        return {k:age(u,records[u]["terminal_index"],transition)-age(u,int(k.split("/")[1]),transition) for k,u in groups.items()}
    name=("leave-one" if leave_one else "held-three")+f"-{transition}"
    return seal(DEST/"clock"/name,predictions,truth,groups,meta)


def signals(records):
    pkg,_,batches=official("bearing")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    source=np.array([r.values[17:] for r in pkg.train if r.sample_id in ids])
    mu,sigma=source.mean(axis=0),source.std(axis=0)
    sigma[sigma<1e-12]=1.
    def normalize(x):
        return np.clip((x-mu)/sigma,-10,10)
    six,_,_=bearing_data()
    history_truth=remaining_minutes()
    rows=five([replace(match_bearing_window(r),label=history_truth[r.sample_id]) for vals in six.values() for r in vals])
    banks={"femto":([r.values[17:] for r in rows],[r.label for r in rows],[r.sample_id for r in rows])}
    local_x, local_y, keys=[],[],[]
    for unit in ("E1","E2","E3"):
        seq=sorted(records[unit]["rows"],key=lambda r:r["index"])
        for i in np.linspace(0,len(seq)-1,5).round().astype(int):
            sample=seq[i]
            local_x.append(sample["features"])
            local_y.append(age(unit,records[unit]["terminal_index"])-age(unit,sample["index"]))
            keys.append(f"{unit}/{sample['index']}")
    banks["local"]=(local_x,local_y,keys)
    features, units, sites, timing=[],{},[],[]
    for unit in ("E4","E5","E6"):
        for sample in records[unit]["rows"]:
            key=f"{unit}/{sample['index']}"
            features.append(sample["features"])
            sites.append(key)
            units[key]=unit
            timing.append(age(unit,sample["index"]))
    def truth():
        return {k:age(units[k],records[units[k]]["terminal_index"])-e for k,e in zip(sites,timing,strict=True)}
    teacher=frozen("bearing")[0]["catboost"]
    p={"source-median":dict.fromkeys(sites,float(np.median([r.label for r in pkg.train if r.sample_id in ids])))}
    meta={"E4_terminal": records["E4"]["terminal_index"]}
    for name,(fit,y,keys) in banks.items():
        start=process_time()
        y=np.array(y,dtype=float)
        w,b=ridge(normalize(np.asarray(fit)),y)
        tree=clone(teacher)
        tree.fit(np.asarray(fit),y)
        predictions={"ridge":np.maximum(normalize(np.asarray(features))@w+b,0),"tree":tree_values(tree,"bearing",features),
                     "mean":np.full(len(sites),np.mean(y)), "median":np.full(len(sites),np.median(y))}
        for arm,values in predictions.items():
            p[name+"-"+arm]=dict(zip(sites,values.tolist(),strict=True))
        meta[name]={"train_ids":keys,"cpu_seconds":process_time()-start}
    life=np.array([age(i,records[i]["terminal_index"]) for i in ("E1","E2","E3")])
    for k,unit,t in zip(sites,units.values(),timing,strict=True):
        clock=predict_clock(life,np.full(3,4.),np.full(3,2400.),LOAD[unit],2400.,t)
        for name,v in clock.items():
            p.setdefault("clock-"+name,{})[k]=float(v)
    return seal(DEST/"waves",p,truth,units,meta)


def main():
    records={r["device"]:r for f in sorted(FOLDER.glob("E*.json")) for r in [read(f)]}
    if records.keys()!=LOAD.keys() or any(abs(age(k,r["terminal_index"])/1440-DAYS[k])>.2 for k,r in records.items() if k!="E4"):
        raise ValueError("试验设备或文件时钟与作者时间不符")
    atomic_json(DEST/"clocks.json",{k:{"archive_minutes":age(k,r["terminal_index"]),
                "paper_days":DAYS[k],"difference_minutes":round(1440*DAYS[k]-age(k,r["terminal_index"]),2)}
                for k,r in records.items()})
    result={"wave":signals(records)}
    for loo in (False,True):
        for transition in (452,453):
            result[f"clock-{loo}-{transition}"]=time_trial(records,loo,transition)
    atomic_json(DEST/"scores.json",result)
    print({k:[round(v,1) for v in values] for k,values in result["clock-False-452"].items()})
    print({k:[round(v,1) for v in values] for k,values in result["wave"].items() if not k.startswith("clock")})


if __name__=="__main__":
    main()
