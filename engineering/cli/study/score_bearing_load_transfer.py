"""以相同载荷折算规则贯通三个台架，从单行运行入口实发预测后才计算寿命误差。"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from compare_fixed_tree_references import OUT
from compare_group_objectives import ROOT, atomic_json, official, read, summary
from read_ferrara_windows import FOLDER
from score_ferrara_calibration import DAYS, LOAD, age, predict_clock
from score_life_training_weights import cohorts

DEST=OUT/"load-life"
SETTINGS={"femto":((4.,1800),(4.2,1650),(5.,1500)), "xjtu":((12.,2100),(11.,2250),(10.,2400))}


def condition(site,unit):
    if site=="ferrara":
        return LOAD[unit],2400.
    family=int(unit.removeprefix("Bearing").split("_")[0])
    if family not in (1,2,3):
        raise ValueError("工况未登记")
    return SETTINGS[site][family-1]


def collect(site, times):
    return [{"device":unit,"life_minutes":float(time),"radial_load_kn":condition(site,unit)[0],"rpm":condition(site,unit)[1]}
            for unit,time in sorted(times.items())]


def issue(name,site,history,observations,answers,known=None):
    work=DEST/name
    archive=work/"completed.json"
    atomic_json(archive,history)
    requests=[]
    p={}
    devices={}
    life=np.array([r["life_minutes"] for r in history])
    load=np.array([r["radial_load_kn"] for r in history])
    rpm=np.array([r["rpm"] for r in history])
    for key,unit,t in observations:
        if unit in {r["device"] for r in history} or key in devices:
            raise ValueError("新设备与已完成历史相交或重复发放")
        service_load,speed=condition(site,unit)
        inputs={"radial_load_kn":service_load,"rpm":speed,"elapsed_minutes":t}
        requests.append({"inputs":inputs})
        q=predict_clock(life,load,rpm,service_load,speed,t)
        same=(load==service_load)&(rpm==speed)
        q["matched-median"]=max(0,float(np.median(life[same] if same.any() else life))-t)
        q["load-only-half"]=float(np.median(life*(rpm/speed)*(load/service_load)**3))/2
        for arm,v in q.items():
            p.setdefault(arm,{})[key]=float(v)
        devices[key]=unit
    start=perf_counter()
    command=[sys.executable,"-X","utf8","-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),
             "bearing-load-predict","--history",str(archive)]
    inputs="\n".join(map(json.dumps,requests))+"\n"
    response=subprocess.run(command,input=inputs,encoding="utf-8",capture_output=True,check=True)
    duration=perf_counter()-start
    actual={k:json.loads(line)["outputs"]["value"] for k,line in zip(devices,response.stdout.splitlines(),strict=True)}
    if max(abs(actual[k]-p["load-cubic"][k]) for k in actual)>1e-8:
        raise ValueError("真实单行接口和独立折算不一致")
    p["load-cubic"]=actual
    if known:
        p.update(known)
    path=work/"issued.json"
    atomic_json(path,{"predictions":p,"inputs":dict(zip(devices,requests,strict=True)),"device":devices,
                "history_sha256":sha256(archive.read_bytes()).hexdigest(),"cli_seconds":duration})
    emitted=read(path)["predictions"]
    table=summary("bearing",answers(),{"parent":emitted["countdown-median"],**emitted},devices)
    atomic_json(work/"scores.json",{"arms":table,"issued_sha256":sha256(path.read_bytes()).hexdigest()})
    return {k:[round(v["score"],4),round(v["worst_group"],4)] for k,v in table.items() if k!="parent"}


def main():
    pkg,_,batches=official("bearing")
    training={}
    for r in pkg.train:
        training[r.unit_id]=max(training.get(r.unit_id,0),r.time_end-1)
    constants=float(np.median([r.label for r in batches[0].samples if r.role=="fit"]))
    public={r.sample_id:r for r in pkg.validation}
    prior=read(ROOT/"data/public/pronostia-sampled/manifest.json")
    femto={r["unit"]:(r["end"]-1)/6 for r in prior}
    results={}
    for c in cohorts("bearing",pkg):
        name,keys,_,groups,answer,base=c
        if name not in ("bearing","roles","new-xjtu","pronostia"):
            continue
        old=read(base/"issued.json")["predictions"]
        refs={k:old[k] for k in ("current","paper","fixed-tree","tree") if k in old}
        refs["source-median"]=dict.fromkeys(keys,constants)
        site="femto" if name=="pronostia" else "xjtu"
        bank=collect(site,femto if site=="femto" else training)
        obs=[]
        for k in keys:
            unit=groups[k].split("/")[-1]
            if name=="pronostia":
                t=(int(k.split("/")[-1])-1)/6
            elif name=="new-xjtu":
                t=int(Path(k).stem)-1
            else:
                t=public[k.split("/")[-1]].time_end-1
            obs.append((k,unit,t))
        results[name]=issue(name,site,bank,obs,answer,refs)
    ends={u:read(FOLDER/(u+".json"))["terminal_index"] for u in DAYS}
    history=collect("ferrara",{u:age(u,ends[u]) for u in ("E1","E2","E3")})
    obs=[(f"{u}/{j}",u,age(u,j)) for u in ("E4","E5","E6") for j in sorted({*range(1,ends[u]+1,60),ends[u]})]
    truth=lambda:{k:age(unit,ends[unit])-t for k,unit,t in obs}
    results["ferrara"]=issue("ferrara","ferrara",history,obs,truth)
    atomic_json(DEST/"scores.json",results)
    for k,t in results.items():
        print(k,{a:t[a] for a in ("constant","matched-median","countdown-median","load-cubic","load-only-half")})


if __name__=="__main__":
    main()
