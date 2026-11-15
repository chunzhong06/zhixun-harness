"""训练折只含源设备；两个候选的目标交付结果在选定工具之后才读入。"""
import json
from collections import defaultdict

import h5py
import numpy as np

from score_ncmapss_reference import RAW, MEMBER_SHA, file_hash, flights
from score_engine_native_condition import EARLIER, PRIOR, official, cohorts, summary, read
from score_ds02_forward import RAW2, read_flights
from harness.contracts import digest
from harness.models.engine_source_choice import source_choice, split_official
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = EARLIER.parent/"engine-source-tool-choice"


def transfer(endpoint,site,previous,old,choices,arms):
    # 此处只传封存预测和已选工具，系统看不到外部答案。
    chosen = choices["choice"]
    p = {k:old["predictions"][v] for k,v in arms.items()}
    p["selected"] = p[chosen]
    payload = {"ids":old["ids"],"predictions":p,"choice":chosen,"source_scores":choices["score"],"prior":previous}
    req = digest([site,payload])
    ack = endpoint.submit(req,payload)
    if not ack["accepted"] or ack["request_id"]!=req:
        raise RuntimeError("源端所选工具交接失败")
    return req,read(endpoint.output/(req+".json"))["payload"]["predictions"]


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists() or file_hash(RAW)!=MEMBER_SHA or file_hash(RAW2)!=read(RAW2.with_suffix(".source.json"))["sha256"]:
            raise ValueError("已完成或源数据变更")
        pkg,_,batches = official("engine")
        fit_ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        source = defaultdict(list)
        for r in pkg.train:
            source[r.unit_id].append(r)
        rows,early,uniform = {},{},{}
        for unit,points in source.items():
            selected = [r for r in points if r.sample_id in fit_ids]
            rows[unit] = (np.array([r.values for r in selected]),np.array([r.label for r in selected]))
            early[unit] = np.array([r.values for r in sorted(points,key=lambda r:r.time_end)[:8]])
            uniform[unit] = rows[unit][0]
        if len(rows)!=496 or any(len(x)!=8 for x,y in rows.values()):
            raise ValueError("官方源设备清单不符")
        choices = {"official":source_choice("official",rows,early,uniform,split_official(rows))}
        with h5py.File(RAW) as base:
            dev,_ = flights(base,"dev")
        rows = {u:(np.array([r[1] for r in dev[u]]),np.array([r[2] for r in dev[u]])) for u in (4,5,6)}
        early = {u:x[:8] for u,(x,y) in rows.items()}
        uniform = {u:x[np.linspace(0,len(x)-1,8).round().astype(int)] for u,(x,y) in rows.items()}
        choices["flight"] = source_choice("flight",rows,early,uniform,[[u] for u in rows])
        atomic_json(DEST/"source.json",choices)
        choices = read(DEST/"source.json")
        endpoint = LocalAcceptance()
        endpoint.output = DEST/"accepted"
        results = {}
        saved = read(PRIOR/"scores.json")
        for site,keys,_,groups,answers,_ in cohorts("engine",pkg):
            req = saved[site]["request"]
            old = read(PRIOR/"accepted"/(req+".json"))["payload"]
            if list(keys)!=old["ids"] or digest([site,old])!=req:
                raise ValueError("官方冻结工具输入变化")
            name,p = transfer(endpoint,site,req,old,choices["official"],{"early":"early-tree","half":"raw-forest-half"})
            scores = summary("engine",answers(),{"parent":p["half"],**p},groups)
            results[site] = {"request":name,"scores":{k:v["score"] for k,v in scores.items() if k!="parent"}}
        for part,directory in (("DS01",EARLIER.parent/"ncmapss-early-reference"),("DS02",EARLIER.parent/"ncmapss-ds02-forward")):
            stored = read(directory/"scores.json")
            with h5py.File(RAW if part=="DS01" else RAW2) as dataset:
                for site,req in stored["requests"].items():
                    old = read(directory/"accepted"/(req+".json"))["payload"]
                    if digest([site,old])!=req:
                        raise ValueError("飞行机队冻结预测已改变")
                    run = part+"-"+site
                    name,p = transfer(endpoint,run,req,old,choices["flight"],{"early":"early","half":"half"})
                    targets = read_flights(dataset,"test" if site in ("service","test") else "dev",answers=True)
                    y = np.array([targets[tuple(k)] for k in old["ids"]])
                    results[run] = {"request":name,"scores":{k:float(np.sqrt(np.mean((y-v)**2))) for k,v in p.items()}}
        atomic_json(DEST/"scores.json",{"choices":{k:v["choice"] for k,v in choices.items()},"cases":results})
        print(json.dumps({"源设备等权误差":{k:v["score"] for k,v in choices.items()},
                          "实际服务误差":{k:v["scores"] for k,v in results.items()}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
