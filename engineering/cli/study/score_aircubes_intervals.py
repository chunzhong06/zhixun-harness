"""按作者十分钟区间配对四台原始读数与参考端记录，确认发放后才返回该区间浓度。"""
from hashlib import sha256
import json
import joblib
import numpy as np
import pandas as pd

from score_aircubes import (DATA,DEST as PRIOR,START,known,station,LocalAcceptance,LocalStore,atomic_json,digest,
                            AirCalibration,BASE,PRIMARY as OLD)
from harness.models.aircubes_pair import PairedMaintenance,PRIMARY,ANCHOR

DEST=PRIOR.parent/"aircubes-intervals"


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    source=pd.read_csv(known("reference-first.csv",manifest),usecols=["timestamp","NO2_ppb"])
    reference=source.set_index("timestamp")["NO2_ppb"]
    if reference.index.has_duplicates:raise ValueError("参考时刻重复")
    # 连同参考缺失的位置一起，逐行核对作者首期R脚本选择的区间。
    aligned=reference.copy()
    aligned.index=aligned.index-600
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    reports={};events={};quantities={};identities={}
    for site in ("AC009","AC010","AC011","AC012"):
        t,x,y=station(site,aligned,manifest)
        start=3811 if site=="AC009" else 85
        author=source.iloc[start:23947]
        if not np.array_equal(author["timestamp"],t+600) or not np.allclose(y,author["NO2_ppb"],equal_nan=True):
            raise ValueError("本地配对与作者实际使用的逐行参考不一致")
        fit=(t+600<=START)&(t<START)&np.isfinite(x).all(1)&np.isfinite(y)
        service=(t>=START)&np.isfinite(x).all(1)
        model=AirCalibration().fit(x[fit],y[fit])
        path=DEST/(site+".joblib");joblib.dump(model,path,compress=3)
        identities[site]=sha256(path.read_bytes()).hexdigest()
        model=joblib.load(path);stamps=t[service];xx=x[service]
        bank=model.predict(xx)
        state=PairedMaintenance();events[site]=[]
        edges=np.r_[0,np.flatnonzero(np.diff(stamps//21600))+1,len(stamps)]
        for a,b in zip(edges[:-1],edges[1:],strict=True):
            out=state.predict(bank[a:b],xx[a:b,2])
            payload={"仪器":site,"区间起点":stamps[a:b].tolist(),"已取参考":state.old.count,
                     "预测":{k:v.tolist() for k,v in out.items()}}
            req=digest(payload)
            if not receiver.submit(req,payload)["accepted"]:raise ValueError("区间预测未交付")
            events[site].append(req)
            end=int(stamps[b-1])+600
            if b<len(stamps) and end%21600==0:
                if end>stamps[b]:raise ValueError("未到参考时间就返回答案")
                val=reference.get(end,np.nan)
                if np.isfinite(val):state.update(float(val))
        all_t=[];cols={}
        for req in events[site]:
            pkt=json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
            if digest(pkt)!=req:raise ValueError("区间交付内容变化")
            all_t.extend(pkt["区间起点"])
            for k,v in pkt["预测"].items():cols.setdefault(k,[]).extend(v)
        if all_t!=stamps.tolist():raise ValueError("区间发放不完整")
        truth=reference.reindex(np.asarray(all_t)+600).to_numpy();ok=np.isfinite(truth)
        reports[site]={n:float(np.sqrt(np.mean((np.array(v)[ok]-truth[ok])**2))) for n,v in cols.items()}
        quantities[site]={"实发":len(all_t),"可评分":int(ok.sum()),"已回标":state.old.count}
    means={k:float(np.mean([r[k] for r in reports.values()])) for k in cols}
    fixed=min((k for k in means if k not in (PRIMARY,OLD)),key=means.get)
    paper=fixed
    gain=1-means[PRIMARY]/means[fixed]
    worst=max(r[PRIMARY] for r in reports.values());fw=max(r[fixed] for r in reports.values())
    d={"现场":PRIMARY,"现场误差":means[PRIMARY],"固定":fixed,"固定误差":means[fixed],"论文":paper,"论文误差":means[paper],
       "降幅":gain,"最差仪器":worst,"固定最差":fw,"进入独立":gain>=.02 and worst<=fw*1.01}
    atomic_json(DEST/"scores.json",{"decision":d,"scores":reports,"means":means,"requests":events,
        "服务与回标":quantities,"源模型SHA256":identities})
    print(json.dumps({"作者实际区间配对":d,"论文四模型保持":means[BASE[-1]+"/保持"],"先前最强固定同配对重训":means[ANCHOR],
        "上版现场同配对重训":means[OLD],"服务与回标":quantities},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("作者区间配对已实发")
        run()
