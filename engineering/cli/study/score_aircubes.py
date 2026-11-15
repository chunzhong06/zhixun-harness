"""四台NO₂探头先发每个可测十分钟，接收6小时槽末读数后更新，再由评价侧核算。"""
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"engineering/src"),str(ROOT/"engineering/cli/data")]
from prepare_aircubes import DATA
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json
from harness.models.aircubes import RAW,BASE,MODES,PRIMARY,FROZEN,HumidityHistory,AirCalibration,StationMaintenance

DEST=ROOT/"results/experiments/local-defaults/whole-maintenance/aircubes-first"
START=int(pd.Timestamp("2018-09-01",tz="UTC").timestamp())


def known(name,manifest):
    path=DATA/name
    if sha256(path.read_bytes()).hexdigest()!=manifest["原件"][name]["sha256"]:
        raise ValueError("空气监测原件身份不符")
    return path


def station(name,reference,manifest):
    frame=pd.read_csv(known(name+".csv",manifest),usecols=["timestamp",*RAW])
    t=frame["timestamp"].to_numpy(dtype=np.int64)
    if np.any(np.diff(t)!=600) or np.any(t%600!=0):raise ValueError("监测时刻不连续")
    x=frame[list(RAW)].to_numpy()
    history=HumidityHistory()
    x=np.column_stack((x,[history.next(val) for val in x[:,3]]))
    y=reference.reindex(t).to_numpy(float)
    return t,x,y


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    ref=pd.read_csv(known("reference-first.csv",manifest),usecols=["timestamp","NO2_ppb"]).set_index("timestamp")["NO2_ppb"]
    if ref.index.has_duplicates:raise ValueError("同一个参考时刻重复")
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    reports={};events={};sources={};budget={};weights={}
    for site in ("AC009","AC010","AC011","AC012"):
        t,x,y=station(site,ref,manifest)
        valid_x=np.isfinite(x).all(1)
        train=(t<START)&valid_x&np.isfinite(y)
        model=AirCalibration().fit(x[train],y[train])
        path=DEST/(site+".joblib");joblib.dump(model,path,compress=3)
        weights[site]=sha256(path.read_bytes()).hexdigest();model=joblib.load(path)
        service=(t>=START)&valid_x
        out_t=t[service];out_x=x[service]
        # 源模型独立逐行，预求值只含当前输入。维护权重按实际批次更新。
        frozen=model.predict(out_x)
        state=StationMaintenance()
        bounds=np.r_[0,np.flatnonzero(np.diff(out_t//21600))+1,len(out_t)]
        seq=[]
        for a,b in zip(bounds[:-1],bounds[1:],strict=True):
            cols,features,bank=state.predict(frozen[a:b],out_x[a:b,2])
            payload={"仪器":site,"时间戳":out_t[a:b].tolist(),"已回数":state.count,
                     "预测":{n:v.tolist() for n,v in cols.items()}}
            key=digest(payload)
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("仪器预测未交接")
            seq.append(key)
            last=int(out_t[b-1])
            if b<len(out_t) and last%21600==21000:
                answer=ref.get(last,np.nan)
                if np.isfinite(answer):
                    state.feedback(features[-1],bank[-1],float(answer))
        aggregate={};ordered=[]
        for key in seq:
            payload=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(payload)!=key:raise ValueError("接收后预测改变")
            ordered.extend(payload["时间戳"])
            for n,v in payload["预测"].items():aggregate.setdefault(n,[]).extend(v)
        if ordered!=out_t.tolist():raise ValueError("缺少已发时间")
        truth=ref.reindex(ordered).to_numpy(float)
        mask=np.isfinite(truth)
        reports[site]={n:float(np.sqrt(np.mean((np.asarray(v)[mask]-truth[mask])**2))) for n,v in aggregate.items()}
        events[site]=seq;sources[site]=int(train.sum())
        budget[site]={"服务时刻":int((t>=START).sum()),"实发":len(ordered),"可评分":int(mask.sum()),"已回标":state.count}
    means={n:float(np.mean([v[n] for v in reports.values()])) for n in aggregate}
    fixed=min((n for n in means if n!=PRIMARY),key=means.get)
    # 每项固定校正以作者实际选出的RLM_22、RF_6或其等权为底层工具。
    paper=min((n for n in means if n!=PRIMARY),key=means.get)
    a=max(v[PRIMARY] for v in reports.values());f=max(v[fixed] for v in reports.values())
    gain=1-means[PRIMARY]/min(means[n] for n in (fixed,paper))
    d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],
       "降幅":gain,"最差仪器":a,"固定最差":f,"进入独立":gain>=.02 and a<=1.01*f}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":reports,"requests":events,
        "各仪器训练数":sources,"服务与回标":budget,"源模型SHA256":weights})
    print(json.dumps({"长期NO2十分钟交付":d,"服务与回标":budget,"源模型等权":means[BASE[-1]+"/保持"]},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("首次同站开发已有交付")
        run()
