"""把本地风速作为双方可见的新观测，每半秒核对补偿版已发布的预测。"""
from hashlib import sha256
import json
import joblib
import numpy as np

from score_vapor_environment import (DATA,DEST as PRIOR,LocalAcceptance,LocalStore,atomic_json,digest,load_verified,
                                     measurements,deliver,score,OLD_CORRECTION,OLD_MIX,PRIMARY as ENV_SELECTION)
from score_vapor_correction import clock_groups
from harness.models.vapor_environment import CHANNELS
from harness.models.vapor_airflow import WIND,READERS,PRIMARY,PREVIOUS,AirHistory,AirReadout

DEST=PRIOR.parent/"vapor-airflow"


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    tools,old=load_verified(PRIOR)
    records=[]
    for i in manifest["源日"]:
        raw,y,wind=measurements(i,manifest,WIND)
        limits=np.r_[0,np.cumsum([len(v) for v in raw])]
        clock=clock_groups(DATA/manifest["记录"][str(i)]["文件"],limits)
        sensor=AirHistory(raw[:20],tools["scale"])
        rows=[sensor.next(v,w,t) for v,w,t in zip(raw,wind,clock,strict=True)]
        records.append(([np.stack(a) for a in zip(*rows,strict=True)],y/tools["cut"]))
    tools["airflow"]=AirReadout().fit(records)
    release=DEST/"release.joblib"
    joblib.dump(tools,release,compress=3)
    checksum=sha256(release.read_bytes()).hexdigest()
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    all_y=[];total={};scores={};requests={};spent={}
    for i in manifest["开发日"]:
        _,_,both=measurements(i,manifest,(*CHANNELS,*WIND))
        env=[v[:,:2] for v in both];wind=[v[:,2:] for v in both]
        seq,y,spent[i]=deliver(i,manifest,old,release,checksum,receiver,clocked=True,environment=env,wind=wind)
        requests[i]=seq; cols={};rows=[]
        for key in seq:
            p=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(p)!=key:raise ValueError("风速预测在交接后变化")
            rows.extend(p["窗口"])
            for n,v in p["阈值倍数"].items():cols.setdefault(n,[]).extend(v)
        if rows!=list(range(20,len(y))):raise ValueError("服务窗口不完整")
        truth=y[rows]/tools["cut"]>=1;all_y.extend(truth)
        scores[i]=score(truth,cols)
        for n,v in cols.items():total.setdefault(n,[]).extend(v)
    means=score(all_y,total)
    fixed=max((n for n in means if n not in (PRIMARY,ENV_SELECTION,OLD_CORRECTION,OLD_MIX)),key=lambda n:means[n]["F1"])
    paper=max((n for n in means if n not in (*READERS,PRIMARY,ENV_SELECTION,OLD_CORRECTION,OLD_MIX)),key=lambda n:means[n]["F1"])
    gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
       "论文F1":means[paper]["F1"],"差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":scores,"requests":requests,
        "源权重SHA256":checksum,"窗口":len(all_y),"各场已回参考":spent})
    print(json.dumps({"同层风速与局部更新速度":d,"同一入口温湿固定F1":means[PREVIOUS]["F1"],"总窗数":len(all_y)},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("气流版已有完整交付")
        run()
