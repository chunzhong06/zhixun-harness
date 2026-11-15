"""按整场校准和回标，同一PID参考只在允许的时点返回。"""
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score,mean_squared_error

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"engineering/src"),str(ROOT/"engineering/cli/data")]
from prepare_vapor_field import DATA
from harness.contracts import digest
from harness.runtime.storage import LocalStore,atomic_json
from harness.runtime.handoff import LocalAcceptance
from harness.models.vapor_field import SENSORS,READERS,POLICIES,PRIMARY,voltage_ratio,ProbeEncoder,OnlineVoltage,ProbePolicy

DEST=ROOT/"results/experiments/local-defaults/whole-maintenance/vapor-field"
NAMES=[f"{reader}/{policy}" for reader in READERS for policy in POLICIES]


def read_flight(number,manifest):
    info=manifest["记录"][str(number)]
    path=DATA/info["文件"]
    if sha256(path.read_bytes()).hexdigest()!=info["sha256"]:raise ValueError("移动探头原件不符")
    frame=pd.read_csv(path,usecols=["time",*SENSORS,"PID-sensor"])
    t=pd.to_datetime(frame["time"],format="mixed")
    clock=(t-t.iloc[0]).dt.total_seconds().to_numpy()
    if np.any(np.diff(clock)<=0):raise ValueError("原采集时钟回退")
    v=frame[list(SENSORS)].to_numpy()
    ref=frame["PID-sensor"].to_numpy()
    if not np.isfinite(ref).all():raise ValueError("同步参考仪出现非有限读数")
    keys=np.floor(clock*2).astype(int)
    bounds=np.r_[0,np.flatnonzero(np.diff(keys))+1,len(keys)]
    windows=keys[bounds[:-1]]
    if not np.array_equal(windows,np.arange(len(windows))):raise ValueError("原场次缺半秒组")
    # 每场初始十秒既是慢通道基线，也是可同等取得的参考仪校准。
    start=keys<20
    base=np.median(voltage_ratio(v[start]),axis=0)
    y=np.add.reduceat(ref,bounds[:-1])/np.diff(bounds)-np.median(ref[start])
    return [v[a:b] for a,b in zip(bounds[:-1],bounds[1:],strict=True)],y,base


def features(record,scale):
    groups,y,baseline=record
    enc=ProbeEncoder(baseline,scale)
    outputs,ctx=[],[]
    for x in groups:
        view,c=enc.next(x);outputs.append(view);ctx.append(c)
    return [np.stack(v) for v in zip(*outputs,strict=True)],np.stack(ctx),y


def train(train_data):
    x,y,baselines=zip(*train_data)
    measured=np.vstack([voltage_ratio(np.vstack(a))-b for a,b in zip(x,baselines,strict=True)])
    scale=np.maximum(np.std(measured,axis=0),.001)
    cutoff=max(.010,float(np.quantile(np.concatenate([v[20:] for v in y]),.75)))
    records=[features(r,scale) for r in train_data]
    def fit(subset):
        target=np.concatenate([records[i][2][20:]/cutoff for i in subset])
        return [OnlineVoltage().fit(np.vstack([records[i][0][j][20:] for i in subset]),target) for j in range(len(READERS))]
    held=[];context=[];truth=[]
    for i,(v,c,target) in enumerate(records):
        models=fit([k for k in range(len(records)) if k!=i])
        for start in range(20,len(c),10):
            sl=slice(start,min(start+10,len(c)))
            preds=np.column_stack([m.predict(a[sl]) for m,a in zip(models,v,strict=True)])
            held.append(preds);context.append(c[sl]);truth.extend(target[sl]/cutoff)
            if sl.stop<len(c):
                j=sl.stop-1
                for m,a in zip(models,v,strict=True):m.update(a[j:j+1],target[j:j+1]/cutoff)
    base=fit(range(len(records)))
    select=ProbePolicy().fit(np.vstack(context),np.vstack(held),np.array(truth))
    return {"scale":scale,"cut":cutoff,"models":base,"policy":select}


def decide(models,policy,v,c):
    bank=np.column_stack([m.predict(x) for m,x in zip(models,v,strict=True)])
    out=dict(zip(NAMES,bank.T,strict=True))
    out[PRIMARY]=policy.predict(c,bank)
    out["全库固定平均"]=bank.mean(1)
    for mode in range(len(POLICIES)):
        pcs=bank[:,mode::len(POLICIES)]
        out["三慢通道均值/"+POLICIES[mode]]=pcs[:,:3].mean(1)
        for j in range(3):
            out[f"固定论文与{SENSORS[j]}各半/{POLICIES[mode]}"]=(pcs[:,j]+pcs[:,5])/2
        out["论文与差分固定各半/"+POLICIES[mode]]=(pcs[:,4]+pcs[:,5])/2
    return bank,out


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    tr,dev=manifest["源日"],manifest["开发日"]
    if tr!=list(range(1,7)) or dev!=list(range(7,14)):raise ValueError("日期划分改变")
    pack=train([read_flight(i,manifest) for i in tr])
    path=DEST/"release.joblib";joblib.dump(pack,path,compress=3)
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    events={};truth={}
    for number in dev:
        pack=joblib.load(path)
        models,policy=pack["models"],pack["policy"]
        v,c,y=features(read_flight(number,manifest),pack["scale"])
        target=y/pack["cut"];truth[number]=target
        events[number]=[]
        for start in range(20,len(y),10):
            stop=min(len(y),start+10);sl=slice(start,stop)
            bank,out=decide(models,policy,[a[sl] for a in v],c[sl])
            payload={"场次":number,"窗":list(range(start,stop)),"已回标":len(policy.site),
                     "阈值倍数":{k:z.tolist() for k,z in out.items()}}
            key=digest(payload)
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("当前半秒预测未交接")
            events[number].append(key)
            if stop<len(y):
                j=stop-1
                policy.update(c[j:j+1],bank[-1:],target[j:j+1])
                for m,a in zip(models,v,strict=True):m.update(a[j:j+1],target[j:j+1])
    pools={};scores={};all_y=[]
    def metrics(y,p):
        return {"F1":float(f1_score(y>=1,p>=1,labels=[False,True],average="macro",zero_division=0)),
                "最弱召回":float(recall_score(y>=1,p>=1,labels=[False,True],average=None,zero_division=0).min()),
                "参考电压RMSE":float(np.sqrt(mean_squared_error(y,p))*pack["cut"])}
    for number,sequence in events.items():
        results={};idx=[]
        for key in sequence:
            p=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(p)!=key:raise ValueError("开发回执被改动")
            idx.extend(p["窗"])
            for k,v in p["阈值倍数"].items():results.setdefault(k,[]).extend(v)
        if idx!=list(range(20,len(truth[number]))):raise ValueError("场次窗口重叠或遗漏")
        y=truth[number][idx];all_y.extend(y)
        scores[number]={k:metrics(y,np.array(v)) for k,v in results.items()}
        for k,v in results.items():pools.setdefault(k,[]).extend(v)
    means={k:metrics(np.array(all_y),np.array(v)) for k,v in pools.items()}
    fixed=max((k for k in means if k!=PRIMARY),key=lambda k:means[k]["F1"])
    paper=max((k for k in means if "论文" in k),key=lambda k:means[k]["F1"])
    gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    dec={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],"论文F1":means[paper]["F1"],
         "差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":dec,"scores":scores,"means":means,"requests":events,"阈值伏":pack["cut"],"源权重SHA256":sha256(path.read_bytes()).hexdigest(),"窗口":len(all_y)})
    print(json.dumps({"移动探头场次开发":dec},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本项已交付")
        run()
