"""在源端留整场学习参考误差的延续性，通过真实无答案服务重新发出开发日全部预测。"""
from hashlib import sha256
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from score_vapor_field import ROOT, DATA, DEST as FIELD, LocalAcceptance, LocalStore, atomic_json, digest, read_flight, features
from score_vapor_boundary import DEST as BOUNDARY, batches
from harness.models.vapor_boundary import LevelFit, PRIMARY as MIXED
from harness.models.vapor_correction import Corrections, ResidualHistory, PAPER, PRIMARY, predictions, losses

DEST = FIELD.parent/"vapor-correction"
CLI = ROOT/"engineering/cli/workflow/vapor_service.py"


def load_verified(folder):
    meta=json.loads((folder/"scores.json").read_bytes())
    path=folder/"release.joblib"
    if sha256(path.read_bytes()).hexdigest()!=meta["源权重SHA256"]:
        raise ValueError("权重与已有交付不符")
    return joblib.load(path),meta


def install(source, scale, cut):
    records=[]
    for record in source:
        v,_,y=features(record,scale)
        records.append((v[5],y/cut))
    trials=[]
    for i,(x,y) in enumerate(records):
        train=[(a[20:],t[20:]) for j,(a,t) in enumerate(records) if i!=j]
        xx,tt=zip(*train,strict=True)
        model=LevelFit().fit(np.vstack(xx),np.concatenate(tt))
        history=ResidualHistory()
        rows=[]; truth=[]
        for idx,_ in batches(len(y)):
            v=np.arcsinh(model.predict(x[idx])[:,1])
            rows.extend(history.context(p,k) for k,p in enumerate(v,1))
            truth.extend(y[idx])
            if idx[-1]+1<len(y):
                history.observe(v[-1],y[idx[-1]])
                model.update(x[idx[-1:]],y[idx[-1:]])
        trials.append((np.stack(rows),np.array(truth)))
    def fit(indices):
        rows=np.vstack([trials[i][0] for i in indices])
        truth=np.concatenate([trials[i][1] for i in indices])
        return Corrections().fit(rows,np.arcsinh(truth)-rows[:,3])
    held=[]
    for i,(rows,truth) in enumerate(trials):
        corrected=fit([j for j in range(len(trials)) if j!=i]).offsets(rows)
        held.append(losses(predictions(rows[:,3],corrected),truth))
    return fit(range(len(records))),np.vstack(held).mean(0)


def communicate(process,request):
    process.stdin.write(json.dumps(request,ensure_ascii=False,allow_nan=False)+"\n")
    process.stdin.flush()
    line=process.stdout.readline()
    if not line:
        raise ValueError("探头服务提前退出")
    return json.loads(line)


def clock_groups(path,ends):
    frame=pd.read_csv(path,usecols=["time"])
    timestamp=pd.to_datetime(frame["time"],format="mixed")
    times=(timestamp-timestamp.iloc[0]).dt.total_seconds().to_numpy()
    return [times[a:b] for a,b in zip(ends[:-1],ends[1:],strict=True)]


def deliver(number,manifest,meta,release,checksum,receiver,clocked=False,environment=None,wind=None):
    raw,y,_=read_flight(number,manifest)
    frame=pd.read_csv(DATA/manifest["记录"][str(number)]["文件"],usecols=["PID-sensor"])
    ref=frame["PID-sensor"].to_numpy()
    ends=np.r_[0,np.cumsum([len(a) for a in raw])]
    reference=[ref[a:b] for a,b in zip(ends[:-1],ends[1:],strict=True)]
    clocks=clock_groups(DATA/manifest["记录"][str(number)]["文件"],ends) if clocked else None
    previous_dir=DEST if clocked or environment is not None else BOUNDARY
    if wind is not None:
        previous_dir=DEST.parent/"vapor-environment"
    process=subprocess.Popen([sys.executable,"-B",str(CLI),"--release",str(release),"--sha256",checksum],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8")
    receipts=[]; used=0
    try:
        setup={"启动波形":[a.tolist() for a in raw[:20]],"启动参考":[p.tolist() for p in reference[:20]]}
        if clocked:
            setup["启动时钟"]=[a.tolist() for a in clocks[:20]]
        if environment is not None:
            setup["启动温湿度"]=[a.tolist() for a in environment[:20]]
        if wind is not None:
            setup["启动气流"]=[a.tolist() for a in wind[:20]]
        boot=communicate(process,setup)
        if boot!={"下个窗口":20}:
            raise ValueError("原始装机失败")
        for (ix,_),key in zip(batches(len(y)),meta["requests"][str(number)],strict=True):
            previous=json.loads((previous_dir/"accepted"/(key+".json")).read_bytes())["payload"]
            if digest(previous)!=key or previous["窗" if previous_dir==BOUNDARY else "窗口"]!=ix.tolist():
                raise ValueError("原始回执身份不符")
            cols={}
            for j in ix.tolist():
                request={"窗口":j,"慢通道":raw[j].tolist()}
                if clocked:
                    request["时钟"]=clocks[j].tolist()
                if environment is not None:
                    request["温湿度"]=environment[j].tolist()
                if wind is not None:
                    request["气流"]=wind[j].tolist()
                reply=communicate(process,request)
                if reply["窗口"]!=j:
                    raise ValueError("窗口次序错误")
                for name,val in reply["预测"].items():
                    cols.setdefault(name,[]).append(val)
            for name in ((PAPER,MIXED) if previous_dir==BOUNDARY else previous["阈值倍数"]):
                if not np.allclose(cols[name],previous["阈值倍数"][name],rtol=1e-8,atol=1e-8):
                    raise ValueError("原始服务入口未复现先前预测 "+name)
            parent_key="边界版本回执" if previous_dir==BOUNDARY else "前一版本回执"
            payload={"场次":number,"窗口":ix.tolist(),"回标已用":used,"阈值倍数":cols,parent_key:key}
            req=digest(payload)
            if not receiver.submit(req,payload)["accepted"]:
                raise ValueError("新已发预测接收失败")
            receipts.append(req)
            if ix[-1]+1<len(y):
                last=int(ix[-1])
                if communicate(process,{"回标窗":last,"参考电压":float(reference[last].mean())})!={"下个窗口":last+1}:
                    raise ValueError("现场参考没有及时接收")
                used+=1
        process.stdin.close()
        if process.wait(timeout=15):
            raise ValueError("服务未正常结束")
    finally:
        if process.poll() is None:
            process.kill();process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
    return receipts,y,used


def score(yy,columns):
    return {name:{"F1":float(f1_score(yy,np.asarray(pred)>=1,labels=[False,True],average="macro",zero_division=0)),
                  "最弱召回":float(recall_score(yy,np.asarray(pred)>=1,labels=[False,True],average=None,zero_division=0).min())}
            for name,pred in columns.items()}


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    source,_=load_verified(FIELD)
    boundary,old=load_verified(BOUNDARY)
    model,prior=install([read_flight(i,manifest) for i in manifest["源日"]],source["scale"],source["cut"])
    release=DEST/"release.joblib"
    joblib.dump({"cut":source["cut"],"scale":source["scale"],"boundary":boundary,"correction":model,
                 "initial_loss":prior},release,compress=3)
    checksum=sha256(release.read_bytes()).hexdigest()
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    requests={};per_flight={}; all_y=[]; all_cols={}; spent={}
    for i in manifest["开发日"]:
        seq,y,used=deliver(i,manifest,old,release,checksum,receiver)
        rows=[];cols={}
        for key in seq:
            pkt=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(pkt)!=key:
                raise ValueError("交接后预测变化")
            rows.extend(pkt["窗口"])
            for name,c in pkt["阈值倍数"].items():
                cols.setdefault(name,[]).extend(c)
        if rows!=list(range(20,len(y))):
            raise ValueError("缺少实际服务窗口")
        labels=y[rows]/source["cut"]>=1
        per_flight[i]=score(labels,cols);requests[i]=seq;spent[i]=used
        all_y.extend(labels)
        for name,c in cols.items():
            all_cols.setdefault(name,[]).extend(c)
    means=score(all_y,all_cols)
    fixed=max((name for name in means if name not in (MIXED,PRIMARY)),key=lambda name:means[name]["F1"])
    # 新的固定校正均以原论文储备池为基础，可供论文端同条件使用。
    gap=means[PRIMARY]["F1"]-means[fixed]["F1"]
    d={"现场":PRIMARY,"固定":fixed,"论文":fixed,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
       "论文F1":means[fixed]["F1"],"差值":gap,
       "进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":per_flight,"requests":requests,
        "源权重SHA256":checksum,"窗口":len(all_y),"各场已回参考":spent})
    print(json.dumps({"逐半秒部署与已回误差校正":d,"实际预测数":len(all_y),"总回标":sum(spent.values())},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已有完整开发日部署")
        run()
