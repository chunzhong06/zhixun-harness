"""从原始源期时间序列学习误差持续性，在四台开发设备交付并与同工具固定版本对照。"""
from hashlib import sha256
import json
import joblib
import numpy as np

from score_aircubes_retrain import DEST as PRIOR,START,inputs,station,windows,score_requests,summarize,ADAPTIVE,LocalStore,LocalAcceptance,atomic_json,digest
from harness.models.aircubes_innovation import TransferRelease,IntervalMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-transfer"


def install(tools,time,raw,truth):
    tools["transfer"]=TransferRelease().fit(time,raw,truth,tools["electrode"])
    return tools


def evaluate(stage,parent,train,controller,primary,adaptive,extend=None):
    manifest,reference,aligned=inputs()
    prev=json.loads((parent/"scores.json").read_bytes())
    receiver=LocalAcceptance();receiver.output=stage/"accepted"
    reports={};events={};quantities={};identity={}
    for site,keys in prev["requests"].items():
        t,x,y=station(site,aligned,manifest)
        fit=(t<START)&np.isfinite(x).all(1)&np.isfinite(y)
        live=(t>=START)&np.isfinite(x).all(1)
        if extend is not None:
            x=extend(site,t,x,manifest)
        path=parent/(site+".joblib")
        if sha256(path.read_bytes()).hexdigest()!=prev["源模型SHA256"][site]:raise ValueError("前次权重身份不符")
        tools=train(joblib.load(path),t[fit],x[fit],y[fit])
        path=stage/(site+".joblib");joblib.dump(tools,path,compress=3);identity[site]=sha256(path.read_bytes()).hexdigest()
        tools=joblib.load(path);state=controller(tools)
        clock=t[live];raw=x[live];paper=tools["paper"].predict(raw[:,:5])
        count=0;seq=[]
        for (a,b),parent_key in zip(windows(clock),keys,strict=True):
            p=json.loads((parent/"accepted"/(parent_key+".json")).read_bytes())["payload"]
            if digest(p)!=parent_key or p["区间起点"]!=clock[a:b].tolist() or p["已取参考"]!=count:
                raise ValueError("前次服务测量或回标预算发生变化")
            out=state.predict(clock[a:b],raw[a:b],paper[a:b])
            for name in [prev["decision"]["现场"],prev["decision"]["固定"],prev["decision"]["论文"]]:
                if not np.allclose(out[name],p["预测"][name],rtol=0,atol=1e-7):raise ValueError(name+"未复现")
            packet={"仪器":site,"区间起点":clock[a:b].tolist(),"已取参考":count,"前序":parent_key,
                    "预测":{n:v.tolist() for n,v in out.items()}}
            req=digest(packet)
            if not receiver.submit(req,packet)["accepted"]:raise ValueError("新增训练结果交付失败")
            seq.append(req)
            end=int(clock[b-1])+600
            if b<len(clock) and end%21600==0:
                val=reference.get(end,np.nan)
                if np.isfinite(val):state.update(float(val));count+=1
        reports[site],used=score_requests(receiver,seq,reference);events[site]=seq
        quantities[site]={"实发":len(clock),"可评分":used,"已回标":count}
    means,d=summarize(reports,primary,adaptive)
    atomic_json(stage/"scores.json",{"decision":d,"scores":reports,"means":means,"requests":events,
        "服务与回标":quantities,"源模型SHA256":identity})
    print(json.dumps({"同条件前向维护":d,"服务与回标":quantities},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("误差持续性已有开发发放")
        evaluate(DEST,PRIOR,install,IntervalMaintenance,PRIMARY,ADAPTIVE|{PRIMARY})
