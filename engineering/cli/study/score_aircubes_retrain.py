"""以源参考初始化电极曲线，从首期开发按作者配对逐槽实发并评分。"""
from hashlib import sha256
import json
import joblib
import numpy as np
import pandas as pd

from score_aircubes_intervals import DATA,DEST as PRIOR,START,known,station,LocalStore,LocalAcceptance,atomic_json,digest,OLD,PRIMARY as PAIRED
from harness.models.aircubes_retrain import ElectrodeRelease,ElectrodeMaintenance,PRIMARY,INCUMBENT

DEST=PRIOR.parent/"aircubes-retrain"
ADAPTIVE={PRIMARY,OLD,PAIRED,"原始重训只按本机平均误差选"}


def inputs():
    manifest=json.loads((DATA/"source.json").read_bytes())
    reference=pd.read_csv(known("reference-first.csv",manifest),usecols=["timestamp","NO2_ppb"]).set_index("timestamp")["NO2_ppb"]
    aligned=reference.copy()
    aligned.index-=600
    return manifest,reference,aligned


def windows(t):
    edges=np.r_[0,np.flatnonzero(np.diff(t//21600))+1,len(t)]
    return zip(edges[:-1],edges[1:],strict=True)


def score_requests(receiver,seq,reference):
    sq={};used=0
    for key in seq:
        packet=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
        if digest(packet)!=key:
            raise ValueError("实际已发记录发生变化")
        labels=reference.reindex(np.array(packet["区间起点"])+600).to_numpy()
        ok=np.isfinite(labels);used+=int(ok.sum())
        for name,values in packet["预测"].items():
            sq[name]=sq.get(name,0.)+np.sum((np.asarray(values)[ok]-labels[ok])**2)
    return {k:float(np.sqrt(v/used)) for k,v in sq.items()},used


def summarize(reports,primary,adaptive):
    means={n:float(np.mean([v[n] for v in reports.values()])) for n in next(iter(reports.values()))}
    fixed=min((n for n in means if n not in adaptive),key=means.get)
    paper_names=[n for n in means if n not in adaptive and ("论文" in n or n.startswith(("四模型","固定现场配对","保持装机时","现场误差组合初值","用已返参考拟合")))]
    paper=min(paper_names,key=means.get)
    gain=1-means[primary]/min(means[fixed],means[paper])
    worst=max(r[primary] for r in reports.values());fw=max(r[fixed] for r in reports.values())
    return means,{"现场":primary,"现场误差":means[primary],"固定":fixed,"固定误差":means[fixed],
        "论文":paper,"论文误差":means[paper],"降幅":gain,"最差仪器":worst,"固定最差":fw,
        "进入独立":gain>=.02 and worst<=fw*1.01}


def run():
    manifest,reference,aligned=inputs()
    prev=json.loads((PRIOR/"scores.json").read_bytes())
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    reports={};events={};quantities={};identities={}
    for site,keys in prev["requests"].items():
        t,x,y=station(site,aligned,manifest)
        fit=(t<START)&(t+600<=START)&np.isfinite(x).all(1)&np.isfinite(y)
        live=(t>=START)&np.isfinite(x).all(1)
        source=PRIOR/(site+".joblib")
        if sha256(source.read_bytes()).hexdigest()!=prev["源模型SHA256"][site]:raise ValueError("论文权重不符")
        tools={"paper":joblib.load(source),"electrode":ElectrodeRelease().fit(x[fit],y[fit])}
        path=DEST/(site+".joblib");joblib.dump(tools,path,compress=3)
        identities[site]=sha256(path.read_bytes()).hexdigest()
        tools=joblib.load(path)
        t=t[live];x=x[live];bank=tools["paper"].predict(x)
        state=ElectrodeMaintenance(tools["electrode"]);seq=[]
        for (a,b),oldkey in zip(windows(t),keys,strict=True):
            earlier=json.loads((PRIOR/"accepted"/(oldkey+".json")).read_bytes())["payload"]
            if digest(earlier)!=oldkey or t[a:b].tolist()!=earlier["区间起点"]:raise ValueError("前序目标发生变化")
            out=state.predict(x[a:b],bank[a:b])
            if not np.allclose(out[INCUMBENT],earlier["预测"][INCUMBENT],rtol=0,atol=1e-8):
                raise ValueError("最强固定未逐条复现")
            packet={"区间起点":t[a:b].tolist(),"仪器":site,"已取参考":state.old.old.count,
                    "预测":{n:v.tolist() for n,v in out.items()},"前序":oldkey}
            k=digest(packet)
            if not receiver.submit(k,packet)["accepted"]:raise ValueError("实际预测未被接收")
            seq.append(k)
            end=int(t[b-1])+600
            if b<len(t) and end%21600==0:
                val=reference.get(end,np.nan)
                if np.isfinite(val):state.update(float(val))
        reports[site],used=score_requests(receiver,seq,reference)
        events[site]=seq;quantities[site]={"实发":len(t),"可评分":used,"已回标":state.old.old.count}
    means,d=summarize(reports,PRIMARY,ADAPTIVE)
    atomic_json(DEST/"scores.json",{"decision":d,"scores":reports,"means":means,"requests":events,
        "服务与回标":quantities,"源模型SHA256":identities})
    print(json.dumps({"现场重训原始电极":d,"对照原固定":means[INCUMBENT],"每台":quantities},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同协议已有实际交付")
        run()
