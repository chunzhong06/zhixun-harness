"""用曾经实际到达的稍后波形作训练参照，发放端按当时输入提前预测。"""
from hashlib import file_digest
import json
from zipfile import ZipFile
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_gas_transitions import RAW,DATA,EXPECTED,DEST as PRIOR,RECEIPTS,seconds,LocalStore,LocalAcceptance,atomic_json,digest,DynamicEncoder,StreamRegression,EdgeEncoder,extract
from harness.models.delayed_teacher import DelayedTeacher,PRIMARY

DEST=PRIOR.parent/"delayed-teacher"


def run(site,origin,raw,labels,meta):
    e=EdgeEncoder(raw[origin][1]);encoder=DynamicEncoder(raw[origin][1])
    edge,diff,res=extract(raw[origin],e,encoder)
    y=np.array([r["label"]==2 for r in labels[origin]],dtype=float)
    # 源教师对每个连续时间块按整块留出预测，防止直接复制自己拟合过的答案。
    teacher=DelayedTeacher.from_source(edge,diff,res,y)
    base=StreamRegression().fit(diff,y);paper=StreamRegression().fit(res,y)
    encoder.reset();e.past=[]
    edge,diff,res=extract(raw[site],e,encoder)
    events=json.loads((RECEIPTS/site/"ongoing/run.json").read_bytes())["accepted"]
    answers={r["id"]:r["label"] for r in labels[site]}
    receiver=LocalAcceptance();receiver.output=DEST/site/"accepted"
    emitted=[];cursor=0
    for prev,event in zip(meta["requests"],events,strict=True):
        old=json.loads((PRIOR/site/"accepted"/(prev+".json")).read_bytes())["payload"]
        if digest([site,old])!=prev or len(teacher.answers)!=old["累计反馈"]:raise ValueError("前项反馈身份变化")
        ids=old["ids"];sl=slice(cursor,cursor+len(ids))
        if ids!=[r["id"] for r in labels[site][sl]]:raise ValueError("窗口编号错误")
        a=np.clip(base.outputs(diff[sl])[:,1],0,1);b=np.clip(paper.outputs(res[sl])[:,1],0,1)
        anchor=(a+b)/2 if site=="co" else b
        if (1+(anchor>=.5)).tolist()!=old["预测"][meta["decision"]["固定"]]:raise ValueError("原固定强基准未复现")
        out=teacher.predict(np.column_stack((edge[sl],diff[sl])),anchor)
        cols=dict(old["预测"])
        for n,p in out.items():
            cols[n]=(1+(p>=.5)).tolist()
            cols["论文与"+n+"各半"]=(1+((b+np.clip(p,0,1))/2>=.5)).tolist()
        payload={"ids":ids,"预测":cols,"上轮":prev,"回标":len(teacher.answers),"已成熟配对":list(teacher.count)}
        h=digest([site,payload])
        if not receiver.submit(h,payload)["accepted"]:raise ValueError("未交付提前量预测")
        emitted.append(h)
        ix=np.array([ids.index(uid) for uid in event["returned"]],int)
        if len(ix):
            yy=np.array([answers[uid]==2 for uid in event["returned"]])
            base.update(diff[sl][ix],yy);paper.update(res[sl][ix],yy)
        teacher.update(ix+cursor,np.array([answers[uid]==2 for uid in event["returned"]]))
        cursor+=len(ids)
    y=[];cols={}
    for h in emitted:
        v=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
        if digest([site,v])!=h:raise ValueError("预测被更改")
        y.extend(answers[i] for i in v["ids"])
        for n,xx in v["预测"].items():cols.setdefault(n,[]).extend(xx)
    y=np.array(y);switch=np.r_[False,np.diff(y)!=0]
    scores={n:{"F1":float(f1_score(y,v,average="macro")),"最弱召回":float(recall_score(y,v,average=None).min()),
               "切换错":int(np.sum((y!=v)&switch)),"稳定错":int(np.sum((y!=v)&~switch))} for n,v in cols.items()}
    const=[meta["decision"]["固定"],*[n for n in out if n!=PRIMARY],*["论文与"+n+"各半" for n in out if n!=PRIMARY]]
    fixed=max(const,key=lambda n:scores[n]["F1"])
    best=max([meta["decision"]["论文"],*["论文与"+n+"各半" for n in out]],key=lambda n:scores[n]["F1"])
    gap=scores[PRIMARY]["F1"]-max(scores[n]["F1"] for n in (fixed,best))
    dec={"现场":PRIMARY,"固定":fixed,"论文":best,"现场F1":scores[PRIMARY]["F1"],"固定F1":scores[fixed]["F1"],"论文F1":scores[best]["F1"],
         "差值":gap,"通过":gap>=.01 and scores[PRIMARY]["最弱召回"]>=scores[fixed]["最弱召回"]-.02}
    return {"decision":dec,"scores":scores,"requests":emitted}


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("提前量已有交付")
        with RAW.open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("原始数据变化")
        raw={}
        with ZipFile(RAW) as z:
            for site,file in (("co","ethylene_CO.txt"),("methane","ethylene_methane.txt")):
                with z.open(file) as stream:raw[site]=seconds(stream)
        meta=json.loads((PRIOR/"scores.json").read_bytes())
        labels={k:json.loads((DATA/k/"source.json").read_bytes()) for k in raw}
        results={s:run(s,t,raw,labels,meta[s]) for s,t in (("co","methane"),("methane","co"))}
        atomic_json(DEST/"scores.json",results)
        print(json.dumps({k+"因果提前":v["decision"] for k,v in results.items()},ensure_ascii=False))
