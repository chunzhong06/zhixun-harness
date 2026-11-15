"""以完整飞越为源端交叉检验单位，提升实际阈值附近的训练权重。"""
from hashlib import sha256
import json
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_vapor_field import DATA,DEST as ORIGINAL,LocalStore,LocalAcceptance,atomic_json,digest,features,read_flight
from score_vapor_reference import DEST as PRIOR,batches
from harness.models.vapor_boundary import LevelFit,EdgeWeights,VIEWS,TARGETS,PRIMARY,measured_views,decision_context

DEST=PRIOR.parent/"vapor-boundary"
NAMES=[a+"/"+b for a in VIEWS for b in TARGETS]


def install(records):
    def fit(indices):
        y=np.concatenate([records[k][1][20:] for k in indices])
        models=[]
        for j in range(3):
            x=np.vstack([records[k][0][j][20:] for k in indices])
            models.append(LevelFit().fit(x,y))
        return models
    predictions=[];errors=[];contexts=[]
    for k,(v,y) in enumerate(records):
        m=fit([i for i in range(len(records)) if i!=k])
        for ix,prev in batches(len(y)):
            q=np.column_stack([model.predict(x[ix]) for model,x in zip(m,v,strict=True)])
            contexts.extend(decision_context(q,y[prev]));predictions.extend(q);errors.extend(y[ix])
            if ix[-1]<len(y)-1:
                for model,x in zip(m,v,strict=True):model.update(x[ix[-1:]],y[ix[-1:]])
    return fit(range(len(records))),EdgeWeights().fit(np.stack(predictions),np.stack(contexts),np.array(errors))


def run():
    spec=json.loads((DATA/"source.json").read_bytes());old=json.loads((PRIOR/"scores.json").read_bytes())
    meta=json.loads((ORIGINAL/"scores.json").read_bytes());bundle=ORIGINAL/"release.joblib"
    if sha256(bundle.read_bytes()).hexdigest()!=meta["源权重SHA256"]:raise ValueError("源读数比例不符")
    pack=joblib.load(bundle);cut=pack["cut"]
    def data(i):
        v,c,y=features(read_flight(i,spec),pack["scale"])
        return measured_views(v,np.arange(len(y))),y/cut
    model_file=DEST/"release.joblib"
    joblib.dump(install([data(i) for i in spec["源日"]]),model_file,compress=3)
    receiver=LocalAcceptance();receiver.output=DEST/"accepted";events={};targets={}
    for i in spec["开发日"]:
        m,w=joblib.load(model_file)
        v,y=data(i);targets[i]=y;events[i]=[];anchor=y[19]
        for (ix,prev),req in zip(batches(len(y)),old["requests"][str(i)],strict=True):
            parent=json.loads((PRIOR/"accepted"/(req+".json")).read_bytes())["payload"]
            if digest(parent)!=req or parent["窗"]!=ix.tolist():raise ValueError("旧交付身份变化")
            q=np.column_stack([model.predict(x[ix]) for model,x in zip(m,v,strict=True)])
            ctx=decision_context(q,anchor)
            out=dict(zip(NAMES,q.T,strict=True));out[PRIMARY]=w.predict(q,ctx)
            out["边界工具固定平均"]=q.mean(1)
            previous=np.array(parent["阈值倍数"][old["decision"]["固定"]])
            for n in NAMES:
                out["原论文与"+n+"各半"]=(out[n]+previous)/2
            values={**parent["阈值倍数"],**{k:p.tolist() for k,p in out.items()}}
            pkt={"场次":i,"窗":ix.tolist(),"上轮":req,"阈值倍数":values,"回标数":len(w.hist)}
            h=digest(pkt)
            if not receiver.submit(h,pkt)["accepted"]:raise ValueError("边界输出未交付")
            events[i].append(h)
            if ix[-1]<len(y)-1:
                anchor=y[ix[-1]]
                w.update(q[-1:],ctx[-1:],np.array([anchor]))
                for model,x in zip(m,v,strict=True):model.update(x[ix[-1:]],np.array([anchor]))
    groups={};all_y=[];total={}
    def metric(y,v):
        return {"F1":float(f1_score(y,v>=1,labels=[False,True],average="macro",zero_division=0)),
                "最弱召回":float(recall_score(y,v>=1,labels=[False,True],average=None,zero_division=0).min())}
    for i,sequence in events.items():
        yy=[];cols={}
        for h in sequence:
            pkt=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
            if digest(pkt)!=h:raise ValueError("已发预测改变")
            yy.extend(targets[i][pkt["窗"]]>=1)
            for k,v in pkt["阈值倍数"].items():cols.setdefault(k,[]).extend(v)
        groups[i]={k:metric(yy,np.array(v)) for k,v in cols.items()};all_y.extend(yy)
        for k,v in cols.items():total.setdefault(k,[]).extend(v)
    means={k:metric(all_y,np.array(v)) for k,v in total.items()}
    fixed=max([old["decision"]["固定"],*[n for n in out if n!=PRIMARY]],key=lambda n:means[n]["F1"])
    paper=max([old["decision"]["论文"],*[n for n in out if "论文" in n]],key=lambda n:means[n]["F1"])
    gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],"论文F1":means[paper]["F1"],
       "差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":groups,"requests":events,"源权重SHA256":sha256(model_file.read_bytes()).hexdigest()})
    print(json.dumps({"阈值附近识别":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同阈值训练已有交付")
        run()
