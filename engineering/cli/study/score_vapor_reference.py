"""同一批次只接收前次参考，比较用真实读数设定局部零点及其控制。"""
import json
from hashlib import sha256
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score
from score_vapor_field import DATA,DEST as PRIOR,LocalStore,LocalAcceptance,atomic_json,digest,features,read_flight
from harness.models.vapor_reference import ReferenceModels,ReferenceTrust,PRIMARY,relative,context
from harness.models.vapor_field import OnlineVoltage

DEST=PRIOR.parent/"vapor-reference"
STRAIGHT="论文储备池/近期装机"


def batches(n):
    for lo in range(20,n,10):
        yield np.arange(lo,min(lo+10,n)),lo-1


def make_base(recs):
    features_by_view=[[],[],[]];ys=[];start=[]
    for views,_,y in recs:
        for idx,prev in batches(len(y)):
            a=relative(views,idx,prev,y[prev])
            for target,v in zip(features_by_view,a,strict=True):target.extend(v)
            ys.extend(y[idx]);start.extend(np.repeat(y[prev],len(idx)))
    models=ReferenceModels().fit(list(map(np.stack,features_by_view)),np.array(ys),np.array(start))
    paper=OnlineVoltage().fit(np.vstack([r[0][5][20:] for r in recs]),np.concatenate([r[2][20:] for r in recs]))
    return models,paper


def predict_batch(models,paper,views,c,idx,prev,anchor):
    x=relative(views,idx,prev,anchor)
    straight=paper.predict(views[5][idx])[:,2]
    out=models.predict(x,anchor,straight)
    ctx=context(views,c,idx,prev,anchor)
    bank=np.column_stack(list(out.values()))
    return x,out,ctx,bank


def fit(recs):
    context_rows=[];cols=[];truth=[]
    for i,(v,c,y) in enumerate(recs):
        m,pp=make_base([r for k,r in enumerate(recs) if k!=i])
        for idx,prev in batches(len(y)):
            x,out,ctx,bank=predict_batch(m,pp,v,c,idx,prev,y[prev])
            context_rows.extend(ctx);cols.extend(bank);truth.extend(y[idx])
            if idx[-1]+1<len(y):
                m.update([a[-1:] for a in x],y[idx[-1:]],y[prev])
                pp.update(v[5][idx[-1:]],y[idx[-1:]])
    return (*make_base(recs),ReferenceTrust().fit(np.stack(context_rows),np.stack(cols),np.array(truth)))


def run():
    meta=json.loads((PRIOR/"scores.json").read_bytes());source=json.loads((DATA/"source.json").read_bytes())
    path=PRIOR/"release.joblib"
    if sha256(path.read_bytes()).hexdigest()!=meta["源权重SHA256"]:raise ValueError("原探头权重变化")
    pack=joblib.load(path);cut=meta["阈值伏"]
    def data(i):
        v,c,y=features(read_flight(i,source),pack["scale"])
        return v,c,y/cut
    bundle=DEST/"release.joblib"
    joblib.dump(fit([data(i) for i in source["源日"]]),bundle,compress=3)
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    events={};truth={}
    for number in source["开发日"]:
        v,c,y=data(number);truth[number]=y
        model,paper,sel=joblib.load(bundle);events[number]=[]
        anchor=y[19]
        for (idx,prev),old_req in zip(batches(len(y)),meta["requests"][str(number)],strict=True):
            old=json.loads((PRIOR/"accepted"/(old_req+".json")).read_bytes())["payload"]
            if digest(old)!=old_req or old["窗"]!=idx.tolist():raise ValueError("前序窗口变化")
            x,out,ctx,bank=predict_batch(model,paper,v,c,idx,prev,anchor)
            if not np.allclose(out["原论文近期读法"],old["阈值倍数"][STRAIGHT],atol=1e-8,rtol=0):raise ValueError("原强论文未复现")
            controls=list(out)
            out[PRIMARY]=sel.predict(ctx,bank)
            probabilities=1/(1+np.exp(-np.clip(3*(bank-1),-50,50)))
            out["所有参照读法固定投票"]=2*probabilities.mean(1)
            for name in controls:
                out["论文与参照各半/"+name]=(out[name]+out["原论文近期读法"])/2
            values={**old["阈值倍数"],**{k:a.tolist() for k,a in out.items()}}
            p={"场次":number,"窗":idx.tolist(),"前次参考窗":int(prev),"阈值倍数":values,"上轮":old_req}
            req=digest(p)
            if not receiver.submit(req,p)["accepted"]:raise ValueError("参照训练结果未交付")
            events[number].append(req)
            if idx[-1]+1<len(y):
                last=y[idx[-1:]]
                sel.update(ctx[-1:],bank[-1:],last);model.update([a[-1:] for a in x],last,anchor);paper.update(v[5][idx[-1:]],last)
                anchor=float(last[0])
    all_y=[];total={};scores={}
    def metric(yy,p):
        return {"F1":float(f1_score(yy,p>=1,average="macro",labels=[False,True],zero_division=0)),
                "最弱召回":float(recall_score(yy,p>=1,average=None,labels=[False,True],zero_division=0).min())}
    for i,seq in events.items():
        yy=[];cols={}
        for h in seq:
            p=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
            if digest(p)!=h:raise ValueError("已交付预测改变")
            yy.extend(truth[i][p["窗"]]>=1)
            for k,a in p["阈值倍数"].items():cols.setdefault(k,[]).extend(a)
        scores[i]={k:metric(yy,np.array(v)) for k,v in cols.items()}
        all_y.extend(yy)
        for k,a in cols.items():total.setdefault(k,[]).extend(a)
    means={k:metric(all_y,np.array(v)) for k,v in total.items()}
    fixed=max([meta["decision"]["固定"],*[k for k in out if k!=PRIMARY]],key=lambda k:means[k]["F1"])
    best=max([meta["decision"]["论文"],*[k for k in out if "论文" in k]],key=lambda k:means[k]["F1"])
    gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,best))
    d={"现场":PRIMARY,"固定":fixed,"论文":best,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],"论文F1":means[best]["F1"],
       "差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":scores,"requests":events,"源权重SHA256":sha256(bundle.read_bytes()).hexdigest()})
    print(json.dumps({"上次实测参照":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本项已经交付")
        run()
