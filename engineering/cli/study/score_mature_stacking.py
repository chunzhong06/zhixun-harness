"""在同一回标池上实际拟合 sklearn 的离散朴素贝叶斯与逻辑堆叠，逐行检验迁移来源。"""
from hashlib import sha256
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,atomic_json,digest,read
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from score_chongqing_feedback import read_split
from score_guarded_gas import SEEDS
from score_observer_errors import datasets
from gas_complementary_feedback import ROOT


class MatureStack:
    def __init__(self,names,classes):
        self.names=tuple(names)
        self.classes=classes
        self.full=np.arange(classes)
        self.category={k:CategoricalNB(alpha=1.,fit_prior=(k=="nb-frequency"),min_categories=classes)
                       for k in ("nb","nb-frequency","nb-diagonal")}
        pseudocounts=np.repeat(self.full[:,None],len(names),axis=1)
        for k,m in self.category.items():
            if k=="nb-frequency":
                # 首次真实回标时才创建经验类先验，零样本时仍由固定投票提供输出。
                continue
            m.partial_fit(pseudocounts,self.full,classes=self.full,sample_weight=np.ones(classes) if k=="nb-diagonal" else np.zeros(classes))
        self.observer=ObserverErrors(names,classes)
        self.x=[]
        self.y=[]

    def matrix(self,p):
        return np.stack([p[k] for k in self.names],axis=1).astype(int)-1

    def onehot(self,x):
        return np.eye(self.classes)[x].reshape(len(x),-1)

    def predict(self,p):
        vote=majority(p,self.classes)
        out={"vote":vote,"observer":self.observer.predict(p,"uniform")}
        arms=(*self.category,"logistic","logistic-balanced")
        if not self.y:
            return {**out,**dict.fromkeys(arms,vote)}
        x=self.matrix(p)
        def choose(probs,labels):
            ties=np.zeros((len(vote),self.classes),dtype=bool)
            ties[:,labels]=np.isclose(probs,probs.max(1,keepdims=True),atol=1e-12,rtol=0.)
            return np.where(ties[np.arange(len(vote)),vote-1],vote,ties.argmax(1)+1)
        for k,m in self.category.items():
            with np.errstate(divide="ignore"):
                out[k]=choose(m.predict_log_proba(x),m.classes_)
        if not np.array_equal(out["nb-diagonal"],out["observer"]):
            raise ValueError("成熟组件的对角伪计数与手工规则结果不一致")
        y=np.concatenate(self.y)
        for k,weight in (("logistic",None),("logistic-balanced","balanced")):
            if len(set(y))==1:
                out[k]=vote
                continue
            m=LogisticRegression(class_weight=weight,C=1.,max_iter=1000,random_state=17)
            m.fit(self.onehot(np.concatenate(self.x)),y)
            out[k]=choose(m.predict_log_proba(self.onehot(x)),m.classes_)
        return out

    def update(self,y,p):
        if not len(y):
            return
        x=self.matrix(p)
        target=np.asarray(y,dtype=int)-1
        for m in self.category.values():
            with np.errstate(divide="ignore"):
                m.partial_fit(x,target,classes=self.full)
        self.x.append(x)
        self.y.append(target)
        self.observer.update(y,p)


def evaluate(name,stream,truth,budget,classes,split=False):
    folder=OUT/"observer-errors/mature"/name
    groups={}
    preds={}
    for seed in SEEDS:
        stack=MatureStack(tuple(stream[0][1]),classes)
        arms={}
        groups[str(seed)]=[]
        for ids,p,feedback,fp,labels in stream:
            idx=np.asarray(sorted(range(len(feedback)),key=lambda j:digest([seed,feedback[j]]))[:budget],int)
            fb={k:v[idx] for k,v in fp.items()}
            y=[labels[i] for i in idx]
            if split:
                excluded=set(idx.tolist())
                service_idx=[j for j in range(len(ids)) if j not in excluded]
                ids=[ids[i] for i in service_idx]
                p={k:v[service_idx] for k,v in p.items()}
                stack.update(y,fb)
            groups[str(seed)].append(ids)
            for k,v in stack.predict(p).items():
                arms.setdefault(k,[]).extend(v.tolist())
            if not split:
                stack.update(y,fb)
        preds[str(seed)]=arms
    sealed=folder/"issued.json"
    atomic_json(sealed,{"groups":groups,"predictions":preds})
    saved=read(sealed)
    results={}
    for seed,arms in saved["predictions"].items():
        g=saved["groups"][seed]
        y=np.asarray([truth[i] for r in g for i in r])
        blocks=np.split(np.arange(len(y)),np.cumsum([len(r) for r in g])[:-1])
        results[seed]={k:scores(y,np.asarray(v),blocks,classes) for k,v in arms.items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"runs":results})
    metrics=("pooled","equal_batches","worst_batch")
    avgs={k:{m:mean(r[k][m] for r in results.values()) for m in metrics} for k in next(iter(results.values()))}
    print(name,{k:round(v["pooled"],5) for k,v in avgs.items()},flush=True)
    return avgs


if __name__=="__main__":
    results={}
    for name,stream,truth,budget,classes in datasets():
        results[name]=evaluate(name,stream,truth,budget,classes)
    for name,group in (("time","tima drift"),("board","board drift")):
        folder=OUT/"observer-errors/chongqing"/name
        saved=read(folder/"fixed.json")
        _,y,keys,_=read_split(ROOT/"data/public/chongqing-enose/drift.rar",group,"after")
        if keys!=saved["service_keys"]:
            raise ValueError("重庆测量编号不一致")
        p={k:np.asarray(v) for k,v in saved["predictions"].items()}
        key="chongqing-"+name
        results[key]=evaluate(key,[(keys,p,keys,p,y)],dict(zip(keys,y,strict=True)),32,14,True)
    atomic_json(OUT/"observer-errors/mature/scores.json",results)
