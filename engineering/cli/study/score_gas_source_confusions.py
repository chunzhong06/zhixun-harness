"""只在源训练行上做折外预测，检验是否值得把源端混淆估计带入未知批次。"""
from hashlib import sha256
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from check_feedback_controls import scores
from gas_complementary_feedback import ROOT,OUT,REGISTERED_SHA256,read_gas_batches,read,atomic_json,digest,official
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from score_guarded_gas import SEEDS,public_stream,external


def learn(pkg,batches):
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=[r for r in pkg.train if r.sample_id in ids]
    x=np.asarray([r.values for r in rows])
    y=np.asarray([r.label for r in rows])
    names=GasFeedbackBank.names
    model=ObserverErrors(names,6)
    issued={}
    for a,b in StratifiedKFold(n_splits=5,shuffle=True,random_state=17).split(x,y):
        scaler=StandardScaler().fit(x[a])
        lr=LogisticRegression(class_weight="balanced",random_state=17,max_iter=1000).fit(scaler.transform(x[a]),y[a])
        class FoldClassifier:
            def predict(self,values):
                return lr.predict_proba(scaler.transform(values))
        bank=GasFeedbackBank([rows[i] for i in a])
        predictions=bank.predict(FoldClassifier(),x[b])
        model.update(y[b],predictions)
        for j,i in enumerate(b):
            issued[rows[i].sample_id]={k:int(v[j]) for k,v in predictions.items()}
    if len(issued)!=len(rows) or not np.all(model.counts.sum(2)>0):
        raise ValueError("源端折外预测缺失")
    counts=model.counts
    prior=4*counts/counts.sum(2,keepdims=True)
    folder=OUT/"observer-errors/source-initial"
    atomic_json(folder/"source.json",{"fit_keys":sorted(issued),"fold_predictions":issued,"counts":counts.tolist(),"mass_per_class":4})
    return prior


def predict(prior,counts,p):
    names=tuple(p)
    smooth=counts+prior+np.eye(6,dtype=int)[None,:,:]+1
    rates=np.log(smooth/smooth.sum(2,keepdims=True))
    values=sum(rates[j].T[np.asarray(p[k])-1] for j,k in enumerate(names))
    vote=majority(p,6)
    tie=np.isclose(values,values.max(1,keepdims=True),rtol=0.,atol=1e-12)
    return np.where(tie[np.arange(len(vote)),vote-1],vote,tie.argmax(1)+1)


def compare(name,stream,truth,prior):
    keys=[k for ids,*_ in stream for k in ids]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(a[0]) for a in stream])[:-1])
    alternatives={"zero":np.zeros_like(prior),"all":prior,"except-original":prior.copy()}
    alternatives["except-original"][0]=0.
    preds={}
    for seed in SEEDS:
        restore=ObserverErrors(tuple(stream[0][1]),6)
        for pos,(_,p,returned,fp,labels) in zip(blocks,stream,strict=True):
            for k,s in alternatives.items():
                for budget,counts in ((32,restore.counts),(0,0)):
                    arm=f"{k}-{budget}-{seed}"
                    preds.setdefault(arm,np.empty(len(keys),int))[pos]=predict(s,counts,p)
            ix=np.asarray(sorted(range(len(returned)),key=lambda i:digest([seed,returned[i]]))[:32],int)
            restore.update([labels[i] for i in ix],{k:v[ix] for k,v in fp.items()})
    folder=OUT/"observer-errors/source-initial"/name
    path=folder/"issued.json"
    atomic_json(path,{"keys":keys,"predictions":{k:v.tolist() for k,v in preds.items()}})
    target=np.asarray([truth[k] for k in keys])
    t={k:scores(target,np.asarray(v),blocks,6) for k,v in read(path)["predictions"].items()}
    baseline=read(OUT/"observer-errors"/name/"issued.json")["predictions"]
    if any(not np.array_equal(preds[f"zero-32-{s}"],baseline[f"uniform-{s}"]) for s in SEEDS):
        raise ValueError("零源端统计没有复现已交付方案")
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":t})
    return {f"{k}-{b}":{m:mean(t[f"{k}-{b}-{s}"][m] for s in SEEDS) for m in ("pooled","worst_batch","equal_batches")}
            for k in alternatives for b in (0,32)}


if __name__=="__main__":
    pkg,_,batches=official("gas")
    prior=learn(pkg,batches)
    stream,truth,_=public_stream()
    result={"public":compare("public",stream,truth,prior)}
    later=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    truth={r.sample_id:r.label for b in later if b.batch_number>=3 for r in b.records}
    result["late"]=compare("late",external(OUT/"fixed.json",truth,GasFeedbackBank.names),truth,prior)
    atomic_json(OUT/"observer-errors/source-initial/scores.json",result)
    print({n:{k:round(v["pooled"],5) for k,v in t.items()} for n,t in result.items()})
