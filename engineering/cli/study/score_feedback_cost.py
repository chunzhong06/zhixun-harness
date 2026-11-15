"""在相同服务批上减少实际回标，比较随机选样与优先抽查分类器分歧。"""
from hashlib import sha256
from statistics import mean

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,atomic_json,digest,read
from harness.models.observer_errors import ObserverErrors
from score_guarded_gas import SEEDS
from score_observer_errors import datasets


def ranked(p,classes):
    a=np.stack(list(p.values()))
    dist=np.stack([np.mean(a==k,axis=0) for k in range(1,classes+1)])
    return -np.sum(dist*np.log(np.maximum(dist,1e-12)),axis=0)


def choose(order,disagreements,budget,rule):
    if rule=="hash":
        return order[:budget]
    pre=order[:budget//2] if rule=="half" else []
    excluded=set(pre)
    # order 已经按固定散列排序，稳定排序让熵相同时的样本选择可复现。
    rest=sorted((i for i in order if i not in excluded),key=lambda i:-disagreements[i])
    return pre+rest[:budget-len(pre)]


def score(name,stream,truth,classes):
    folder=OUT/"observer-errors/cost"/name
    keys=[k for ids,*_ in stream for k in ids]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(r[0]) for r in stream])[:-1])
    preds={}
    costs={}
    entropies=[ranked(fp,classes) for _,_,_,fp,_ in stream]
    for seed in SEEDS:
        orders=[sorted(range(len(ids)),key=lambda i:digest([seed,ids[i]])) for _,_,ids,_,_ in stream]
        for budget,policy in ((0,"hash"),(4,"hash"),(8,"hash"),(32,"hash"),(8,"disagree"),(8,"half")):
            key=f"{budget}-{policy}-{seed}"
            outputs=[]
            spent=0
            learner=ObserverErrors(tuple(stream[0][1]),classes)
            for b,(_,p,ids,fp,y) in enumerate(stream):
                outputs.extend(learner.predict(p,"uniform").tolist())
                ix=np.asarray(choose(orders[b],entropies[b],budget,policy),int)
                if len(set(ix))!=min(budget,len(ids)):
                    raise ValueError("实际回标数或唯一性错误")
                learner.update([y[j] for j in ix],{k:v[ix] for k,v in fp.items()})
                if b<len(stream)-1:
                    spent+=len(ix)
            preds[key]=outputs
            costs[key]=spent
    sealed=folder/"issued.json"
    atomic_json(sealed,{"keys":keys,"predictions":preds,"labels_before_final":costs})
    saved=read(sealed)
    y=np.asarray([truth[k] for k in keys])
    table={k:scores(y,np.asarray(v),blocks,classes) for k,v in saved["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table,
                                      "labels_before_final":costs})
    return {key:{m:mean(table[f"{key}-{s}"][m] for s in SEEDS) for m in ("pooled","worst_batch","equal_batches")}
            for key in ("0-hash","4-hash","8-hash","32-hash","8-disagree","8-half")}


if __name__=="__main__":
    results={}
    for name,stream,truth,_,classes in datasets():
        if name.endswith("labels-4"):
            continue
        row=score(name,stream,truth,classes)
        results[name]=row
        print(name,{k:round(v["pooled"],4) for k,v in row.items()},flush=True)
    atomic_json(OUT/"observer-errors/cost/scores.json",results)
