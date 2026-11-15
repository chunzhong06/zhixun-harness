"""比较集中回标和分批回标，给每条流严格相同的标签总数与获取资格。"""
from hashlib import sha256
from statistics import mean

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,atomic_json,digest,read
from harness.models.observer_errors import ObserverErrors
from score_guarded_gas import SEEDS
from score_observer_errors import datasets


def spread(limit,capacities):
    assigned=[0]*len(capacities)
    while sum(assigned)<limit:
        for j,cap in enumerate(capacities):
            if assigned[j]<cap:
                assigned[j]+=1
                if sum(assigned)==limit:
                    break
    return assigned


def evaluate(name,stream,truth,classes):
    caps=[len(r[2]) for r in stream[:-1]]
    if not caps:
        return {"rounds":1,"decision":"无后续回标服务轮"}
    keys=[i for ids,*_ in stream for i in ids]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(r[0]) for r in stream])[:-1])
    forecasts={}
    schedules={}
    for requested in (32,64):
        total=min(requested,caps[0],sum(caps))
        plans={"first":[total]+[0]*(len(caps)-1),"spread":spread(total,caps)}
        remaining=total
        early=[]
        for capacity in caps:
            take=min(32,remaining,capacity)
            early.append(take)
            remaining-=take
        if remaining==0:
            plans["early-capped"]=early
        plans["spread-freeze"]=plans["spread"]
        plans["spread-last"]=plans["spread"]
        for mode,plan in plans.items():
            if sum(plan)!=total:
                raise ValueError("两种校准使用了不同的反馈总数")
            schedules[f"{mode}-{requested}"]=plan+[0]
        for seed in SEEDS:
            orders=[sorted(range(len(r[2])),key=lambda i:digest([seed,r[2][i]])) for r in stream]
            for mode,plan in plans.items():
                model=ObserverErrors(tuple(stream[0][1]),classes)
                selected=set()
                pred=[]
                for j,(_,p,ids,returned,labels) in enumerate(stream):
                    pred.extend(model.predict(p,"uniform").tolist())
                    k=plan[j] if j<len(plan) else 0
                    if not k:
                        continue
                    ix=np.asarray(orders[j][:k],int)
                    sampled={ids[i] for i in ix}
                    if len(sampled)!=k or selected&sampled:
                        raise ValueError("回标编号重复")
                    selected.update(sampled)
                    if mode=="spread-freeze" and j>0:
                        continue
                    if mode=="spread-last":
                        model=ObserverErrors(tuple(p),classes)
                    model.update([labels[i] for i in ix],{a:v[ix] for a,v in returned.items()})
                if len(selected)!=total:
                    raise ValueError("实际反馈数不同于预定总数")
                forecasts[f"{mode}-{requested}-{seed}"]=pred
    folder=OUT/"observer-errors/same-total"/name
    sealed=folder/"issued.json"
    atomic_json(sealed,{"keys":keys,"schedules":schedules,"predictions":forecasts})
    saved=read(sealed)
    y=np.asarray([truth[i] for i in keys])
    arms={k:scores(y,np.asarray(v),blocks,classes) for k,v in saved["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":arms,"schedules":schedules})
    result={prefix:{m:mean(arms[f"{prefix}-{seed}"][m] for seed in SEEDS) for m in ("pooled","worst_batch","equal_batches")} for prefix in schedules}
    print(name,{k:round(v["pooled"],5) for k,v in result.items() if k in ("first-32","spread-32","first-64","spread-64")})
    return {"schedules":schedules,"arms":result}


if __name__=="__main__":
    results={}
    for name,stream,truth,_,classes in datasets():
        if not name.endswith("labels-4"):
            results[name]=evaluate(name,stream,truth,classes)
    atomic_json(OUT/"observer-errors/same-total/scores.json",results)
