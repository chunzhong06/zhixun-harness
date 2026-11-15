"""与常数改判、固定票选以及事后局部固定作同等回标比较，保留各批真实选择和评分。"""
from hashlib import sha256
from time import perf_counter

import numpy as np

from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,read_gas_batches,read,atomic_json,digest
from harness.models.feedback_routing import FeedbackRouter,apply_choices


def scores(y,p,blocks,classes):
    def f1(ix):
        matrix=np.zeros((classes,classes),dtype=int)
        np.add.at(matrix,(y[ix]-1,p[ix]-1),1)
        den=matrix.sum(0)+matrix.sum(1)
        return float(np.mean(2*matrix.diagonal()[den>0]/den[den>0]))
    values=[f1(ix) for ix in blocks]
    recall=[float(np.mean(p[y==g]==g)) if np.any(y==g) else None for g in range(1,classes+1)]
    return {"pooled":f1(np.arange(len(p))),"equal_batches":float(np.mean(values)),"worst_batch":min(values),"by_batch":values,"recall":recall}


def replay(folder,pred,truth,groups,budgets,classes,*,expected=None):
    names=list(pred)
    keys=[k for group in groups for k in group]
    if len(keys)!=len(set(keys)) or any(v.shape!=(len(keys),) for v in pred.values()):
        raise ValueError("预测长度与分批清单不符")
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(g) for g in groups])[:-1])
    streams=dict(pred)
    votes=np.stack([np.sum(np.stack(list(pred.values()))==label,axis=0) for label in range(1,classes+1)],axis=1)
    tied=votes[np.arange(len(keys)),pred["current"]-1]==votes.max(axis=1)
    streams["vote"]=np.where(tied,pred["current"],votes.argmax(axis=1)+1)
    traces,times={},{}
    for seed in (17,47,89):
        order=[sorted(block,key=lambda j:digest([seed,keys[j]])) for block in blocks]
        for budget in budgets:
            tic=perf_counter()
            router=FeedbackRouter(names,classes)
            for batch,b in enumerate(blocks):
                published={k:p[b] for k,p in pred.items()}
                for mode in ("global","local","relabel"):
                    arm=f"{mode}-{budget}-{seed}"
                    choice=router.choices(mode)
                    streams.setdefault(arm,np.zeros(len(keys),int))[b]=apply_choices(published,choice)
                    traces.setdefault(arm,[]).append(choice)
                visible=np.asarray(order[batch][:budget],int)
                router.update(truth[visible],{k:v[visible] for k,v in pred.items()})
            times[f"{budget}-{seed}"]=perf_counter()-tic
            if expected and budget==32 and any(not np.array_equal(streams[f"{m}-32-{seed}"],expected[f"cumulative-{m}-{seed}"]) for m in ("global","local")):
                raise ValueError("重放与原始冻结决策不同")
    sealed=folder/"issued.json"
    atomic_json(sealed,{"keys":groups,"traces":traces,"predictions":{k:v.tolist() for k,v in streams.items()}})
    stored=read(sealed)["predictions"]
    table={k:scores(truth,np.array(v),blocks,classes) for k,v in stored.items()}
    # 事后给每个基础类别选择净增加正确数最多的固定候选，按正确数有利，未优化 Macro-F1。
    posthoc=FeedbackRouter(names,classes)
    posthoc.update(truth,pred)
    choices=posthoc.choices("local")
    table["posthoc-local"]=scores(truth,apply_choices(pred,choices),blocks,classes)
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"fixed":names,"arms":table,
                                      "posthoc_choices":choices,"measured_seconds":times})
    for k in ("current","vote","posthoc-local",*[f"{m}-{budgets[-1]}-{s}" for m in ("local","relabel") for s in (17,47,89)]):
        a=table[k]
        print(k,*(round(a[f],4) for f in ("equal_batches","pooled","worst_batch")),flush=True)


def main():
    fixed=read(OUT/"fixed.json")
    batches=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    y=np.asarray([r.label for b in batches for r in b.records])
    if fixed["keys"]!=[[r.sample_id for r in b.records] for b in batches]:
        raise ValueError("回标库分批身份不同")
    replay(OUT/"controls",{k:np.array(v) for k,v in fixed["predictions"].items()},y,fixed["keys"],(0,8,16,32),6,
           expected=read(OUT/"issued.json")["predictions"])


if __name__=="__main__":
    main()
