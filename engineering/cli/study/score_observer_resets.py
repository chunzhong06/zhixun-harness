"""同仪器的连续发放，检验只凭刚到达的回标来触发混淆计数重置。"""
from hashlib import sha256
from statistics import mean

import numpy as np
from scipy.special import logsumexp

from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,read_gas_batches,read,digest,atomic_json
from score_guarded_gas import SEEDS,public_stream,external
from check_feedback_controls import scores
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.observer_errors import ObserverErrors


def loss(router,p,y):
    counts=router.counts+np.eye(router.classes,dtype=int)[None,:,:]+1
    rates=np.log(counts/counts.sum(2,keepdims=True))
    logits=sum(rates[i].T[v-1] for i,v in enumerate(p[k] for k in router.names))
    return float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y-1]))


def main():
    first,y1,_=public_stream()
    batches=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    y2={r.sample_id:r.label for b in batches if b.batch_number>=3 for r in b.records}
    second=external(OUT/"fixed.json",y2,GasFeedbackBank.names)
    stream=first+second
    arms=("carry","boundary","fixed-two","last-two","trigger-one","trigger-two")
    out={}
    events={}
    for seed in SEEDS:
        fresh=ObserverErrors(GasFeedbackBank.names,6)
        models={k:ObserverErrors(GasFeedbackBank.names,6) for k in arms}
        streak={k:0 for k in arms}
        increments=[]
        for j,(ids,p,fbid,fp,y) in enumerate(stream):
            ix=np.asarray(sorted(range(len(fbid)),key=lambda i:digest([seed,fbid[i]]))[:32])
            outputs={k:v[ix] for k,v in fp.items()}
            labels=np.asarray([y[i] for i in ix])
            next_counts=ObserverErrors(GasFeedbackBank.names,6)
            next_counts.update(labels,outputs)
            for k in arms:
                if (k=="boundary" and j==len(first)) or (k=="fixed-two" and j%2==0):
                    models[k]=ObserverErrors(GasFeedbackBank.names,6)
                model=models[k]
                forecast=model.predict(p,"uniform")
                name=f"{k}-{seed}"
                out.setdefault(name,[]).append(forecast.tolist())
                should_reset=False
                if k.startswith("trigger"):
                    worse=loss(model,outputs,labels)>loss(fresh,outputs,labels)+1e-12
                    streak[k]=streak[k]+1 if worse else 0
                    if streak[k]>=(1 if k=="trigger-one" else 2):
                        model=ObserverErrors(GasFeedbackBank.names,6)
                        streak[k]=0
                        should_reset=True
                if k=="last-two":
                    counts=next_counts.counts if not increments else increments[-1]+next_counts.counts
                    model=ObserverErrors(GasFeedbackBank.names,6,counts)
                else:
                    model.update(labels,outputs)
                models[k]=model
                events.setdefault(name,[]).append(should_reset)
            increments.append(next_counts.counts)
    dest=OUT/"observer-errors/online-reset"
    sealed=dest/"issued.json"
    atomic_json(sealed,{"keys":[r[0] for r in stream],"predictions":out,"reset_after_batch":events})
    stored=read(sealed)
    table={}
    for suite,part,truth,offset in (("public",first,y1,0),("late",second,y2,len(first))):
        target=np.asarray([truth[key] for g,*_ in part for key in g])
        blocks=np.split(np.arange(len(target)),np.cumsum([len(r[0]) for r in part])[:-1])
        scope={k:scores(target,np.concatenate(v[offset:offset+len(part)]),blocks,6) for k,v in stored["predictions"].items()}
        source=read(OUT/"observer-errors"/suite/"issued.json")["predictions"]
        if any(np.concatenate(stored["predictions"][f"boundary-{s}"][offset:offset+len(part)]).tolist()!=source[f"uniform-{s}"] for s in SEEDS):
            raise ValueError("已知边界的对照未复现两段现场输出")
        means={k:{m:mean(scope[f"{k}-{s}"][m] for s in SEEDS) for m in ("pooled","equal_batches","worst_batch")} for k in arms}
        table[suite]={"means":means,"arms":scope}
        print(suite,{k:round(v["pooled"],5) for k,v in means.items()})
    atomic_json(dest/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"suites":table})


if __name__=="__main__":
    main()
