"""固定服务流，仅改变允许返回的回标；另用相同次数、错开时机的配方检验收益归因。"""
from hashlib import sha256

import numpy as np
from sklearn.metrics import f1_score, recall_score

from harness.contracts import digest
from harness.models.site_training import RECIPES, SiteTraining
from harness.runtime.storage import atomic_json
from score_guarded_gas import SEEDS
from score_recall_floor import TEMPERATURE, official, external, read
from score_site_training import OUT, ROOT


PROTOCOL=ROOT/"engineering/configs/experiments/recipe-feedback-sensitivity.json"


def evaluate(name,suite):
    base, source_stream, truth, budget, n=suite
    stream=list(source_stream)
    actual=truth()
    ref=read(OUT/name/"scores.json")
    results=[]
    for seed in SEEDS:
        policy=SiteTraining(base.source,base.labels,base.observer.names,base.classes,"recipe-ongoing")
        ids=[]; predictions=[]; plans=[]; returned=[]; once=None
        for j,(keys,x,votes,pool,xp,fp,reveal) in enumerate(stream):
            before=policy.predict(x,votes)
            ids.extend(keys)
            plans.append(policy.current())
            predictions.append(before)
            if seed==17:
                p=read(OUT/name/"accepted"/(ref["requests"][j]+".json"))["payload"]
                if (p["ids"]!=list(keys) or p["chosen"]!=policy.current() or
                        any(before[a].tolist()!=p["predictions"][a] for a in RECIPES)):
                    raise ValueError(f"{name} 第{j+1}批可重载配方与独立实验不同")
            ix=sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:budget] if j+1<n else []
            returned.append([pool[i] for i in ix])
            if ix:
                xx=xp[ix]; v={k:vals[ix] for k,vals in fp.items()}
                old=policy.predict(xx,v)
                policy.update(xx,v,reveal(ix),old)
                if once is None:
                    once=policy.current()
        labels=np.asarray([actual[k] for k in ids])
        sequences={a:[a]*n for a in RECIPES}
        sequences["ongoing"]=plans
        sequences["once"]=[RECIPES[0]]+[once]*(n-1)
        sequences["fixed-midway"]=[RECIPES[0] if j<5 else RECIPES[1] for j in range(n)]
        classes=np.arange(1,base.classes+1)
        def issued(sequence):
            return np.concatenate([row[a] for row,a in zip(predictions,sequence,strict=True)])
        table={}
        for arm,seq in sequences.items():
            output=issued(seq)
            recalls=recall_score(labels,output,labels=classes,average=None,zero_division=0)
            table[arm]={"macro":float(f1_score(labels,output,labels=classes,average="macro",zero_division=0)),
                        "recall":recalls.tolist(),"worst":float(min(recalls))}
        rotations=[float(f1_score(labels,issued(plans[k:]+plans[:k]),labels=classes,average="macro",
                                  zero_division=0)) for k in range(n)]
        records=[{"keys":list(keys),"chosen":a,"issued":row[a].tolist(),
                  "predictions":{tool:row[tool].tolist() for tool in RECIPES},"returned_after":rv}
                 for (keys,*_),row,a,rv in zip(stream,predictions,plans,returned,strict=True)]
        issued_file=OUT/"draws"/name/f"seed-{seed}.json"
        atomic_json(issued_file,records)
        results.append({"seed":seed,"receipt_sha256":sha256(issued_file.read_bytes()).hexdigest(),"scores":table,
                        "rotations":rotations,"higher_rotations":sum(v>rotations[0]+1e-12 for v in rotations),
                        "tied_rotations":sum(abs(v-rotations[0])<=1e-12 for v in rotations),
                        "choices":plans,"fits":policy.fits,"cpu_seconds":policy.cpu})
    means={a:{k:float(np.mean([r["scores"][a][k] for r in results])) for k in ("macro","worst")}
           for a in results[0]["scores"]}
    summary={"protocol":sha256(PROTOCOL.read_bytes()).hexdigest(),"draws":results,"mean":means,
             "wins_vs_best_fixed":sum(r["scores"]["ongoing"]["macro"]>max(r["scores"][a]["macro"] for a in RECIPES) for r in results)}
    atomic_json(OUT/"draws"/name/"scores.json",summary)
    print(name, {a:round(s["macro"],5) for a,s in means.items()},summary["wins_vs_best_fixed"],flush=True)
    return summary


if __name__=="__main__":
    result={"official":evaluate("official",official())}
    for location in ("full","replacement"):
        result[location]=evaluate(location,external(TEMPERATURE/location))
    atomic_json(OUT/"draws/scores.json",result)
