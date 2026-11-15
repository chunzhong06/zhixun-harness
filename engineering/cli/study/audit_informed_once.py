"""同一反馈时钟上，在模型首次真正分歧且答案已到达时作一次选择，再与持续重选比较。"""
from hashlib import sha256

import numpy as np
from sklearn.metrics import f1_score,recall_score

from audit_recipe_draws import RECIPES, SEEDS, SiteTraining, TEMPERATURE, atomic_json, digest, external, official, read
from score_site_training import OUT, ROOT


PROTOCOL=ROOT/"engineering/configs/experiments/temperature-matched-sensors.json"


def evaluate(site,suite):
    base, observations, get_truth, budget, n=suite
    obs=list(observations)
    results=[]
    for seed in SEEDS:
        policy=SiteTraining(base.source,base.labels,base.observer.names,base.classes,"recipe-ongoing")
        frozen=None
        chosen_after=None
        service=[]
        returns=[]
        for j,(ids,x,votes,pool,xp,fp,reveal) in enumerate(obs):
            p=policy.predict(x,votes)
            output={k:p[k] for k in RECIPES}
            output["ongoing"]=p[policy.current()]
            output["informed-once"]=p[frozen or RECIPES[0]]
            output["fixed-local-after-two"]=p[RECIPES[0] if j<2 else "local-source"]
            service.append({"ids":list(ids),"ongoing":policy.current(),"once":frozen or RECIPES[0],
                            "predictions":{a:v.tolist() for a,v in output.items()}})
            ix=sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:budget] if j+1<n else []
            returns.append([pool[i] for i in ix])
            if ix:
                xx=xp[ix]; sampled={k:v[ix] for k,v in fp.items()}
                before=policy.predict(xx,sampled)
                had_choice=any(np.any(before[a]!=before[RECIPES[0]]) for a in RECIPES[1:])
                policy.update(xx,sampled,reveal(ix),before)
                if frozen is None and had_choice:
                    frozen=policy.current()
                    chosen_after=j+1
        path=OUT/"matched"/site/f"seed-{seed}.json"
        atomic_json(path,{"issued":service,"returned":returns})
        saved=read(path)["issued"]
        truth=get_truth()
        yy=[truth[k] for batch in saved for k in batch["ids"]]
        classes=range(1,base.classes+1)
        table={}
        vectors={}
        for k in saved[0]["predictions"]:
            vectors[k]=np.concatenate([b["predictions"][k] for b in saved])
            table[k]={"macro":float(f1_score(yy,vectors[k],labels=classes,average="macro",zero_division=0)),
                      "worst":float(min(recall_score(yy,vectors[k],labels=classes,average=None,zero_division=0)))}
        prior=OUT/"draws"/site/f"seed-{seed}.json"
        if prior.exists():
            original=read(prior)
            if any(a["keys"]!=b["ids"] or a["issued"]!=b["predictions"]["ongoing"] or
                   a["returned_after"]!=r for a,b,r in zip(original,saved,returns,strict=True)):
                raise ValueError("已研究的配方或回标未复现")
        results.append({"seed":seed,"scores":table,"chose_after":chosen_after,"once_recipe":frozen,
                        "changed":int(sum(vectors["ongoing"]!=vectors["informed-once"])),
                        "issued_sha256":sha256(path.read_bytes()).hexdigest()})
    means={a:{k:float(np.mean([row["scores"][a][k] for row in results])) for k in ("macro","worst")} for a in table}
    wins=sum(r["scores"]["ongoing"]["macro"]>r["scores"]["informed-once"]["macro"]+1e-12 for r in results)
    result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"rows":results,"mean":means,
            "ongoing_wins_vs_informed":wins,"identical_runs":sum(not r["changed"] for r in results)}
    print(site,{a:round(v["macro"],5) for a,v in means.items()},wins,result["identical_runs"],flush=True)
    atomic_json(OUT/"matched"/site/"scores.json",result)
    return result


if __name__=="__main__":
    r={"official":evaluate("official",official())}
    for name in ("full","a-to-a","replacement","b-to-a","b-to-b"):
        r[name]=evaluate(name,external(TEMPERATURE/name))
    atomic_json(OUT/"matched/scores.json",r)
