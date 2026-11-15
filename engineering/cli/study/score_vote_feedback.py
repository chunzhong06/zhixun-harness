"""在固定投票与原版启动的分类选择之间做配对比较，先交付后读取抽中反馈。"""
from hashlib import sha256

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,atomic_json,digest,read,read_gas_batches
from harness.models.feedback_routing import FeedbackRouter,apply_choices
from harness.models.vote_feedback import VoteFeedbackRouter
from harness.models.gas_feedback_bank import GasFeedbackBank


def experiment(folder,pred,keys,truth,budgets,classes):
    names=list(pred)
    blocks=np.split(np.arange(len(truth)),np.cumsum([len(b) for b in keys])[:-1])
    all_ids=[i for b in keys for i in b]
    issued=dict(pred)
    choices={}
    for seed in (17,47,89,*range(101,111)):
        for budget in budgets:
            old=FeedbackRouter(names,classes)
            route=VoteFeedbackRouter(names,classes)
            for j,b in enumerate(blocks):
                forecasts={a:v[b] for a,v in pred.items()}
                votes=route.bank(forecasts)
                issued.setdefault("vote",np.empty(len(truth),int))[b]=votes["current"]
                for mode in ("local","vote-local","vote-confirm"):
                    selections=old.choices(mode) if mode=="local" else route.choices(mode)
                    data=forecasts if mode=="local" else votes
                    arm=f"{mode}-{budget}-{seed}"
                    issued.setdefault(arm,np.empty(len(truth),int))[b]=apply_choices(data,selections)
                    choices.setdefault(arm,[]).append(selections)
                ix=np.asarray(sorted(b,key=lambda k:digest([seed,all_ids[k]]))[:budget],int)
                returned={a:v[ix] for a,v in pred.items()}
                route.update(truth[ix],returned)
                old.update(truth[ix],returned)
    path=folder/"issued.json"
    atomic_json(path,{"keys":keys,"choices":choices,"predictions":{k:v.tolist() for k,v in issued.items()}})
    table={k:scores(truth,np.asarray(v),blocks,classes) for k,v in read(path)["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table,"fixed":names})
    return table


def main():
    fixed=read(OUT/"fixed.json")
    data=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    y=np.asarray([r.label for b in data for r in b.records])
    if fixed["keys"]!=[[r.sample_id for r in b.records] for b in data]:
        raise ValueError("来源清单不匹配")
    pred={k:np.asarray(fixed["predictions"][k]) for k in GasFeedbackBank.names}
    results=experiment(OUT/"vote",pred,fixed["keys"],y,(32,),6)
    print_result("UCI",results,32)
    from score_new_electronic_nose import load_rows
    y,names,keys=load_rows()
    source=read(OUT/"new-instrument/issued.json")
    if keys!=source["keys"]:
        raise ValueError("独立仪器的日分组错误")
    pred={k:np.asarray(source["predictions"][k]) for k in read(OUT/"new-instrument/scores.json")["fixed"]}
    results=experiment(OUT/"new-instrument/vote",pred,keys,y,(4,),len(names))
    print_result("one-year",results,4)


def print_result(label,results,budget):
    print(label,"vote",results["vote"]["equal_batches"],results["vote"]["pooled"])
    for mode in ("local","vote-local","vote-confirm"):
        a=[results[f"{mode}-{budget}-{s}"] for s in (17,47,89,*range(101,111))]
        print(label,mode,{k:np.round([min(r[k] for r in a),np.mean([r[k] for r in a]),max(r[k] for r in a)],4).tolist() for k in ("pooled","equal_batches","worst_batch")})


if __name__=="__main__":
    main()
