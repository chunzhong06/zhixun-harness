"""复用已发出的分类预测，对比不同回标合并方法及原有实时选择。"""
from hashlib import sha256

import numpy as np
import pandas as pd

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,atomic_json,digest,read,read_gas_batches
from harness.models.feedback_routing import apply_choices
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import VoteFeedbackRouter,majority
from score_guarded_gas import SEEDS,external,public_stream
from score_new_electronic_nose import load_rows
from score_twin_transfer import DAYS,FIXED


def evaluate(name,stream,truth,budget,classes):
    keys=[k for ids,*_ in stream for k in ids]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(r[0]) for r in stream])[:-1])
    if len(set(keys))!=len(keys):
        raise ValueError("服务编号重复")
    predictions={}
    for seed in SEEDS:
        names=tuple(stream[0][1])
        full=ObserverErrors(names,classes)
        recent=ObserverErrors(names,classes)
        vote=VoteFeedbackRouter(names,classes)
        for block,(ids,p,returned,fp,y) in zip(blocks,stream,strict=True):
            local=apply_choices(vote.bank(p),vote.choices("vote-local"))
            arms={"current":p["current"],"vote":majority(p,classes),f"vote-local-{seed}":local,
                  f"last-{seed}":recent.predict(p,"uniform")}
            for policy in ("uniform","frequency","accuracy"):
                arms[f"{policy}-{seed}"]=full.predict(p,policy)
            for k,v in arms.items():
                predictions.setdefault(k,np.empty(len(keys),int))[block]=v
            ix=np.asarray(sorted(range(len(returned)),key=lambda i:digest([seed,returned[i]]))[:budget],dtype=int)
            sampled={k:v[ix] for k,v in fp.items()}
            revealed=[y[i] for i in ix]
            full.update(revealed,sampled)
            recent=ObserverErrors(names,classes)
            recent.update(revealed,sampled)
            vote.update(revealed,sampled)
    folder=OUT/"observer-errors"/name
    path=folder/"issued.json"
    atomic_json(path,{"keys":keys,"predictions":{k:v.tolist() for k,v in predictions.items()}})
    target=np.asarray([truth[k] for k in keys])
    table={k:scores(target,np.asarray(v),blocks,classes) for k,v in read(path)["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
    metrics=("pooled","equal_batches","worst_batch")
    avg={}
    for policy in ("current","vote","vote-local","last","uniform","frequency","accuracy"):
        runs=[table[policy]] if policy in table else [table[f"{policy}-{s}"] for s in SEEDS]
        avg[policy]={m:float(np.mean([r[m] for r in runs])) for m in metrics}
    print(name,{k:round(v["pooled"],4) for k,v in avg.items()},flush=True)
    return avg


def datasets():
    stream,truth,_=public_stream()
    yield "public",stream,truth,32,6
    batches=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    truth={r.sample_id:r.label for b in batches if b.batch_number>=3 for r in b.records}
    yield "late",external(OUT/"fixed.json",truth,GasFeedbackBank.names),truth,32,6
    y,labels,groups=load_rows()
    truth=dict(zip((k for row in groups for k in row),y,strict=True))
    folder=OUT/"new-instrument"
    yield "one-year",external(folder/"issued.json",truth,read(folder/"scores.json")["fixed"]),truth,4,len(labels)
    source=ROOT/"data/public/twin-gas-arrays/twin_features.csv"
    if sha256(source.read_bytes()).hexdigest()!="81cc1a20a36d5d87dc2d453e75bf5279daf37e76584f7b9ac9ba9e82d418ad65":
        raise ValueError("传感器板作者表已改变")
    data=pd.read_csv(source)
    rows=data[[f"f{k:02d}" for k in range(64)]].to_numpy(dtype="<f8")
    key=["twin-"+sha256(r.tobytes()).hexdigest() for r in rows]
    truth=dict(zip(key,data.gas.map({"GCO":1,"GEa":2,"GEy":3,"GMe":4}),strict=True))
    for protocol in ("archived","chronological"):
        for board in range(2,6):
            for budget in (4,32):
                loc=f"{protocol}/board-{board}-labels-{budget}"
                directory=OUT/"twin-author-features"/loc
                path=directory/"issued.json"
                if sha256(path.read_bytes()).hexdigest()!=read(directory/"scores.json")["issued_sha256"]:
                    raise ValueError("历史传感器板预测身份变化")
                issued=read(path)
                ids=issued["keys"]
                groups=[]
                for rep,day in enumerate(DAYS[board],1):
                    if protocol=="chronological" and day<=4:
                        continue
                    groups.append([key[i] for i in np.flatnonzero((data.board==board)&(data.rep==rep))])
                if [k for g in groups for k in g]!=ids:
                    raise ValueError("按日交付的传感器板查询顺序不同")
                stream=[]
                pos=0
                for g in groups:
                    p={a:np.asarray(issued["predictions"][a][pos:pos+len(g)]) for a in FIXED}
                    stream.append((g,p,g,p,[truth[k] for k in g]))
                    pos+=len(g)
                yield loc,stream,truth,budget,4


if __name__=="__main__":
    results={name:evaluate(name,stream,truth,budget,classes) for name,stream,truth,budget,classes in datasets()}
    atomic_json(OUT/"observer-errors/scores.json",results)
