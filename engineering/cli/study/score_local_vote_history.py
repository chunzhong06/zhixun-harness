"""逐批用同一回标清单比较分类器的相关性和历史相似错误；额外轮换源设备。"""
from collections import defaultdict
from hashlib import sha256

import numpy as np
import pandas as pd

from check_feedback_controls import scores
from gas_complementary_feedback import OUT, ROOT, atomic_json, digest, read
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from score_guarded_gas import SEEDS
from score_observer_errors import datasets
from score_twin_transfer import DAYS, predict


DEST=OUT/"vote-history"
ARMS=("uniform","decorrelated","local","half","select","class-count","vote","all")


def decide(names, classes, votes, history, labels, observer):
    x=np.column_stack([votes[n] for n in names]).astype(int)-1
    base=majority(votes,classes)
    uniform=observer.predict(votes,"uniform")
    answer={k:base.copy() for k in ARMS}
    answer["uniform"]=uniform
    answer["all"]=votes.get("all",base)
    if not len(labels):
        return answer
    labeled=np.asarray(labels,dtype=int)
    hist=np.asarray(history,dtype=int)
    counts=observer.counts+1+np.eye(classes,dtype=int)[None,:,:]
    logp=np.log(counts/counts.sum(axis=2,keepdims=True))
    evidence=np.stack([logp[i].T[x[:,i]] for i in range(x.shape[1])])
    agreement=(hist[:,:,None]==hist[:,None,:]).mean(axis=0)
    weights=1/(agreement**2).sum(axis=1)
    def tied(table,tie):
        mx=table.max(axis=1)
        best=np.isclose(table,mx[:,None],atol=1e-12,rtol=0)
        return np.where(best[np.arange(len(tie)),tie-1],tie,best.argmax(1)+1)
    answer["decorrelated"]=tied((weights[:,None,None]*evidence).sum(axis=0),base)
    total=evidence.sum(axis=0)
    glob=np.exp(total-total.max(axis=1,keepdims=True))
    glob/=glob.sum(axis=1,keepdims=True)
    class_counts=np.bincount(labeled-1,minlength=classes)
    answer["class-count"]=tied(np.tile(class_counts,(len(x),1)),base)
    correct=hist==(labeled[:,None]-1)
    patterns,index=np.unique(x,axis=0,return_inverse=True)
    local=[]
    selected=[]
    for pattern in patterns:
        distance=(hist!=pattern).sum(axis=1)
        pos=min(5,len(hist))-1
        group=distance<=np.partition(distance,pos)[pos]
        prior=(np.bincount(pattern,minlength=classes)+1)/(len(pattern)+classes)
        c=np.bincount(labeled[group]-1,minlength=classes).astype(float)
        # 等类质量，保持邻域有效票数，使和全局后验的各半具有相同量纲。
        c=c*len(labeled)/(classes*np.maximum(1,class_counts))
        prob=c+prior
        local.append(prob/prob.sum())
        acc=correct[group].mean(axis=0)
        best=np.isclose(acc,acc.max(),atol=1e-12,rtol=0)
        selected.append(np.bincount(pattern[best],minlength=classes))
    loc=np.asarray(local)[index]
    answer["local"]=tied(loc,base)
    answer["half"]=tied((loc+glob)/2,base)
    answer["select"]=tied(np.asarray(selected)[index],base)
    return answer


def evaluate(folder,stream,truth,budget,classes):
    ids=[i for batch,*_ in stream for i in batch]
    if len(set(ids))!=len(ids):
        raise ValueError("服务观测重复")
    blocks=np.split(np.arange(len(ids)),np.cumsum([len(s[0]) for s in stream])[:-1])
    names=tuple(stream[0][1])
    predictions={}
    for seed in SEEDS:
        observer=ObserverErrors(names,classes)
        history,labels=[],[]
        output=defaultdict(list)
        for keys,votes,returned,fp,y in stream:
            picked=decide(names,classes,votes,history,labels,observer)
            for arm,p in picked.items():
                output[arm].extend(np.asarray(p).tolist())
            sample=sorted(range(len(returned)),key=lambda i:digest([seed,returned[i]]))[:budget]
            sampled={k:np.asarray(fp[k])[sample] for k in names}
            labs=[int(y[i]) for i in sample]
            observer.update(labs,sampled)
            history.extend(np.column_stack([sampled[k] for k in names])-1)
            labels.extend(labs)
        predictions[str(seed)]=dict(output)
    path=DEST/folder/"issued.json"
    atomic_json(path,{"keys":ids,"predictions":predictions,"labels_per_batch":budget})
    recorded=read(path)["predictions"]
    y=np.asarray([truth[i] for i in ids])
    table={seed:{a:scores(y,np.array(p),blocks,classes) for a,p in arms.items()} for seed,arms in recorded.items()}
    summary={a:{m:float(np.mean([r[a][m] for r in table.values()])) for m in ("pooled","worst_batch")} for a in ARMS}
    atomic_json(path.with_name("scores.json"),{"summary":summary,"arms":table,"issued_sha256":sha256(path.read_bytes()).hexdigest()})
    return summary


def rotate():
    source=ROOT/"data/public/twin-gas-arrays/twin_features.csv"
    if sha256(source.read_bytes()).hexdigest()!="81cc1a20a36d5d87dc2d453e75bf5279daf37e76584f7b9ac9ba9e82d418ad65":
        raise ValueError("传感器板记录身份变化")
    data=pd.read_csv(source)
    x=data[[f"f{i:02d}" for i in range(64)]].to_numpy(dtype=float)
    y=data.gas.map({"GCO":1,"GEa":2,"GEy":3,"GMe":4}).to_numpy(dtype=int)
    ids=np.array(["twin-"+sha256(r.astype("<f8").tobytes()).hexdigest() for r in x])
    board=data.board.to_numpy()
    days=np.array([DAYS[b][r-1] for b,r in zip(board,data.rep,strict=True)])
    total={}
    for start in (2,3,4,5):
        day=days[board==start].min()
        src=np.flatnonzero((board==start)&(days==day))
        test=np.flatnonzero((board!=start)&(days>day))
        p=predict(x,y,src,test)
        fold={}
        for target in np.unique(board[test]):
            stream=[]
            for d in sorted(set(days[test[board[test]==target]])):
                ix=np.flatnonzero((board[test]==target)&(days[test]==d))
                keys=ids[test[ix]].tolist()
                pred={k:v[ix] for k,v in p.items()}
                stream.append((keys,pred,keys,pred,y[test[ix]]))
            for budget in (4,32):
                branch=f"rotation/source-{start}/target-{target}/labels-{budget}"
                fold[f"{target}/{budget}"]=evaluate(branch,stream,dict(zip(ids[test],y[test],strict=True)),budget,4)
        total[str(start)]={"fit_day":int(day),"training_ids":ids[src].tolist(),"targets":fold}
    atomic_json(DEST/"rotation/scores.json",total)
    return total


def main():
    current={}
    for name,stream,truth,budget,classes in datasets():
        if name.startswith("archived/"):
            continue
        current[name]=evaluate("existing/"+name,stream,truth,budget,classes)
        reference=read(OUT/"observer-errors/scores.json")[name]["uniform"]["pooled"]
        if abs(current[name]["uniform"]["pooled"]-reference)>1e-12:
            raise ValueError("同回标清单下原混淆模型没有复现")
        print(name,{k:round(v["pooled"],4) for k,v in current[name].items()},flush=True)
    atomic_json(DEST/"existing/scores.json",current)
    rotated=rotate()
    for budget in (4,32):
        by_source=[{k:np.mean([v[k]["pooled"] for target,v in fold["targets"].items() if target.endswith("/"+str(budget))])
                    for k in ARMS} for fold in rotated.values()]
        print("rotation",budget,{k:round(np.mean([r[k] for r in by_source]),4) for k in ARMS})


if __name__=="__main__":
    main()
