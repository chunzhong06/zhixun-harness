"""在相同已见温控读数和历史八个回标上，比较源模型与现场图传播；独立资料继续封存。"""
import json
import numpy as np
from sklearn.metrics import f1_score,recall_score

from score_gas_global_calibration import DEST as PREV, ENTRY, FIRST, CHANNELS, INPUT, read, digest, LocalStore, LocalAcceptance, atomic_json
from harness.methods.gas_phase_graph import propagate, LEVELS, PRIMARY, PAPER

DEST=FIRST.parent/"gas-phase-graph"
ANCHOR="相位电阻合训"
ARMS=(ANCHOR,*[str(v) for v in LEVELS],PAPER)


def run_one(site,policy,view,bank):
    rel,_,tc=CHANNELS[site]
    root=INPUT/rel
    old=ENTRY/site/policy/view
    previous=read(old/"run.json")
    out=DEST/site/policy/view
    receiver=LocalAcceptance()
    receiver.output=out/"accepted"
    identity=digest([previous["identity"],"phase-graph-01"])
    shown,source,anchors,labels,events=[],[],[],[],[]
    evidence={k:np.zeros((3,3)) for k in ARMS}
    offset=0
    for event in previous["events"]:
        key=event["request"]
        p=read(old/"accepted"/(key+".json"))["payload"]
        if digest([previous["identity"],p])!=key or p["已见反馈"]!=len(labels):
            raise ValueError("服务事件与已知标签错位")
        ids=p["ids"]
        vals=np.stack([bank[k][1:,tc].ravel() for k in ids])
        shown.append(vals)
        cols=p["预测"]
        hz=np.eye(3)[np.array(cols[ANCHOR])-1]
        prior=.65*hz+.35*np.mean([np.eye(3)[np.array(cols[k])-1] for k in ("相位电阻仅源","相位电阻仅现场","相位比合训")],axis=0)
        source.append(prior)
        rescored=propagate(np.vstack(shown),np.vstack(source),np.asarray(anchors,dtype=int),np.asarray(labels,dtype=int))
        pred={ANCHOR:np.array(cols[ANCHOR]),**{str(k):1+v[offset:].argmax(1) for k,v in rescored.items()}}
        def quality(k):
            cm=evidence[k]+12*evidence[ANCHOR]/max(1,evidence[ANCHOR].sum())
            den=cm.sum(0)+cm.sum(1)
            return np.divide(2*cm.diagonal(),den,out=np.zeros(3),where=den>0).mean()
        best=max(ARMS,key=quality)
        if quality(best)<quality(ANCHOR)+.01:best=ANCHOR
        pred[PRIMARY]=pred[best]
        item={"ids":ids,"预测":{k:v.tolist() for k,v in pred.items()},"采用":best,"已见回标":len(labels),"基础回执":key}
        request=digest([identity,item])
        if not receiver.submit(request,item)["accepted"]:raise ValueError("图传播未完成交接")
        events.append(request)
        if event["selected"]:
            selected=[ids.index(k) for k in event["selected"]]
            y=np.array(event["labels"])
            weights=np.ones(len(selected))
            if policy=="混合抽查":
                half=len(selected)//2
                weights[half:]=(len(ids)-half)/(len(selected)-half)
            anchors.extend(offset+j for j in selected);labels.extend(y.tolist())
            for k in ARMS:np.add.at(evidence[k],(y-1,pred[k][selected]-1),weights)
        offset+=len(ids)
    truth=read(root/"evaluation.json")
    y,cols=[],{}
    for key in events:
        p=read(receiver.output/(key+".json"))["payload"]
        if digest([identity,p])!=key:raise ValueError("回执被改动")
        y.extend(truth[k] for k in p["ids"])
        for k,v in p["预测"].items():cols.setdefault(k,[]).extend(v)
    scores={k:{"F1":float(f1_score(y,v,average="macro")),"最差召回":float(recall_score(y,v,average=None).min())} for k,v in cols.items()}
    atomic_json(out/"run.json",{"identity":identity,"requests":events,"scores":scores})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已验收图传播")
        with np.load(FIRST/"exposures.npz") as data:
            bank=dict(zip(data["ids"],data["features"],strict=True))
        previous=read(PREV/"scores.json")
        streams={s:{} for s in CHANNELS}
        for a in previous["means"]:
            q,view=a.split("-")
            for s in CHANNELS:streams[s][a]=run_one(s,q,view,bank)
        means={a:{k:{m:float(np.mean([v[a][k][m] for v in streams.values()])) for m in ("F1","最差召回")} for k in scores}
               for a,scores in streams["全通道"].items()}
        get=lambda a:means[a[0]][a[1]]["F1"]
        fixed=max(((a,k) for a in means if not a.startswith("BADGE") for k in ARMS),key=get)
        paper=max(((a,k) for a in means for k in ARMS if a.startswith("BADGE") or k==PAPER),key=get)
        main_arm=("混合抽查-均值",PRIMARY)
        f=max(get(fixed),previous["decision"]["固定F1"])
        p=max(get(paper),previous["decision"]["论文F1"])
        gap=get(main_arm)-max(f,p)
        d={"选法":main_arm,"现场F1":get(main_arm),"本项固定":fixed,"固定F1":f,"本项论文":paper,"论文F1":p,"最小领先":gap,
           "进独立":gap>=.01 and means[main_arm[0]][PRIMARY]["最差召回"]>=means[fixed[0]][fixed[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":d})
        print(json.dumps({"同一反馈上的传播":d},ensure_ascii=False))


if __name__=="__main__":main()
