"""在相同旧查询和服务输出上检验传感器分位读法与实际回标驱动的通道权重。"""
import json
from pathlib import Path
from hashlib import sha256
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"engineering/src"),str(ROOT)]
from score_gas_joint_query import (BASE, STREAMS, DEST as PREVIOUS, PRIOR as QUERY_RUNS, QUERIES,
                                   read, checked, LocalAcceptance, LocalStore, atomic_json, digest,
                                   f1_score, recall_score, TRAIN)
from harness.methods import gas_rank_readout
from harness.methods.gas_rank_readout import RankReadout,aggregate

DEST=PREVIOUS.parent/"gas-rank-readout"


def evaluate(name,location,trajectory):
    folder=BASE/location
    spec=read(folder/"stream.json")
    classes=spec["classes"]
    source=checked(folder/spec["source"],spec["source_sha256"])
    initial=RankReadout(np.array([r["values"] for r in source]),np.array([r["label"] for r in source]))
    old=PREVIOUS/name/trajectory
    run=read(old/"run.json")
    qfolder=QUERY_RUNS/name/trajectory/"17"
    queries=read(qfolder/"run.json")
    out=DEST/name/trajectory
    receiver=LocalAcceptance()
    receiver.output=out/"accepted"
    events,history,labels,seen,past=[],[],[],[],[]
    ident=digest([run["identity"],queries["identity"],sha256(Path(__file__).read_bytes()).hexdigest(),
                  sha256(Path(gas_rank_readout.__file__).read_bytes()).hexdigest()])
    for b,req,e in zip(spec["batches"],run["requests"],queries["accepted"],strict=True):
        p=read(old/"accepted"/(req+".json"))["payload"]
        if digest([run["identity"],p])!=req or p["查询回执"]!=e["request"]:
            raise ValueError("查询或发布轨迹与原条件不一致")
        rows=checked(folder/b["inputs"],b["inputs_sha256"])
        ids=[r["id"] for r in rows]
        if p["ids"]!=ids:
            raise ValueError("旧现场分类不是同一组测量")
        values=np.asarray([r["values"] for r in rows])
        seen.append(values)
        bank=initial.predict(values,np.vstack(seen),np.array(history),np.array(labels,dtype=int))
        probabilities,w=aggregate(bank,past,labels)
        outputs=dict(p["predictions"])
        for k,probs in probabilities.items():
            outputs[k]=(1+probs.argmax(1)).tolist()
        # 固定同一额外工具与旧合训核各占一份，以及相同投票结构中的动态替代。
        kernel=np.eye(classes)[np.asarray(outputs["固定合训核"])-1]
        for k,probs in probabilities.items():
            outputs[k+"与合训核"]=((probs+.4*kernel).argmax(1)+1).tolist()
        # 直接固定使用所有单通道，从强固定集合中排除“挑通道”带来的虚假增量。
        for a in range(bank.shape[1]):
            half=bank.shape[1]//2
            domain="源" if a<half else "现场"
            outputs[f"定通道/{domain}/{a%half+1}"]=(1+bank[:,a].argmax(1)).tolist()
        payload={"ids":ids,"预测":outputs,"分量":w.tolist(),"已见回标":len(labels),"来源回执":req}
        key=digest([ident,payload])
        if not receiver.submit(key,payload)["accepted"]:
            raise ValueError("新读法未发出")
        events.append(key)
        if e["selected"]:
            truth=checked(folder/b["feedback"],b["feedback_sha256"])
            ix=[ids.index(k) for k in e["selected"]]
            history.extend(values[ix].tolist())
            labels.extend(truth[k] for k in e["selected"])
            past.append(bank[ix])
    oracle=read(folder/"evaluation.json")
    truth,issued=[],{}
    for key in events:
        p=read(receiver.output/(key+".json"))["payload"]
        if digest([ident,p])!=key:
            raise ValueError("交接后预测内容改变")
        truth.extend(oracle[k] for k in p["ids"])
        for arm,v in p["预测"].items():
            issued.setdefault(arm,[]).extend(v)
    klist=np.arange(1,classes+1)
    scores={k:{"F1":float(f1_score(truth,v,labels=klist,average="macro",zero_division=0)),
               "最差召回":float(recall_score(truth,v,labels=klist,average=None,zero_division=0).min())} for k,v in issued.items()}
    atomic_json(out/"run.json",{"identity":ident,"requests":events,"scores":scores,"回标":len(labels)})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("秩读法已完成")
        streams={s:{q:evaluate(s,path,q) for q in QUERIES} for s,path in STREAMS.items()}
        # 两个来源的通道数不同，同一个固定编号不能把现场通道误当源通道。
        # 汇总用跨来源有一致解释的规则；另给各来源可独立选最强单通道的对手。
        means={}
        for q in QUERIES:
            means[q]={}
            names=sorted(set.intersection(*(set(v[q]) for v in streams.values())))
            for k in names:
                means[q][k]={m:float(np.mean([v[q][k][m] for v in streams.values()])) for m in ("F1","最差召回")}
            best_channel={s:max((k for k in runs[q] if k.startswith("定通道")),key=lambda k:runs[q][k]["F1"])
                          for s,runs in streams.items()}
            means[q]["各流回看最强单通道"]={m:float(np.mean([streams[s][q][k][m] for s,k in best_channel.items()]))
                                             for m in ("F1","最差召回")}
        new=("按现场通道与域的可靠性","按现场通道与域的可靠性与合训核")
        dynamic=(*new,"全局历史发布","近邻历史发布")
        fixed=[(q,k) for q in QUERIES if q not in ("BADGE","历史辨别力") for k in means[q] if k not in dynamic]
        paper=[("BADGE",k) for k in means["BADGE"]]
        candidates=[(q,k) for q in QUERIES if q!="BADGE" for k in new]
        f=lambda a:means[a[0]][a[1]]["F1"]
        a,b,c=(max(group,key=f) for group in (candidates,fixed,paper))
        valid_fixed=max((arm for arm in fixed if arm[1]!="各流回看最强单通道"),key=f)
        valid_paper=max((arm for arm in paper if arm[1]!="各流回看最强单通道"),key=f)
        decision={"候选":a,"候选F1":f(a),"最强可行固定":valid_fixed,"可行固定F1":f(valid_fixed),
                  "论文查询最佳":valid_paper,"论文F1":f(valid_paper),"含事后单通道最强固定":b,"含事后单通道固定F1":f(b),
                  "开发通过":f(a)-max(f(valid_fixed),f(valid_paper))>=.01 and means[a[0]][a[1]]["最差召回"]>=means[valid_fixed[0]][valid_fixed[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":decision})
        print(json.dumps({"开发取舍":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
