"""给同一在用混合模型施加由过去回标估计的类别补偿，先交付，再揭晓本批标签。"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import process_time
import json

import numpy as np
from sklearn.metrics import f1_score,recall_score

from engineering.cli.workflow.field_classification import EXPERTS, checked, source_experts
from harness.contracts import digest
from harness.models.field_classification import FieldClassifier, model_scores
from harness.models.vote_feedback import majority
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import atomic_json
from score_feedback_refit import sources
from prepare_temperature_trials import ROOT, DEST as TEMPERATURE


PROTOCOL = ROOT/"engineering/configs/experiments/field-recall-floor.json"
OUT = ROOT/"results/experiments/local-defaults/whole-maintenance/recall-floor"
PREVIOUS = ROOT/"results/experiments/local-defaults/whole-maintenance/field-delivery"
DYNAMIC = ROOT/"data/prepared/gas/dynamic-mixtures"


def read(path):
    return json.loads(path.read_bytes())


class RecallFloor:
    def __init__(self, classes):
        self.classes = classes
        self.margins = [[] for _ in range(classes)]
        self.hits = np.zeros((2, classes),dtype=int)
        self.first = None

    @staticmethod
    def corrected(logs, offsets):
        return (logs+offsets).argmax(axis=1)+1

    def offsets(self):
        warned = np.zeros(self.classes)
        fixed = np.zeros(self.classes)
        flags = []
        for c, margins in enumerate(self.margins):
            n = len(margins)
            if n < 5:
                continue
            ordered = np.sort(margins)
            def needed(target_recall):
                # 达到目标计数的最小加分；只在浮点同分处向外取一个小增量。
                rank = int(np.ceil(n*target_recall))
                if rank <= 0 or ordered[rank-1] < 0:
                    return 0.
                return float(np.clip(ordered[rank-1]+1e-12,0,np.log(4.)))
            fixed[c] = needed(.95)
            advantage = self.hits[1,c]-self.hits[0,c]
            if advantage >= 2 and advantage/n >= .02:
                flags.append(c+1)
                warned[c] = needed(self.hits[1,c]/n-.02)
        if self.first is None and warned.any():
            self.first = warned.copy()
        return flags,warned,fixed

    def update(self,logs,observer,labels):
        native = logs.argmax(axis=1)+1
        for c in range(self.classes):
            mask = labels==c+1
            if not mask.any():
                continue
            competitors=np.delete(logs[mask],c,axis=1).max(axis=1)
            self.margins[c].extend((competitors-logs[mask,c]).tolist())
            self.hits[0,c]+=int(np.sum(native[mask]==c+1))
            self.hits[1,c]+=int(np.sum(observer[mask]==c+1))


def external(folder):
    spec=read(folder/"stream.json")
    source=checked(folder/spec["source"],spec["source_sha256"])
    x=np.asarray([r["values"] for r in source])
    y=np.asarray([r["label"] for r in source])
    policy=FieldClassifier(x,y,EXPERTS,spec["classes"],"ongoing")
    vote=source_experts(x,y,policy.lda)
    def stream():
        for batch in spec["batches"]:
            obs=checked(folder/batch["inputs"],batch["inputs_sha256"])
            keys=[r["id"] for r in obs]
            values=np.asarray([r["values"] for r in obs])
            votes=vote(values)
            def reveal(selected, b=batch, k=keys):
                lab=checked(folder/b["feedback"],b["feedback_sha256"])
                return [lab[k[i]] for i in selected]
            yield keys,values,votes,keys,values,votes,reveal
    return policy,stream(),lambda:read(folder/"evaluation.json"),32,len(spec["batches"])


def official():
    name,x,y,rows,old,truth,budget,ncls=next(sources())
    if name!="public":
        raise ValueError("缺官方数据流")
    policy=FieldClassifier(x,y,tuple(old[0][1]),ncls,"ongoing")
    def stream():
        for ids,votes,pool,fp,labels in old:
            yield ids,np.asarray([rows[k] for k in ids]),votes,pool,np.asarray([rows[k] for k in pool]),fp,lambda ix,y=labels:[y[i] for i in ix]
    return policy,stream(),lambda:truth,budget,len(old)


def evaluate(name,suite):
    policy,stream,truth,budget,total=suite
    folder=OUT/name
    ep=LocalAcceptance()
    ep.output=folder/"accepted"
    guard=RecallFloor(policy.classes)
    once=recent="observer"
    history=[]
    seen=set()
    returned_batches=[]
    protocol=sha256(PROTOCOL.read_bytes()).hexdigest()
    extra_cpu=0.
    for j,(keys,x,votes,pool,xp,fp,reveal) in enumerate(stream):
        p=policy.predict(x,votes)
        raw=policy.models()["mixed"]
        logs=np.log(np.maximum(model_scores(raw,x,policy.classes),np.finfo(float).tiny))
        tic=process_time()
        flags,offsets,uniform=guard.offsets()
        new={"adjust":guard.corrected(logs,offsets), "flat-target":guard.corrected(logs,uniform),
             "first-adjust":guard.corrected(logs,guard.first if guard.first is not None else np.zeros(policy.classes)),
             "fallback":p["observer"] if flags else p["mixed"]}
        extra_cpu+=process_time()-tic
        all_predictions={**p,**new,"ongoing":p[policy.current()],"once":p[once],"recent-backtest":p[recent],
                         "vote":majority(votes,policy.classes),**{"source-"+a:v for a,v in votes.items()}}
        if set(keys)&seen or len(keys)!=len(set(keys)):
            raise ValueError("服务行重复")
        seen.update(keys)
        payload={"batch":j+1,"ids":list(keys),"flags":flags,"offsets":offsets.tolist(),"counts":[len(a) for a in guard.margins],
                 "old_hits":guard.hits.tolist(),"predictions":{a:v.tolist() for a,v in all_predictions.items()}}
        request=digest([protocol,name,payload])
        receipt=ep.submit(request,payload)
        if not receipt.get("accepted") or receipt["request_id"]!=request:
            raise RuntimeError("本次交付没有被接受")
        history.append(request)
        ix=sorted(range(len(pool)),key=lambda i:digest([17,pool[i]]))[:budget] if j+1<total else []
        returned_batches.append([pool[i] for i in ix])
        if ix:
            xx=xp[ix]; fb={a:v[ix] for a,v in fp.items()}
            before=policy.predict(xx,fb)
            scores=model_scores(raw,xx,policy.classes)
            yy=np.asarray(reveal(ix),int)
            tic=process_time()
            guard.update(np.log(np.maximum(scores,np.finfo(float).tiny)),before["observer"],yy)
            extra_cpu+=process_time()-tic
            old=policy.loss.copy()
            policy.update(xx,fb,yy,before)
            if j==0:
                once=policy.current()
            from harness.models.field_classification import TOOLS,macro
            recent_scores=[macro(v) for v in policy.loss-old]
            which=int(np.argmax(recent_scores))
            if recent_scores[which]>=recent_scores[TOOLS.index(recent)]+.01:
                recent=TOOLS[which]
    actual=truth()
    by_arm=defaultdict(list)
    y=[]
    alerts=Counter()
    alerts_on_harm=0
    cases_of_harm=0
    first=None
    for request in history:
        record=read(ep.output/(request+".json"))
        p=record["payload"]
        if digest([protocol,name,p])!=request or record["request_id"]!=request:
            raise ValueError("发放数据摘要变化")
        labels=np.asarray([actual[k] for k in p["ids"]])
        y.extend(labels.tolist())
        preds=p["predictions"]
        for a,values in preds.items():
            by_arm[a].extend(values)
        for c in range(1,policy.classes+1):
            subset=labels==c
            if not subset.any():
                continue
            worse=np.sum(np.asarray(preds["mixed"])[subset]==c)<np.sum(np.asarray(preds["observer"])[subset]==c)
            cases_of_harm+=int(worse)
            if c in p["flags"]:
                alerts[c]+=1
                alerts_on_harm+=int(worse)
                if first is None: first=p["batch"]
    yy=np.asarray(y)
    classes=np.arange(1,policy.classes+1)
    table={}
    for a,values in by_arm.items():
        recalls=recall_score(yy,values,labels=classes,average=None,zero_division=0)
        table[a]={"macro":float(f1_score(yy,values,labels=classes,average="macro",zero_division=0)),
                  "recall":recalls.tolist(),"worst":float(recalls.min())}
    ref_output=PREVIOUS/name
    if name in ("official","co","methane"):
        ref=read(ref_output/"scores.json")
        oldrun=ref_output/"ongoing"/"run.json" if name!="official" else ref_output/"ongoing/gas/run.json"
        olds=read(oldrun)["accepted"]
        for j,request in enumerate(history):
            oldp=read(oldrun.parent/"accepted"/(olds[j]["request"]+".json"))["payload"]
            newp=read(ep.output/(request+".json"))["payload"]
            if olds[j]["returned"]!=returned_batches[j]:
                raise ValueError("比较的回标身份或时钟变化")
            for a in ("observer","mixed","lda","recent-tree"):
                baseline=dict(zip(oldp["candidate_ids"],oldp["candidate_predictions"][a],strict=True))
                if baseline!=dict(zip(newp["ids"],newp["predictions"][a],strict=True)):
                    raise ValueError(f"{name} {a} 历史输出未复现")
        if any(abs(table[a]["macro"]-ref["scores"][a]["macro_f1"])>1e-12 for a in ref["scores"]):
            raise ValueError("历史对照分数未复现")
    controls=[a for a in table if a!="adjust"]
    best=max(controls,key=lambda a:table[a]["macro"])
    different=np.asarray(by_arm["adjust"])!=np.asarray(by_arm["mixed"])
    result={"protocol_sha256":protocol,"rounds":history,"labels":sum(map(len,returned_batches)),"service":len(y),
            "table":table,"best_control":best,"gate":table["adjust"]["macro"]>=table[best]["macro"]+.01 and
            table["adjust"]["worst"]>=table[best]["worst"]-.02,"warning_evaluable_class_batches":sum(alerts.values()),
            "warning_by_class":dict(alerts),"warnings_followed_by_harm":alerts_on_harm,"all_harm_class_batches":cases_of_harm,
            "first_evaluable_warning":first,"changed":int(different.sum()),
            "rescued":int(np.sum(different & (by_arm["adjust"]==yy))),"spoiled":int(np.sum(different & (by_arm["mixed"]==yy))),
            "candidate_fits":policy.fits,"shared_fit_cpu_seconds":policy.cpu,"extra_fits":0,"repair_cpu_seconds":extra_cpu}
    atomic_json(folder/"scores.json",result)
    print(name,len(y),{a:round(table[a]["macro"],5) for a in ("mixed","adjust","flat-target","fallback","ongoing",best)},
          result["changed"],result["rescued"],result["spoiled"],flush=True)
    return result


if __name__=="__main__":
    results={"official":evaluate("official",official())}
    for site,folder in (("co",DYNAMIC),("methane",DYNAMIC),("full",TEMPERATURE),("replacement",TEMPERATURE)):
        results[site]=evaluate(site,external(folder/site))
    atomic_json(OUT/"scores.json",results)
