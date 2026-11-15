"""作者循环只取已返回的32个答案；每次触发前用同一棵树封存整批100个服务预测。"""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd
from river.drift.binary import DDM
from sklearn.metrics import accuracy_score, f1_score, recall_score

from score_author_new_arrays import AwaitingResponse, IssuedForest, ObserverErrors, ROOT, atomic_json, author_definition, author_loop, digest


DATA = ROOT/"data/prepared/gas/temperature-trials"
BASE = ROOT/"results/experiments/local-defaults/whole-maintenance"
DEST = BASE/"temperature-author"
PROTOCOL = ROOT/"engineering/configs/experiments/temperature-site-training.json"
AUTHOR_HASH = "c56695ca92297f5b338af2d8c08dd9f49af71f5de474c925c8daa4ba10b70a70"
NAMES = ("current","logistic","forest","lda","svm")


def read(p):
    return json.loads(p.read_bytes())


class FullBatchForest(IssuedForest):
    def __init__(self, visible, batches, folder):
        super().__init__(visible,folder)
        self.batches=batches

    def predict(self,x):
        j=len(self.events)
        if j<len(self.keys) and x.index.tolist()==self.keys[j]:
            full=self.batches[j]
            output=pd.Series(self.tree.predict(full),index=full.index)
            event={"keys":full.index.tolist(),"predictions":output.tolist(),"fits_before":self.fit_count,
                   "author_input":self.keys[j]}
            atomic_json(self.folder/f"round-{j+1:02}.json",event)
            self.events.append(event)
            return output.loc[x.index].to_numpy()
        return self.tree.predict(x)


def evaluate(site):
    folder=DATA/site
    spec=read(folder/"stream.json")
    src=read(folder/spec["source"])
    names=[f"R{i+1}" for i in range(len(src[0]["values"]))]
    def to_frame(rows):
        return pd.DataFrame([r["values"] for r in rows],index=[r["id"] for r in rows],columns=names)
    xs=to_frame(src)
    ys=pd.Series([r["label"] for r in src],index=xs.index)
    batches=[]; visible=[]; answers=[]
    for j,b in enumerate(spec["batches"]):
        rows=read(folder/b["inputs"])
        xx=to_frame(rows)
        if set(xx.index)&set(xs.index):
            raise ValueError("源设备与服务日期重合")
        keys=xx.index.tolist()
        if j+1<len(spec["batches"]):
            kept=set(sorted(keys,key=lambda k:digest([17,k]))[:32])
            keys=[k for k in keys if k in kept]
        batches.append(xx)
        visible.append(keys)
        labels=read(folder/b["feedback"])
        answers.append(pd.Series([labels[k] for k in keys],index=keys))
    small=[b.loc[idx] for b,idx in zip(batches,visible)]
    out=DEST/site
    source_run=read(BASE/"recall-floor"/site/"scores.json")
    results={}
    for arm in ("ddm","hllm"):
        status,detail="complete",None
        model=FullBatchForest(visible,batches,out/arm)
        ddm=DDM(drift_threshold=.8,warm_start=3)
        calls=[]
        try:
            if arm=="hllm":
                h,method,calls=author_loop(site,output=DEST,columns=names,
                    context="Identify why classification of carbon monoxide concentration bands from gas-sensor resistance may have degraded.")
                method(h,model,xs,ys,small,answers,None,ddm,0,0)
            else:
                method=author_definition("study_2.py",AUTHOR_HASH,"adaptive_batch_with_drift_detection_ddm",
                                         {"accuracy_score":accuracy_score})
                method(model,xs,ys,small,answers,small[-1],ddm)
        except AwaitingResponse as missing:
            status,detail="waiting_driver",str(missing)
        except (ValueError,SyntaxError,TypeError,KeyError) as error:
            status,detail="failed",f"{type(error).__name__}: {error}"
        scores={}
        if status=="complete":
            if len(model.events)!=len(batches):
                raise ValueError("作者轨迹未覆盖全部服务日")
            observer=ObserverErrors(NAMES,3)
            native,calibrated,truth=[],[],[]
            all_y=read(folder/"evaluation.json")
            for j,event in enumerate(model.events):
                request=source_run["rounds"][j]
                p=read(BASE/"recall-floor"/site/"accepted"/(request+".json"))["payload"]
                if event["keys"]!=p["ids"]:
                    raise ValueError("作者和项目服务行未对齐")
                v={k:np.asarray(p["predictions"]["source-"+k]) for k in NAMES}
                v["current"]=np.asarray(event["predictions"])
                cal=observer.predict(v,"uniform")
                atomic_json(out/arm/f"issued-{j+1:02}.json",{"ids":event["keys"],"raw":event["predictions"],"calibrated":cal.tolist()})
                native.extend(event["predictions"]); calibrated.extend(cal.tolist()); truth.extend(all_y[k] for k in p["ids"])
                if j+1<len(batches):
                    ix=[p["ids"].index(k) for k in visible[j]]
                    observer.update(answers[j].to_numpy(),{k:t[ix] for k,t in v.items()})
            for tag,p in (("raw",native),("calibrated",calibrated)):
                scores[tag]={"macro":float(f1_score(truth,p,labels=[1,2,3],average="macro")),
                             "recall":recall_score(truth,p,labels=[1,2,3],average=None).tolist()}
        results[arm]={"status":status,"detail":detail,"requests":calls,"rounds":len(model.events),
                      "fits":model.fit_count,"attempts":model.fit_attempts,"scores":scores}
        print(site,arm,status,detail,scores,flush=True)
    atomic_json(out/"scores.json",{"protocol":sha256(PROTOCOL.read_bytes()).hexdigest(),"arms":results})


if __name__=="__main__":
    for name in ("full","replacement"):
        evaluate(name)
