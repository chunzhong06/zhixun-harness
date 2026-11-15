"""按目录中的批次清单交付现场分类结果，确认后开放该批回标，支持从已确认位置续接。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import process_time, perf_counter
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from harness.contracts import digest
from harness.models import field_classification as field
from harness.models.vote_feedback import majority
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json


EXPERTS = ("current","logistic","forest","lda","svm")


def checked(path, expected):
    data=path.read_bytes()
    if sha256(data).hexdigest()!=expected:
        raise ValueError(f"输入内容已变：{path.name}")
    return json.loads(data)


def source_experts(x,y,lda):
    small=RandomForestClassifier(n_estimators=1,random_state=0,n_jobs=1).fit(x,y)
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",
                               random_state=17,max_iter=1000)).fit(x,y)
    forest=RandomForestClassifier(n_estimators=100,max_depth=6,class_weight="balanced",random_state=17,n_jobs=1).fit(x,y)
    svm=make_pipeline(StandardScaler(),SVC(C=1.,class_weight="balanced")).fit(x,y)
    def votes(values):
        return {"current":small.predict(values),"logistic":logistic.predict(values),"forest":forest.predict(values),
                "lda":1+field.model_scores(lda,values,len(np.unique(y))).argmax(axis=1),"svm":svm.predict(values)}
    return votes


def run(manifest,out,mode,budget,through=None):
    spec=json.loads(manifest.read_bytes())
    batches=spec["batches"]
    end=len(batches) if through is None else through
    if not 0<=end<=len(batches) or budget<0 or len({r["batch"] for r in batches})!=len(batches):
        raise ValueError("交付终点、批次身份或回标预算无效")
    folder=manifest.parent
    input_files=[folder/spec["source"],manifest]
    input_files.extend(folder/r[key] for r in batches for key in ("inputs","feedback"))
    if out.is_relative_to(folder) or any(p.resolve().is_relative_to(out) for p in input_files):
        raise ValueError("输入与输出目录必须独立")
    start=perf_counter()
    source=checked(folder/spec["source"],spec["source_sha256"])
    train_ids={r["id"] for r in source}
    if len(train_ids)!=len(source):
        raise ValueError("源训练身份重复")
    identity=digest([spec,field.implementation(),sha256(Path(__file__).read_bytes()).hexdigest(),mode,budget,17])
    store=LocalStore(out/"checkpoint")
    endpoint=LocalAcceptance()
    endpoint.output=out/"accepted"
    with store.exclusive():
        saved=store.load() or {"identity":identity,"accepted":[],"field":None,
                               "cost":{"source_expert_builds":0,"candidate_fits":0,"fit_cpu_seconds":0.}}
        if saved["identity"]!=identity or len(saved["accepted"])>end:
            raise ValueError("恢复时模式、来源、预算或运行终点不同")
        seen=set(train_ids)
        for entry in saved["accepted"]:
            rec=json.loads((endpoint.output/(entry["request"]+".json")).read_bytes())
            if rec["request_id"]!=entry["request"] or digest([identity,rec["payload"]])!=entry["request"]:
                raise ValueError("已交接条目无法重放")
            seen.update(rec["payload"]["candidate_ids"])
        if len(saved["accepted"])<end:
            x=np.asarray([r["values"] for r in source])
            y=np.asarray([r["label"] for r in source])
            tic=process_time()
            policy=field.FieldClassifier(x,y,EXPERTS,spec["classes"],mode,saved["field"])
            predict=source_experts(x,y,policy.lda)
            saved["cost"]["source_expert_builds"]+=1
            saved["cost"]["fit_cpu_seconds"]+=process_time()-tic
        for j in range(len(saved["accepted"]),end):
            b=batches[j]
            rows=checked(folder/b["inputs"],b["inputs_sha256"])
            if any(set(r)!={"id","values"} for r in rows):
                raise ValueError("服务输入只接收身份和测量")
            ids=[r["id"] for r in rows]
            if not ids or set(ids)&seen or len(set(ids))!=len(ids):
                raise ValueError("测试、源训练或已交付记录重叠")
            values=np.asarray([r["values"] for r in rows],dtype=float)
            votes=predict(values)
            candidates=policy.predict(values,votes)
            chosen=policy.current()
            payload={"round":b["batch"],"candidate_ids":ids,"selected_tool":chosen,
                     "inputs_sha256":b["inputs_sha256"],"feedback_seen":len(policy.answers),
                     "predictions":dict(zip(ids,candidates[chosen].tolist(),strict=True)),
                     "candidate_predictions":{k:v.tolist() for k,v in candidates.items()},
                     "source_predictions":{k:v.tolist() for k,v in votes.items()},
                     "vote":majority(votes,spec["classes"]).tolist()}
            request=digest([identity,payload])
            receipt=endpoint.submit(request,payload)
            if not receipt["accepted"] or receipt["request_id"]!=request:
                raise RuntimeError("接收端尚未确认，暂不开放回标")
            # 首次读取此批答案发生在 submit 成功之后。末批没有后续预测，保持未回标。
            selected=sorted(range(len(ids)),key=lambda i:digest([17,ids[i]]))[:budget] if j+1<len(batches) else []
            if selected:
                labels=checked(folder/b["feedback"],b["feedback_sha256"])
                returned_x=values[selected]
                policy.update(returned_x,{k:v[selected] for k,v in votes.items()},[labels[ids[i]] for i in selected],
                              {k:v[selected] for k,v in candidates.items()})
            seen.update(ids)
            saved["accepted"].append({"round":b["batch"],"request":request,"returned":[ids[i] for i in selected]})
            saved["field"]=policy.state()
            saved["cost"]["candidate_fits"]+=policy.fits
            saved["cost"]["fit_cpu_seconds"]+=policy.cpu
            policy.fits,policy.cpu=0,0.
            store.save(saved)
        result={"identity":identity,"mode":mode,"budget":budget,"accepted":saved["accepted"],
                "cost":saved["cost"],"this_invocation_seconds":perf_counter()-start}
        atomic_json(out/"run.json",result)
        return result


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--stream",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--mode",required=True,choices=field.MODES)
    parser.add_argument("--feedback",type=int,default=32)
    parser.add_argument("--through",type=int)
    args=parser.parse_args(argv)
    result=run(args.stream.resolve(),args.output.resolve(),args.mode,args.feedback,args.through)
    print(f"已交付 {len(result['accepted'])} 批，当前模式 {args.mode}。",flush=True)
