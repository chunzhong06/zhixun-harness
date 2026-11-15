"""用相同现场反馈分离训练资料来源、标准化方式与自主挑选模型的贡献。"""
from hashlib import sha256
from time import process_time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,recall_score
from sklearn.preprocessing import StandardScaler

from harness.contracts import digest
from harness.models.field_classification import macro, weights
from harness.models.gas_feedback_bank import discriminate, fit_discriminant
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import atomic_json
from score_recall_floor import DYNAMIC, TEMPERATURE, ROOT, external, read
from score_feedback_refit import sources


PROTOCOL=ROOT/"engineering/configs/experiments/temperature-site-training.json"
OUT=ROOT/"results/experiments/local-defaults/whole-maintenance/site-training"
ARMS=("pooled-source","pooled-separate","local-source","local-separate")


def external_input(folder):
    policy, stream, truth, budget, n = external(folder)
    return policy.source,policy.labels,stream,truth,budget,n


def prior_input(study):
    name,x,y,rows,old,truth,budget,_=study
    x=np.asarray(x,dtype=float)
    use=np.isfinite(x).any(axis=0)
    med=np.nanmedian(np.where(np.isfinite(x[:,use]),x[:,use],np.nan),axis=0)
    def clean(v):
        subset=np.asarray(v)[:,use]
        return np.where(np.isfinite(subset),subset,med)
    def stream():
        for ids,votes,pool,fp,labels in old:
            yield ids,clean([rows[k] for k in ids]),votes,pool,clean([rows[k] for k in pool]),fp,lambda ix,y=labels:[y[i] for i in ix]
    return name,(clean(x),y,stream(),lambda:truth,budget,len(old))


def evaluate(name, inputs):
    source,sy,stream,truth,budget,total=inputs
    base_scale=StandardScaler().fit(np.asfortranarray(source))
    src=base_scale.transform(np.asfortranarray(source))
    source_w=weights(sy,len(sy))
    def fit(z,y,w):
        return LogisticRegression(C=1.,random_state=17,max_iter=1000).fit(z,y,sample_weight=w)
    start=process_time()
    source_model=fit(src,sy,source_w)
    cpu=process_time()-start
    history_x=[]
    history_y=[]
    nf=len(np.unique(sy))
    matrices=np.zeros((4,nf,nf),dtype=int)
    active=once=0
    ep=LocalAcceptance()
    ep.output=OUT/name/"accepted"
    protocol=sha256(PROTOCOL.read_bytes()).hexdigest()
    returned_ids=[]
    requests=[]
    fits=0
    for j,(ids,values,_,pool,feedback_x,_,reveal) in enumerate(stream):
        ix=sorted(range(len(pool)),key=lambda i:digest([17,pool[i]]))[:budget] if j+1<total else []
        # 先得到本轮所有模型的预测，再把反馈行的标签开放给选择器和下轮训练。
        both=np.vstack((values,feedback_x[ix]))
        test_src=base_scale.transform(both)
        cold=source_model.predict(test_src)
        result={k:cold for k in (*ARMS,"local-lda","local-tree")}
        if history_y:
            tx=np.asarray(history_x); ty=np.asarray(history_y)
            target_w=weights(ty,len(sy))
            tic=process_time()
            site_scale=StandardScaler().fit(np.asfortranarray(tx))
            for suffix,scale in (("source",base_scale),("separate",site_scale)):
                local=scale.transform(np.asfortranarray(tx))
                test=scale.transform(both)
                combined=fit(np.vstack((src,local)),np.r_[sy,ty],np.r_[source_w,target_w])
                result["pooled-"+suffix]=combined.predict(test)
                fits+=1
                if len(np.unique(ty))>=2:
                    only=fit(local,ty,2*target_w)
                    result["local-"+suffix]=only.predict(test)
                    fits+=1
            if len(ty)>len(np.unique(ty)) and len(np.unique(ty))>=2:
                result["local-lda"]=discriminate(fit_discriminant(tx,ty),both)
                fits+=1
            result["local-tree"]=RandomForestClassifier(n_estimators=1,random_state=0,n_jobs=1).fit(tx,ty).predict(both)
            fits+=1
            cpu+=process_time()-tic
        result["ongoing"]=result[ARMS[active]]
        result["once"]=result[ARMS[once]]
        payload={"round":j+1,"ids":list(ids),"labels_used":len(history_y),"source_rows":len(sy),
                 "chosen":ARMS[active], "first":ARMS[once],"predictions":{a:v[:len(ids)].tolist() for a,v in result.items()}}
        req=digest([protocol,name,payload])
        ack=ep.submit(req,payload)
        if not ack.get("accepted") or ack["request_id"]!=req:
            raise RuntimeError("未确认本次服务输出")
        requests.append(req)
        returned_ids.append([pool[i] for i in ix])
        if ix:
            labels=np.asarray(reveal(ix),dtype=int)
            for a,k in enumerate(ARMS):
                np.add.at(matrices[a],(labels-1,result[k][len(ids):]-1),1)
            scores=[macro(m) for m in matrices]
            best=int(np.argmax(scores))
            if scores[best]>=scores[active]+.01:
                active=best
            if j==0:
                once=active
            history_x.extend(feedback_x[ix].tolist())
            history_y.extend(labels.tolist())
    records=[read(ep.output/(r+".json"))["payload"] for r in requests]
    if any(digest([protocol,name,p])!=r for p,r in zip(records,requests,strict=True)):
        raise ValueError("已发预测身份不符")
    ids=[k for p in records for k in p["ids"]]
    if len(ids)!=len(set(ids)):
        raise ValueError("重复服务行")
    answers=truth()
    y=[answers[k] for k in ids]
    all_p={a:np.concatenate([p["predictions"][a] for p in records]) for a in records[0]["predictions"]}
    table={}
    for a,v in all_p.items():
        rr=recall_score(y,v,labels=range(1,nf+1),average=None,zero_division=0)
        table[a]={"macro":float(f1_score(y,v,labels=range(1,nf+1),average="macro",zero_division=0)),"recall":rr.tolist(),"worst":float(rr.min())}
    baseline=OUT.parent/"recall-floor"/name/"scores.json"
    prior={}
    if baseline.exists():
        old=read(baseline)
        if abs(old["table"]["mixed"]["macro"]-table["pooled-source"]["macro"])>1e-12:
            raise ValueError("同配方混合模型未复现")
        prior={a:old["table"][a] for a in ("mixed","observer","recent-tree","ongoing","recent-backtest")}
    else:
        old=read(ROOT/"results/experiments/local-defaults/gas-complementary-feedback/feedback-refit"/name/"scores.json")
        if abs(old["arms"]["refit"]["pooled"]-table["pooled-source"]["macro"])>1e-12:
            raise ValueError("以往的固定混合逻辑回归未复现")
        for a in ("observer","refit","adaptive"):
            prior[a]={"macro":old["arms"][a]["pooled"],"worst":min(v for v in old["arms"][a]["recall"] if v is not None)}
    summary={"protocol_sha256":protocol,"requests":requests,"feedback_digest":digest(returned_ids),"labels_used":len(history_y),
             "service":len(ids),"source_fits":1,"candidate_fits_all_arms":fits,"joint_cpu_seconds":cpu,
             "choices":[p["chosen"] for p in records],"different_vs_once":int(np.sum(all_p["ongoing"]!=all_p["once"])),
             "scores":table,"existing_comparators":prior}
    atomic_json(OUT/name/"scores.json",summary)
    print(name,{a:round(v["macro"],5) for a,v in table.items()},flush=True)
    return summary


if __name__=="__main__":
    tables={}
    for site,folder in (("full",TEMPERATURE),("replacement",TEMPERATURE),("co",DYNAMIC),("methane",DYNAMIC)):
        tables[site]=evaluate(site,external_input(folder/site))
    for old in sources():
        name,data=prior_input(old)
        label="official" if name=="public" else name
        tables[label]=evaluate(label,data)
    atomic_json(OUT/"scores.json",tables)
