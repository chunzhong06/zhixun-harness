"""源/现场共享与专属边界的成组对照；保留实际交付和折叠后的单行模型。"""
from hashlib import sha256
import json
from pathlib import Path
from time import process_time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from score_gas_probability_fusion import (ROOT,BASE,ARCHIVE,SEEDS,stream,metrics,read,
                                        read_official_gas,GasFeedbackBank,SiteTraining,RECIPES,
                                        model_scores,LocalAcceptance,LocalStore,atomic_json,digest)
from harness.models.field_classification import weights,fixed_snapshot

DEST=BASE/"whole-maintenance/gas-domain-augmentation"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-domain-augmentation.json"


def fitted(source,sy,history,answers,base):
    """同一训练输入上改变表示和样本权重，校验折叠模型与直接增广预测相同。"""
    if not len(answers):
        initial=base.models()["pooled-source"]
        return {f"{form}-{balance}-{scale}-c{c}":initial
                for scale in ("source","separate") for balance in ("domain","union")
                for form,c in (("pool",1),("pool",2),("augment",1))},0.
    x=np.asfortranarray(history);ty=np.asarray(answers,dtype=int)
    feature=source.shape[1]
    src=base.source_z
    domain_w=np.r_[weights(sy,len(sy)),weights(ty,len(sy))]
    labels=np.r_[sy,ty]
    byclass=np.bincount(labels-1,weights=domain_w,minlength=6)
    union_w=domain_w*domain_w.sum()/(6*byclass[labels-1])
    if not np.allclose(np.bincount(labels-1,weights=union_w,minlength=6),domain_w.sum()/6):
        raise ValueError("按类别平衡后总权重不一致")
    result={};start=process_time()
    for suffix,scaler in (("source",base.source_scale),("separate",StandardScaler().fit(x))):
        target=scaler.transform(x)
        pooled=np.vstack((src,target))
        aug=np.vstack((np.column_stack((src,src,np.zeros_like(src))),
                       np.column_stack((target,np.zeros_like(target),target))))
        for balance,w in (("domain",domain_w),("union",union_w)):
            for form,c in (("pool",1),("pool",2),("augment",1)):
                # 原C1/按域平衡模型直接复用，避免为归因重复拟合。
                key=f"{form}-{balance}-{suffix}-c{c}"
                if form=="pool" and c==1 and balance=="domain":
                    result[key]=base.models()["pooled-"+suffix]
                    continue
                data=pooled if form=="pool" else aug
                reg=LogisticRegression(C=float(c),random_state=17,max_iter=1000).fit(data,labels,sample_weight=w)
                record=fixed_snapshot("mixed",(scaler,reg))
                if form=="augment":
                    full=reg.coef_.copy()
                    record["coef"]=(full[:,:feature]+full[:,2*feature:]).tolist()
                    # 使用训练端少量实际读数核对部署折叠，不以其答案检查性能。
                    test=target[:min(8,len(target))]
                    direct=reg.predict_proba(np.column_stack((test,np.zeros_like(test),test)))
                    raw=x[:len(test)]
                    np.testing.assert_allclose(model_scores(record,raw,6),direct,rtol=0,atol=1e-10)
                result[key]=record
    return result,process_time()-start


def evaluate(seed,pkg,batches):
    source=np.asarray([r.values for r in pkg.train]);sy=np.asarray([r.label for r in pkg.train])
    base=SiteTraining(source,sy,GasFeedbackBank.names,6,"pooled-source")
    history=[];answers=[];received=set();requests=[];cpu=0.
    folder=DEST/f"seed-{seed}";ep=LocalAcceptance();ep.output=folder/"accepted"
    for j,(ids,x,votes,pool,fx,fv,pool_y,weight) in enumerate(batches):
        ix=sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:32] if j+1<len(batches) else []
        selected=[pool[i] for i in ix]
        if set(selected)&(received|set(ids)):raise ValueError("回标重复或服务交叉")
        both=np.vstack((x,fx[ix]))
        records,cost=fitted(source,sy,history,answers,base);cpu+=cost
        original=base.predict(both,{})
        pred={k:model_scores(m,both,6).argmax(1)+1 for k,m in records.items()}
        all_votes={k:np.r_[votes[k],fv[k][ix]] for k in votes}
        pred.update(original)
        pred["observer"]=base.observer.predict(all_votes,"uniform")
        pred["fixed-midway"]=original["pooled-source" if j<5 else "pooled-separate"]
        snapshot={"models":records,"source_weight":weight,"feedback_seen":len(received)}
        snap=digest(snapshot);atomic_json(folder/"snapshots"/(snap+".json"),snapshot)
        payload={"round":j+1,"ids":ids,"snapshot":snap,"feedback_seen":len(received),
                 "predictions":{k:v[:len(ids)].tolist() for k,v in pred.items()}}
        req=digest([digest(read(PROTOCOL)),seed,payload]);ack=ep.submit(req,payload)
        if not ack.get("accepted") or ack["request_id"]!=req:raise RuntimeError("输出尚未确认")
        requests.append(req)
        if ix:
            revealed=[pool_y[i] for i in ix]
            base.update(fx[ix],{k:fv[k][ix] for k in fv},revealed,{k:v[len(ids):] for k,v in original.items()})
            history.extend(fx[ix].tolist());answers.extend(revealed)
            atomic_json(folder/f"feedback-{j+1:02}.json",{"ids":selected,"labels":revealed})
        received.update(selected)
    logs=[read(ep.output/(r+".json"))["payload"] for r in requests]
    ids=[k for r in logs for k in r["ids"]];truth={r.sample_id:r.label for r in pkg.validation}
    if len(ids)!=414 or len(set(ids))!=414 or len(received)!=288:raise ValueError("服务和回标与既有公开协议不同")
    y=np.asarray([truth[k] for k in ids]);p={a:np.concatenate([r["predictions"][a] for r in logs]) for a in logs[0]["predictions"]}
    table={a:metrics(y,v) for a,v in p.items()}
    ref=read(BASE/"whole-maintenance/site-training/draws/official/scores.json")
    old=next(r for r in ref["draws"] if r["seed"]==seed)["scores"]
    for a in (*RECIPES,"fixed-midway"):
        if abs(table[a]["macro"]-old[a]["macro"])>1e-12:raise ValueError(f"固定参照未复现：{a}")
    result={"seed":seed,"requests":requests,"scores":table,"new_fit_cpu_seconds":cpu,"existing_bank_cpu_seconds":base.cpu}
    atomic_json(folder/"scores.json",result)
    print(json.dumps({"seed":seed,"macro":{a:round(v["macro"],5) for a,v in table.items()}}),flush=True)
    return result


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有完整对照，请复用")
        pkg=read_official_gas(ARCHIVE);batches=stream(pkg,GasFeedbackBank(pkg.train))
        runs=[evaluate(seed,pkg,batches) for seed in SEEDS]
        mean={a:{"macro":float(np.mean([r["scores"][a]["macro"] for r in runs])),
                 "worst":float(np.mean([r["scores"][a]["worst"] for r in runs]))} for a in runs[0]["scores"]}
        arms=[a for a in mean if a.startswith("augment-")]
        controls=[a for a in mean if a not in arms and a not in ("local-source","local-separate")]
        best_augmentation=max(arms,key=lambda a:mean[a]["macro"])
        best=max(controls,key=lambda a:mean[a]["macro"])
        new=[a for a in mean if a.startswith(("augment-","pool-"))]
        selected=max(new,key=lambda a:mean[a]["macro"])
        prior=max((*RECIPES,"fixed-midway"),key=lambda a:mean[a]["macro"])
        standard=runs[0]["scores"]
        passes=(mean[selected]["macro"]>mean[prior]["macro"] and mean[selected]["worst"]>=mean["observer"]["worst"]-.02
                and standard[selected]["macro"]>standard["observer"]["macro"] and standard[selected]["worst"]>=standard["observer"]["worst"]-.02)
        result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"means":mean,"selected":selected,"best_augmentation":best_augmentation,
                "best_control":best,"previous_best":prior,
                "advances":bool(passes),"development_only":True,"seeds":list(SEEDS)}
        atomic_json(DEST/"scores.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
