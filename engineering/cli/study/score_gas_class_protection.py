"""重载冻结模型，以预测当时的类别支持和历史修错决定是否保留旧能力。

真值只用于已交付后的指定回标和最后计分；事后理想组合只出现在诊断结果。
"""
from hashlib import sha256
import json
import sys
from pathlib import Path

import numpy as np

from score_gas_probability_fusion import (ROOT,BASE,ARCHIVE,SEEDS,read,metrics,read_official_gas,
                                         model_scores,LocalStore,LocalAcceptance,atomic_json,digest)

SOURCE=BASE/"whole-maintenance/gas-domain-augmentation"
DEST=BASE/"whole-maintenance/gas-class-protection"
PROTOCOL=ROOT/"engineering/configs/experiments/gas-unseen-class-protection.json"
ADAPTED=tuple(f"augment-{weight}-{scale}-c1" for weight in ("domain","union") for scale in ("source","separate"))


def evaluate(seed, values, truth, samples):
    loc=SOURCE/f"seed-{seed}";old=read(loc/"scores.json")
    target=DEST/f"seed-{seed}";ep=LocalAcceptance();ep.output=target/"accepted"
    counts=np.zeros(6,dtype=int);gain={a:np.zeros(6,dtype=int) for a in ADAPTED}
    requests=[];received=set();events=[];oracles={a:[] for a in ADAPTED}
    for j,req in enumerate(old["requests"]):
        accepted=read(loc/"accepted"/(req+".json"));r=accepted["payload"]
        if accepted["request_id"]!=req or digest([digest(read(ROOT/"engineering/configs/experiments/gas-domain-augmentation.json")),seed,r])!=req:
            raise ValueError("原服务回执内容或协议身份不符")
        snapshot=read(loc/"snapshots"/(r["snapshot"]+".json"))
        if digest(snapshot)!=r["snapshot"] or r["round"]!=j+1 or r["feedback_seen"]!=len(received):
            raise ValueError("原快照与当时反馈数不一致")
        fb=loc/f"feedback-{j+1:02}.json"
        returned=read(fb) if fb.exists() else {"ids":[],"labels":[]}
        ids=r["ids"];rid=returned["ids"]
        eligible=[k for k,v in samples.items() if v["available_at"]==j and v["role"] in ("explore","adopt")]
        expected=sorted(eligible,key=lambda k:digest([seed,k]))[:32] if j<9 else []
        if rid!=expected or any(samples[k]["role"]!="serve" or samples[k]["available_at"]!=j or samples[k]["label"] is not None for k in ids):
            raise ValueError("服务角色、标签可见时点或回标清单变化")
        if set(rid)&(received|set(ids)):raise ValueError("反馈重复或服务交叉")
        both=np.asarray([values[k] for k in [*ids,*rid]])
        records=snapshot["models"]
        ordinary=f"pool-domain-{'source' if j<5 else 'separate'}-c1"
        backbone_prob=model_scores(records[ordinary],both,6)
        backbone=backbone_prob.argmax(1)+1
        if backbone[:len(ids)].tolist()!=r["predictions"]["fixed-midway"]:
            raise ValueError("重载后未复现固定中途配方")
        # 比较对象接收完全相同的旧、新概率与已公开历史，不向判断函数传服务标签。
        outputs={"backbone":backbone,"observer":np.r_[r["predictions"]["observer"],np.zeros(len(rid),dtype=int)]}
        feedback={}
        for a in ADAPTED:
            p=model_scores(records[a],both,6);new=p.argmax(1)+1
            if new[:len(ids)].tolist()!=r["predictions"][a]:raise ValueError("重载未复现增广模型")
            outputs[a]=new
            unseen=(counts[backbone-1]==0)|(counts[new-1]==0)
            outputs[a+"/unseen"]=np.where(unseen,backbone,new)
            outputs[a+"/history"]=np.where(gain[a][backbone-1]>0,new,backbone)
            outputs[a+"/geometric"]=(np.log(np.clip(p,1e-6,1))+np.log(np.clip(backbone_prob,1e-6,1))).argmax(1)+1
            for label in range(1,7):
                outputs[a+f"/fixed-{label}"]=np.where((backbone==label)|(new==label),backbone,new)
            feedback[a]=new[len(ids):]
        payload={"round":j+1,"ids":ids,"source_snapshot":r["snapshot"],"feedback_counts_before":counts.tolist(),
                 "historical_gains_before":{k:v.tolist() for k,v in gain.items()},
                 "predictions":{k:v[:len(ids)].tolist() for k,v in outputs.items()}}
        key=digest([digest(read(PROTOCOL)),seed,payload]);ack=ep.submit(key,payload)
        if not ack.get("accepted") or ack["request_id"]!=key:raise RuntimeError("新预测尚未确认")
        requests.append(key)
        if rid:
            labels=np.asarray(returned["labels"],dtype=int)
            if labels.tolist()!=[truth[k] for k in rid]:raise ValueError("原指定回标答案身份有误")
            group=backbone[len(ids):]-1
            for a,new in feedback.items():
                np.add.at(gain[a],group,(new==labels).astype(int)-(backbone[len(ids):]==labels).astype(int))
            counts+=np.bincount(labels-1,minlength=6)
        received.update(rid)
        # 预测均已保存后，诊断器才看服务答案；这些选法绝不传回运行时。
        y=np.asarray([truth[k] for k in ids])
        for a in ADAPTED:
            new=outputs[a][:len(ids)]
            oracles[a].extend(np.where((new==y)|(backbone[:len(ids)]==y),y,backbone[:len(ids)]).tolist())
        events.append({"round":j+1,"returned_after":rid,"counts_after":counts.tolist()})
    logs=[read(ep.output/(r+".json"))["payload"] for r in requests]
    ids=[k for r in logs for k in r["ids"]]
    if len(ids)!=414 or len(set(ids))!=414 or len(received)!=288:raise ValueError("服务或反馈范围变化")
    y=np.asarray([truth[k] for k in ids]);pred={a:np.concatenate([r["predictions"][a] for r in logs]) for a in logs[0]["predictions"]}
    table={a:metrics(y,p) for a,p in pred.items()}
    for a,b in (("backbone","fixed-midway"),("observer","observer")):
        if abs(table[a]["macro"]-old["scores"][b]["macro"])>1e-12:raise ValueError("原基线分数未复现")
    result={"seed":seed,"requests":requests,"scores":table,"events":events,
            "row_oracle_diagnosis":{a:metrics(y,np.asarray(p)) for a,p in oracles.items()}}
    atomic_json(target/"scores.json",result)
    adaptive=[k for k in table if k.endswith(("/unseen","/history"))]
    static=[k for k in table if k not in adaptive]
    print(json.dumps({"seed":seed,"best_adaptive":max(adaptive,key=lambda a:table[a]["macro"]),
                      "adaptive_macro":max(table[a]["macro"] for a in adaptive),
                      "best_static_macro":max(table[a]["macro"] for a in static)}),flush=True)
    return result


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有完整对照，请复用")
        pkg=read_official_gas(ARCHIVE);rows=(*pkg.train,*pkg.validation)
        values={r.sample_id:r.values for r in rows};truth={r.sample_id:r.label for r in rows}
        samples=read(BASE/"selected/gas/state.json")["samples"]
        runs=[evaluate(seed,values,truth,samples) for seed in SEEDS]
        mean={a:{"macro":float(np.mean([r["scores"][a]["macro"] for r in runs])),
                 "worst":float(np.mean([r["scores"][a]["worst"] for r in runs]))} for a in runs[0]["scores"]}
        adaptive=[a for a in mean if a.endswith(("/unseen","/history"))]
        static=[a for a in mean if a not in adaptive]
        best_adaptive=max(adaptive,key=lambda a:mean[a]["macro"]);best_static=max(static,key=lambda a:mean[a]["macro"])
        highest=max(mean,key=lambda a:mean[a]["macro"]);default=runs[0]["scores"]
        eligible=[a for a in mean if mean[a]["worst"]>=mean["observer"]["worst"]-.02
                  and default[a]["worst"]>=default["observer"]["worst"]-.02]
        selected=max(eligible,key=lambda a:mean[a]["macro"])
        advances=(mean[selected]["macro"]>mean["backbone"]["macro"] and mean[selected]["worst"]>=mean["observer"]["worst"]-.02
                  and default[selected]["macro"]>default["observer"]["macro"] and default[selected]["worst"]>=default["observer"]["worst"]-.02)
        result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"means":mean,"selected":selected,
                "best_adaptive":best_adaptive,"best_static":best_static,"advances":bool(advances),"new_training":False,
                "best_unconstrained":highest,"eligible":eligible,"development_only":True,"seeds":list(SEEDS)}
        atomic_json(DEST/"scores.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
