"""以完全相同的留组校准比较暴露前干空气、稳态原读数及二者的差。"""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression

from verify_dsps_cycle_release import DEST as PRIOR, SOURCE, read, digest, LocalStore, LocalAcceptance, atomic_json, tally
from harness.models.dsps_cycle_release import kernel

DEST=PRIOR.parent/"dsps-air-control"
PRIMARY="按源留组效果选择参照与融合"


def candidates(x,src,tr,te,y,groups):
    probabilities={}
    for name,target in x.items():
        held=np.zeros((tr.sum(),3));train_ids=np.flatnonzero(tr)
        def fit(mask):
            return kernel(np.vstack((target[mask],src[name][mask])),np.tile(y[mask],2))
        for g in np.unique(groups[tr]):
            ev=groups[train_ids]==g
            held[ev]=fit(tr&(groups!=g)).decision_function(target[train_ids[ev]])
        service=fit(tr).decision_function(target[te])
        calibrate=LogisticRegression(C=1,max_iter=500).fit(held,y[tr])
        probabilities[name]=calibrate.predict_proba(service)
    return probabilities


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有暴露前对照")
        meta=read(SOURCE/"source.json")
        hashes={r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
        with np.load(SOURCE/"development.npz") as v:ids,group,y=v["ids"],v["repetition"],v["y"]
        src={};target={}
        for k in ids:
            path=SOURCE/"members"/k
            if sha256(path.read_bytes()).hexdigest()!=hashes[k]:raise ValueError("原件已改变")
            obj=loadmat(path,squeeze_me=True,struct_as_record=False)[path.stem]
            for row,bank in ((0,src),(11,target)):
                def extract(stage):
                    a=getattr(obj,stage)[:,128*row:128*(row+1)].reshape(-1,20,5,16,8)
                    return np.median(a[:,:,:,7,:],axis=2)
                base=extract("Bl")[0];stable=np.median(extract("St"),axis=0);early=np.median(extract("Tr")[:2],axis=0)
                for name,feat in (("暴露前干空气",base),("暴露稳态",stable),("稳态减干空气",stable-base),
                                  ("稳态相对干空气",np.arcsinh((stable-base)/np.maximum(abs(base),1e-4))),
                                  ("进入与稳态各温度差",early-stable),
                                  ("稳态并列基线差",np.concatenate((stable,stable-base),axis=0))):
                    bank.setdefault(name,[]).append(feat.ravel())
        src={k:np.stack(v) for k,v in src.items()};target={k:np.stack(v) for k,v in target.items()}
        old=read(PRIOR/"scores.json")
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        truth=[];cols={};events=[]
        for rep,parent in zip(np.unique(group),old["requests"],strict=True):
            tr=group!=rep;te=~tr; probs=candidates(target,src,tr,te,y,group)
            previous=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),previous])!=parent or previous["ids"]!=ids[te].tolist():raise ValueError("对照回执不匹配")
            pred={k:1+v.argmax(1) for k,v in probs.items()}
            if not np.array_equal(pred["暴露稳态"],previous["预测"]["整周期留组类别映射"]):raise ValueError("原50条正确未复现")
            # 当前读数能分辨暴露净响应时，原稳态及净变化可共同投票。干空气只作负对照。
            for name in ("稳态减干空气","稳态相对干空气","进入与稳态各温度差","稳态并列基线差"):
                pred["固定稳态与"+name]=1+(probs["暴露稳态"]+probs[name]).argmax(1)
            primary="稳态并列基线差"
            # 源端额外留一整组：源仅剩一个重复时，k折校准改为按浓度档交叉生成决策分数。
            # 源标签只用于选现场工具，不传给当前暴露。
            rates={k:[] for k in ("暴露稳态",primary,"稳态减干空气","进入与稳态各温度差")}
            dose=np.array([int(Path(k).stem[1:]) for k in ids])
            for g in np.unique(group[tr]):
                fit=tr&(group!=g);ev=tr&(group==g)
                inner=candidates({k:target[k] for k in rates},{k:src[k] for k in rates},fit,ev,y,dose)
                for k,a in inner.items(): rates[k].append(tally(y[ev],1+a.argmax(1))["F1"])
            best=max(rates,key=lambda k:np.mean(rates[k]))
            pred[PRIMARY]=1+(probs["暴露稳态"]+probs[best]).argmax(1)
            item={"ids":ids[te].tolist(),"预测":{k:v.tolist() for k,v in pred.items()},"安装选择":best}
            key=digest([int(rep),item]);events.append(key)
            if not receiver.submit(key,item)["accepted"]:raise ValueError("负对照交接失败")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),sent])!=key:raise ValueError("负对照回执错位")
            truth.extend(y[te])
            for k,v in sent["预测"].items():cols.setdefault(k,[]).extend(v)
        scores={k:tally(np.array(truth),np.array(v)) for k,v in cols.items()}
        fixed=max((k for k in cols if k not in (PRIMARY,"暴露前干空气")),key=lambda k:scores[k]["F1"])
        previous_best=old["scores"][old["decision"]["固定"]]
        paper=old["scores"][old["decision"]["论文"]]
        delta=scores[PRIMARY]["F1"]-max(scores[fixed]["F1"],previous_best["F1"],paper["F1"],scores["暴露前干空气"]["F1"])
        d={"主候选":PRIMARY,"固定":fixed,"最小领先":delta,"进独立":delta>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":events,"scores":scores,"decision":d})
        print(json.dumps({"控制":scores,"取舍":d},ensure_ascii=False))


if __name__=="__main__":main()
