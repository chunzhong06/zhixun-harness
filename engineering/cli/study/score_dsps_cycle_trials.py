"""仅使用既有三个开发重复，对照同一暴露中的五次稳态循环是否能帮助选模型。"""
import json
import joblib
import numpy as np
from scipy.io import loadmat
from hashlib import sha256
from score_dsps_install_trials import DEST as PRIOR, LocalStore, LocalAcceptance, read, digest, atomic_json, tally
from score_dsps_dose_contrast import SOURCE
from harness.models.dsps_cycle_trials import CycleTrials,PRIMARY,cycles
from harness.models.dsps_dose_contrast import EQUAL,PAPER

DEST=PRIOR.parent/"dsps-cycle-trials"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同暴露的实验已登记")
        hashes={r["成员"]:r["SHA256"] for r in read(SOURCE/"source.json")["已核验成员"]}
        with np.load(SOURCE/"development.npz") as z:
            ids,y,g=z["ids"],z["y"],z["repetition"]
        a=[];b=[];level=[]
        for k in ids:
            p=SOURCE/"members"/k
            if sha256(p.read_bytes()).hexdigest()!=hashes[k]:raise ValueError("原始暴露身份不符")
            mat=loadmat(p,squeeze_me=True,struct_as_record=False)[p.stem]
            for row,bank in ((0,a),(11,b)):
                bank.append(cycles(mat.St[:,128*row:128*(row+1)].reshape(500,16,8).transpose(0,2,1)))
            level.append(int(p.stem[1:]))
        a,b,level=map(np.asarray,(a,b,level)); previous=read(PRIOR/"scores.json")
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        truth=[];cols={};events=[]
        for group,parent in zip(np.unique(g),previous["requests"],strict=True):
            test=g==group
            m=CycleTrials().fit(a[~test],b[~test],y[~test],g[~test],level[~test])
            out,quality=m.predict(b[test]);prior=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(group),prior])!=parent or prior["ids"]!=ids[test].tolist() or any(out[k].tolist()!=prior["预测"][k] for k in (PAPER,EQUAL)):
                raise ValueError("原始循环汇总与先前预测不符")
            payload={"ids":ids[test].tolist(),"预测":{**prior["预测"],**{k:v.tolist() for k,v in out.items()}},"循环留出偏差":quality.tolist()}
            key=digest([int(group),payload])
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("接收失败")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(group),sent])!=key:raise ValueError("已发内容不同")
            events.append(key);truth.extend(y[test])
            for k,v in sent["预测"].items():cols.setdefault(k,[]).extend(v)
        scores={k:tally(np.asarray(truth),np.array(v)) for k,v in cols.items()}
        controls=[k for k in out if k not in (PRIMARY,"只按循环复测试选")]
        fixed=max([previous["decision"]["固定"],*controls],key=lambda k:scores[k]["F1"])
        paper=max([PAPER,"直接标准化用循环平均","直接标准化展开循环"],key=lambda k:scores[k]["F1"])
        gain=scores[PRIMARY]["F1"]-max(scores[fixed]["F1"],scores[paper]["F1"])
        d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":scores[PRIMARY]["F1"],"固定F1":scores[fixed]["F1"],
           "论文F1":scores[paper]["F1"],"差值":gain,"进独立":gain>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        f=DEST/"release.joblib";joblib.dump(CycleTrials().fit(a,b,y,g,level),f)
        atomic_json(DEST/"scores.json",{"decision":d,"scores":scores,"requests":events,"权重SHA256":sha256(f.read_bytes()).hexdigest()})
        print(json.dumps({"开发交付":d,"源和目标实际暴露":len(y)},ensure_ascii=False))


if __name__=="__main__":main()
