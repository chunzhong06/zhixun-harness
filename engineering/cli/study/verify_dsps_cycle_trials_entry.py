"""从每次真实暴露原始三阶段提交分类结果，同时重发固定和现场五循环版本。"""
import json
from hashlib import sha256
import subprocess
import sys
import numpy as np
import joblib
from scipy.io import loadmat
from score_dsps_cycle_trials import DEST as PRIOR,SOURCE,LocalStore,LocalAcceptance,read,digest,atomic_json,tally,CycleTrials,cycles
from harness.models.dsps_cycle_trials import EXPANDED,PRIMARY
from engineering.cli.workflow.dsps_cycle_predict import ROOT,predict

DEST=PRIOR.parent/"dsps-cycle-trials-entry"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("五周期入口已核验")
        old=read(PRIOR/"scores.json"); final=PRIOR/"release.joblib"
        if sha256(final.read_bytes()).hexdigest()!=old["权重SHA256"]:raise ValueError("装机权重身份变化")
        hashes={r["成员"]:r["SHA256"] for r in read(SOURCE/"source.json")["已核验成员"]}
        with np.load(SOURCE/"development.npz") as z:ids,y,g=z["ids"],z["y"],z["repetition"]
        a=[];b=[];level=[];records=[]
        for k in ids:
            p=SOURCE/"members"/k
            if sha256(p.read_bytes()).hexdigest()!=hashes[k]:raise ValueError("实测原件身份变化")
            mat=loadmat(p,squeeze_me=True,struct_as_record=False)[p.stem]
            for row,bank in ((0,a),(11,b)):
                inp={n:getattr(mat,n)[:,128*row:128*(row+1)].reshape(-1,16,8).transpose(0,2,1) for n in ("Bl","Tr","St")}
                bank.append(cycles(inp["St"]))
                if row==11:records.append({"inputs":inp})
            level.append(int(p.stem[1:]))
        a,b,level=map(np.asarray,(a,b,level))
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        truth=[];issued={k:[] for k in (EXPANDED,PRIMARY)};events=[]
        for group,parent in zip(np.unique(g),old["requests"],strict=True):
            test=g==group
            model=CycleTrials().fit(a[~test],b[~test],y[~test],g[~test],level[~test])
            f=DEST/f"fold-{int(group)}.joblib";joblib.dump(model,f);sha=sha256(f.read_bytes()).hexdigest()
            frozen=joblib.load(f);idx=np.flatnonzero(test)
            prior=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(group),prior])!=parent or prior["ids"]!=ids[test].tolist():raise ValueError("外层重复身份不同")
            pred={}
            for k,mode in ((EXPANDED,"cycle-average"),(PRIMARY,"cycle-trials")):
                values=[predict(frozen,records[i],k)["outputs"]["class"] for i in idx]
                if values!=prior["预测"][k]:raise ValueError("实际发放与研究回执不同")
                raw={"inputs":{s:v.tolist() for s,v in records[idx[0]]["inputs"].items()}}
                cmd=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/dsps_cycle_predict.py"),
                     "--method",mode,"--artifact",str(f),"--sha256",sha]
                line=subprocess.run(cmd,input=json.dumps(raw)+"\n",text=True,capture_output=True,check=True,cwd=ROOT)
                if json.loads(line.stdout)["outputs"]["class"]!=values[0]:raise ValueError("外部命令结果不同")
                pred[k]=values
            payload={"ids":ids[test].tolist(),"预测":pred,"权重SHA256":sha,"原回执":parent}
            key=digest([int(group),payload])
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("交接未成功")
            got=read(receiver.output/(key+".json"))["payload"]
            if digest([int(group),got])!=key:raise ValueError("交接身份变化")
            events.append(key);truth.extend(y[test])
            for k in issued:issued[k].extend(got["预测"][k])
        score={k:tally(np.asarray(truth),np.asarray(v)) for k,v in issued.items()}
        atomic_json(DEST/"scores.json",{"实际评分":score,"requests":events,"物理暴露数":len(truth),"命令行样本":6,"装机权重SHA256":old["权重SHA256"]})
        print(json.dumps({"原始交付":score,"物理暴露数":len(truth)},ensure_ascii=False))


if __name__=="__main__":main()
