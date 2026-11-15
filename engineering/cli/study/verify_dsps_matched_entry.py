"""从安装配对重建实际论文赢家，走逐条输入和命令行，对照同一重复已有的54份预测。"""
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys
import joblib
import numpy as np
from scipy.io import loadmat

from score_dsps_matched_paper import DEST as PRIOR, SOURCE, read, target_features, digest, LocalStore, LocalAcceptance, atomic_json, tally
from harness.models.dsps_cycle_release import CalibratedCycleRelease
from engineering.cli.workflow.dsps_cycle_predict import predict, ROOT

DEST=PRIOR.parent/"dsps-calibrated-entry"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("本权重已核验")
        meta=read(SOURCE/"source.json")
        hashes={r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
        with np.load(SOURCE/"development.npz") as z:ids,y,groups=z["ids"],z["y"],z["repetition"]
        receivers=LocalAcceptance();receivers.output=DEST/"accepted"
        source=[];target=[];records=[]
        for uid in ids:
            path=SOURCE/"members"/uid
            if sha256(path.read_bytes()).hexdigest()!=hashes[uid]:raise ValueError("原件身份不匹配")
            m=loadmat(path,squeeze_me=True,struct_as_record=False)[path.stem]
            for row,data in ((0,source),(11,target)):
                inputs={k:getattr(m,k)[:,128*row:128*(row+1)].reshape(-1,16,8).transpose(0,2,1) for k in ("Bl","Tr","St")}
                data.append(target_features(*inputs.values()))
                if row==11:records.append({"inputs":inputs})
        a,b=np.stack(source),np.stack(target)
        previous=read(PRIOR/"scores.json")
        events=[];all_y=[];all_pred=[]
        for rep,parent in zip(np.unique(groups),previous["requests"],strict=True):
            mask=groups==rep
            m=CalibratedCycleRelease().fit(a[~mask],b[~mask],y[~mask],groups[~mask])
            weight=DEST/f"fold-{rep}.joblib";joblib.dump(m,weight)
            sha=sha256(weight.read_bytes()).hexdigest(); model=joblib.load(weight)
            ix=np.flatnonzero(mask)
            pred=[predict(model,records[j],"direct-calibration")["outputs"]["class"] for j in ix]
            p=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),p])!=parent or p["ids"]!=ids[mask].tolist() or p["预测"]["DS"]!=pred:
                raise ValueError("直接校准实际入口未重现已有预测")
            inp={k:v.tolist() for k,v in records[ix[0]]["inputs"].items()}
            cmd=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/dsps_cycle_predict.py"),"--method","direct-calibration","--artifact",str(weight),"--sha256",sha]
            run=subprocess.run(cmd,input=json.dumps({"inputs":inp})+"\n",capture_output=True,text=True,cwd=ROOT,check=True)
            if json.loads(run.stdout)["outputs"]["class"]!=pred[0]:raise ValueError("外部命令输出不同")
            payload={"ids":ids[mask].tolist(),"prediction":pred,"权重":sha}
            key=digest([int(rep),payload]);events.append(key)
            if not receivers.submit(key,payload)["accepted"]:raise ValueError("交付失败")
            r=read(receivers.output/(key+".json"))["payload"]
            if digest([int(rep),r])!=key:raise ValueError("交付身份不同")
            all_y.extend(y[mask].tolist());all_pred.extend(r["prediction"])
        final=CalibratedCycleRelease().fit(a,b,y,groups)
        dest=DEST/"direct.joblib";joblib.dump(final,dest)
        result=tally(np.array(all_y),np.array(all_pred))
        atomic_json(DEST/"scores.json",{"requests":events,"score":result,"发布权重SHA256":sha256(dest.read_bytes()).hexdigest(),"装机暴露":len(y)})
        print(json.dumps({"论文固定工具开发交付":result,"命令行样本":len(events)},ensure_ascii=False))


if __name__=="__main__":main()
