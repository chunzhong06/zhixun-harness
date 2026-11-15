"""从原始实测器件走完整入口，比较固定的整周期类别映射与源浓度指导的现场模型。"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from score_dsps_cycle_labels import DEST as PRIOR, SOURCE, read, LocalAcceptance, LocalStore, atomic_json, digest, tally
from harness.models.dsps_cycle_release import CycleRelease, FIXED, PRIMARY, target_features
from engineering.cli.workflow.dsps_cycle_predict import predict

DEST=PRIOR.parent/"dsps-cycle-entry"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("原始入口已复核")
        meta=read(SOURCE/"source.json")
        hashes={r["成员"]:r["SHA256"] for r in meta["已核验成员"]}
        with np.load(SOURCE/"development.npz") as z:
            ids,y,groups=z["ids"],z["y"],z["repetition"]
        dose=np.array([int(Path(k).stem[1:]) for k in ids])
        previous=read(PRIOR/"scores.json")
        source=[];target=[];records=[]
        for k in ids:
            p=SOURCE/"members"/k
            if sha256(p.read_bytes()).hexdigest()!=hashes[k]:raise ValueError("DS-PS原始成员身份不同")
            raw=loadmat(p,squeeze_me=True,struct_as_record=False)[p.stem]
            for dev,features in ((0,source),(11,target)):
                inp={s:getattr(raw,s)[:,128*dev:128*(dev+1)].reshape(-1,16,8).transpose(0,2,1) for s in ("Bl","Tr","St")}
                features.append(target_features(**{s.lower():a for s,a in inp.items()}))
                if dev==11:records.append({"inputs":inp})
        x,xt=np.stack(source),np.stack(target)
        receiver=LocalAcceptance();receiver.output=DEST/"accepted"
        events=[];truth=[];cols={};dose_correct=0
        for rep,parent in zip(np.unique(groups),previous["requests"],strict=True):
            tr=groups!=rep;te=~tr
            model=CycleRelease().fit(x[tr],xt[tr],y[tr],groups[tr],dose[tr])
            weight=DEST/f"fold-{rep}.joblib";joblib.dump(model,weight)
            move=joblib.load(weight); expected,level=move.predict(xt[te])
            old=read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(rep),old])!=parent or old["ids"]!=ids[te].tolist():raise ValueError("开发重复身份不符")
            fixed_id=previous["decision"]["固定"]
            if not np.array_equal(expected[FIXED],old["预测"][fixed_id]):raise ValueError("源端留组映射未复现")
            indices=np.flatnonzero(te)
            predictions={k:v.tolist() for k,v in expected.items()}
            for arm in (FIXED,PRIMARY):
                sent=[predict(move,records[j],arm)["outputs"]["class"] for j in indices]
                if sent!=predictions[arm]:raise ValueError("单次输入推理和独立批量预测不同")
                predictions[arm]=sent
            # 再启动命令行进程；文件名、源标签及浓度都没有载入该进程的请求。
            example=records[indices[0]]["inputs"]
            cmd=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/dsps_cycle_predict.py"),"--artifact",str(weight),"--sha256",sha256(weight.read_bytes()).hexdigest()]
            raw=json.dumps({"inputs":{k:v.tolist() for k,v in example.items()}},allow_nan=False)
            actual=subprocess.run(cmd,input=raw+"\n",capture_output=True,text=True,check=True,cwd=ROOT)
            if json.loads(actual.stdout)["outputs"]["class"]!=predictions[FIXED][0]:raise ValueError("JSONL部署结果不同")
            payload={"ids":ids[te].tolist(),"预测":{**old["预测"],**predictions},"估计浓度":level.tolist(),"模型SHA256":sha256(weight.read_bytes()).hexdigest()}
            key=digest([int(rep),payload]);events.append(key)
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("接收端未接受")
            sent=read(receiver.output/(key+".json"))["payload"]
            if digest([int(rep),sent])!=key:raise ValueError("接收端预测已改变")
            truth.extend(y[te]);dose_correct+=int(np.sum(level==dose[te]))
            for k,v in sent["预测"].items():cols.setdefault(k,[]).extend(v)
        scores={k:tally(np.array(truth),np.array(v)) for k,v in cols.items()}
        fixed=max([previous["decision"]["固定"],*[k for k in expected if k not in (PRIMARY,"仅按估计浓度映射")]],key=lambda k:scores[k]["F1"])
        paper=previous["decision"]["论文"]
        gap=scores[PRIMARY]["F1"]-max(scores[fixed]["F1"],scores[paper]["F1"])
        d={"固定":fixed,"主候选":PRIMARY,"论文":paper,"差值":gap,"进独立":gap>=.01 and scores[PRIMARY]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        final=CycleRelease().fit(x,xt,y,groups,dose)
        f=DEST/"release.joblib";joblib.dump(final,f)
        atomic_json(DEST/"scores.json",{"scores":scores,"decision":d,"requests":events,"估中浓度":dose_correct,
                    "装机暴露":len(y),"发布权重SHA256":sha256(f.read_bytes()).hexdigest()})
        print(json.dumps({"实际交付":{k:scores[k] for k in (FIXED,PRIMARY,fixed,paper)},"取舍":d,"估中浓度":dose_correct},ensure_ascii=False))


if __name__=="__main__":main()
