"""每台重新装机；在新进程按原测点顺序交接频谱与时间块，逐条核对开发胜出工具。"""
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from score_bearing_prefix import ARCHIVE_SIZE, DATA, ROOT, LocalAcceptance, LocalStore, atomic_json, describe, devices, digest, fingerprint, frozen, read
from score_bearing_wear_rate import DEST as PRIOR, SPEC as PRIOR_SPEC
from harness.models.bearing_prefix_release import BearingPrefixRelease

DEST=PRIOR.parent/"bearing-prefix-entry"
SPEC=ROOT/"engineering/configs/experiments/bearing-prefix-entry.json"
TOOL="与前期固定各半/固定速度窗80上限1"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同一入口已复核")
        units=devices()
        report=read(PRIOR/"scores.json")["calendar"]
        prior_hash=fingerprint(PRIOR_SPEC)
        receiver=LocalAcceptance()
        receiver.output=DEST/"accepted"
        template=frozen("bearing")[0]["catboost"]
        truth,issued,events=[],[],[]
        maximum=0.
        for u,parent in zip(units[ARCHIVE_SIZE:],report["requests"],strict=True):
            past=[r for r in units if r.end<u.start]
            model=BearingPrefixRelease().fit(units[:ARCHIVE_SIZE],past,template)
            weight=DEST/(u.name+".joblib")
            joblib.dump(model,weight)
            sha=fingerprint(weight)
            old=read(PRIOR/"calendar"/"accepted"/(parent+".json"))["payload"]
            if digest([prior_hash,"calendar",old])!=parent or old["ids"]!=u.ids:raise ValueError("前次回执错位")
            times=pd.to_datetime(pd.read_csv(DATA/"op"/(u.name+".csv")).iloc[:,0],format="%d-%b-%Y %H:%M:%S")
            seconds=times.to_numpy(dtype="datetime64[s]").astype(np.int64)
            raw=np.load(DATA/"fft"/(u.name+".npy"),mmap_mode="r")
            lines=[]
            last=-1
            for uid in u.ids:
                ix=int(uid.split("/")[1])
                lines.append(json.dumps({"inputs":{"spectrum":raw[ix].tolist(),"timestamps":seconds[last+1:ix+1].tolist()}}))
                last=ix
            cmd=[sys.executable,"-B",str(ROOT/"engineering/cli/workflow/bearing_prefix_predict.py"),
                 "--artifact",str(weight),"--sha256",sha]
            res=subprocess.run(cmd,input="\n".join(lines)+"\n",capture_output=True,text=True,check=True,cwd=ROOT)
            # 必须每行返回，且误差只容许浮点求和顺序差异。
            pred=np.asarray([json.loads(line)["outputs"]["remaining_calendar_minutes"] for line in res.stdout.splitlines()])
            if pred.shape!=(len(u.ids),):raise ValueError("命令行漏发真实测点")
            err=float(np.max(np.abs(pred-np.asarray(old["预测"][TOOL]))))
            if err>1e-6:raise ValueError(f"{u.name}逐条复现差异{err}")
            maximum=max(maximum,err)
            payload={"ids":u.ids,"预测":pred.tolist(),"权重":sha,"原回执":parent}
            key=digest([fingerprint(SPEC),payload])
            if not receiver.submit(key,payload)["accepted"]:raise ValueError("单条入口发放失败")
            actual=read(receiver.output/(key+".json"))["payload"]
            if digest([fingerprint(SPEC),actual])!=key:raise ValueError("接收端预测更改")
            truth.append(u.durations["calendar"]-u.ages["calendar"])
            issued.append(np.asarray(actual["预测"]));events.append(key)
        final=DEST/"release.joblib"
        joblib.dump(BearingPrefixRelease().fit(units[:ARCHIVE_SIZE],units,template),final)
        score=describe({TOOL:issued},truth)[TOOL]
        result={"固定入口":score,"复现最大差值":maximum,"条数":sum(map(len,issued)),"requests":events,"发布权重SHA256":fingerprint(final)}
        atomic_json(DEST/"scores.json",result)
        print(json.dumps({"日历误差":score["RMSE"],"最差整台":score["最差整台"],"复现最大差值":maximum,"条数":result["条数"]},ensure_ascii=False))


if __name__=="__main__":main()
