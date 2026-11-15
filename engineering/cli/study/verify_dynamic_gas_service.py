"""真实进程只接收当前十分钟原始采样，取得接收回执后再发送本批合法标签。"""
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score
from score_dynamic_gas_memory import RAW,ROOT,DATA,EXPECTED,DEST as PRIOR,PRIOR as BASE,seconds,PRIMARY,LocalStore,LocalAcceptance,atomic_json,digest

DEST = PRIOR.parent/"dynamic-memory-entry"
CLI = ROOT/"engineering/cli/workflow/dynamic_gas_service.py"
FILES = {"co":"ethylene_CO.txt","methane":"ethylene_methane.txt"}


def batches(stream):
    pending=[];current=None
    for chunk in pd.read_csv(stream,sep=r"\s+",skiprows=1,header=None,chunksize=100000,usecols=[0,*range(3,19)]):
        rows=chunk.to_numpy(dtype=float)
        batch=np.floor(rows[:,0]/600).astype(int)
        bounds=np.r_[0,np.flatnonzero(np.diff(batch))+1,len(batch)]
        for a,b in zip(bounds[:-1],bounds[1:],strict=True):
            if current is not None and batch[a]!=current:
                yield np.vstack(pending)
                pending=[]
            pending.append(rows[a:b]);current=batch[a]
    if pending:yield np.vstack(pending)


def communicate(proc,request):
    proc.stdin.write(json.dumps(request)+"\n");proc.stdin.flush()
    line=proc.stdout.readline()
    if not line:raise ValueError("持续服务进程未返回已发预测")
    return json.loads(line)


def issue(site,source,z,folder,d,score_root=PRIOR,destination=DEST,fixed_combo=False,lag=False):
    src=json.loads((DATA/source/"source.json").read_bytes())
    with z.open(FILES[source]) as stream:t,x,_=seconds(stream)
    np.savez(folder/"source.npz",seconds=x,windows=t//10,labels=np.array([r["label"] for r in src]))
    old=json.loads((BASE/site/"ongoing/run.json").read_bytes())
    truth=json.loads((DATA/site/"evaluation.json").read_bytes())
    proc=subprocess.Popen([sys.executable,"-B",str(CLI),"--source",str(folder/"source.npz"),"--site",site,*(["--calibrate-lag"] if lag else [])],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,encoding="utf-8",text=True)
    receiver=LocalAcceptance();receiver.output=destination/site/"accepted"
    receipts=[];spent=0
    names={"现场":PRIMARY,"固定":d["decision"]["固定"],"论文":d["decision"]["论文"]}
    if fixed_combo and site=="co" and not lag:
        names.update({"固定":"差分与论文固定各半","论文":"差分与论文固定各半"})
    if lag:
        from harness.models.delayed_teacher import PRIMARY as LAG
        names["现场"]=LAG
        if names["论文"]=="状态与论文各半/原差分":names["论文"]="差分与论文固定各半"
    try:
        with z.open(FILES[site]) as full:
            for raw,prior,e in zip(batches(full),d["requests"],old["accepted"],strict=True):
                prior_p=json.loads((score_root/site/"accepted"/(prior+".json")).read_bytes())["payload"]
                if digest([site,prior_p])!=prior:raise ValueError("前次回执变化")
                ancestor=prior_p
                if lag:
                    parent=score_root.parent/"gas-transitions"/site/"accepted"/(prior_p["上轮"]+".json")
                    ancestor=json.loads(parent.read_bytes())["payload"]
                    if digest([site,ancestor])!=prior_p["上轮"]:raise ValueError("切换试验前序变化")
                if fixed_combo:
                    identity=ancestor["上轮"]
                    ancestor=json.loads((PRIOR/site/"accepted"/(identity+".json")).read_bytes())["payload"]
                    if digest([site,ancestor])!=identity:raise ValueError("先前动态预测变化")
                if ancestor["上轮"]!=e["request"]:raise ValueError("反馈前序不符")
                batch_path=folder/"batch.npz"
                np.savez(batch_path,clock=raw[:,0],sensors=raw[:,1:])
                value=communicate(proc,{"batch":str(batch_path)})
                ids=value["ids"]
                if ids!=prior_p["ids"] or value["used"]!=spent:raise ValueError("重放窗口或回标时序变化")
                for arm,vals in value["predictions"].items():
                    if arm=="差分与论文固定各半" and not fixed_combo:continue
                    ref_arm="状态与论文各半/原差分" if arm=="差分与论文固定各半" else arm
                    if vals!=prior_p["预测"][ref_arm]:raise ValueError("因果部署入口的分类与研究输出不一致")
                payload={"ids":ids,"预测":{t:value["predictions"][arm] for t,arm in names.items()},"回标已用":spent,"上轮":prior}
                h=digest([site,payload])
                if not receiver.submit(h,payload)["accepted"]:raise ValueError("重发未确认")
                receipts.append(h)
                labels={uid:truth[uid] for uid in e["returned"]}
                ack=communicate(proc,{"feedback":labels})
                spent+=len(labels)
                if ack["used"]!=spent:raise ValueError("真实回标没有正确进入下一批")
        proc.stdin.close()
        if proc.wait(timeout=20):raise ValueError("连续入口未正常收束")
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        if not proc.stdin.closed:proc.stdin.close()
        proc.stdout.close()
    ys=[];cols={name:[] for name in names}
    for h in receipts:
        p=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
        if digest([site,p])!=h:raise ValueError("重放回执改变")
        ys.extend(truth[uid] for uid in p["ids"])
        for n,v in p["预测"].items():cols[n].extend(v)
    scores={n:{"F1":float(f1_score(ys,v,average="macro")),"最弱召回":float(recall_score(ys,v,average=None).min())} for n,v in cols.items()}
    return {"scores":scores,"requests":receipts,"回标":spent,"实际预测":len(ys)}


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("连续服务已经验收")
        with RAW.open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("原始连续测量改变")
        meta=json.loads((PRIOR/"scores.json").read_bytes())
        with ZipFile(RAW) as archive:
            results={s:issue(s,t,archive,Path(temp),meta[s]) for s,t in (("co","methane"),("methane","co"))}
        atomic_json(DEST/"scores.json",results)
        print(json.dumps({s+"原始入口":{m+"F1":v["F1"] for m,v in r["scores"].items()} for s,r in results.items()},ensure_ascii=False))
