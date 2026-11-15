"""各安装从原始读数重发四种策略；核对装机数量已知时的强固定基准。"""
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score, recall_score

from score_year_gas_calibration import DEST as BASE, DATA, NAME, HASH, ROOT, LocalStore, LocalAcceptance, atomic_json, digest
from score_year_sensor_pairs import DEST as PAIRED, PRIMARY

DEST = PAIRED.parent/"year-sensor-entry"
CLI = ROOT/"engineering/cli/workflow/year_gas_predict.py"


def run(folder):
    first = json.loads((BASE/"scores.json").read_bytes())
    second = json.loads((PAIRED/"scores.json").read_bytes())
    with (DATA/NAME).open("rb") as f:
        if file_digest(f,"md5").hexdigest()!=HASH:raise ValueError("作者原件变化")
    d = loadmat(DATA/NAME,variable_names=["sensor","target"],simplify_cells=True)
    target = d["target"]
    states = target["states"]
    valid = np.isfinite(states)&np.isfinite(target["acetone"])&np.isfinite(target["toluene"])
    ids = sorted(set(states[valid].astype(int)),key=lambda i:sha256(f"year-gas-v1/{i}".encode()).digest())
    service = ids[120:]
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    seen, records = set(), {}
    for key,req in second["requests"].items():
        chip,gas,size = key.split("-")
        old_key = first["requests"][key]
        old = json.loads((BASE/"accepted"/(old_key+".json")).read_bytes())["payload"]
        paired = json.loads((PAIRED/"accepted"/(req+".json")).read_bytes())["payload"]
        if digest(old)!=old_key or digest(paired)!=req or paired["上轮回执"]!=old_key or paired["服务配方"]!=service:
            raise ValueError("开发范围与回执不符")
        source_file = folder/(chip+".npz")
        if chip not in seen:
            take = np.concatenate([np.flatnonzero(valid&(states==i)) for i in service])
            raw = np.stack([d["sensor"][f"{chip}_sensor{i}"][take] for i in range(4)],axis=1)
            np.savez(source_file,group=states[take].astype(int),cycles=raw)
            seen.add(chip)
        manifest = DEST/(key+".json")
        atomic_json(manifest,{"装机数":int(size),"阈值":old["阈值"],"旧权重":os.path.relpath(BASE/(key+".joblib"),DEST),
                    "旧摘要":old["权重SHA256"],"配对权重":os.path.relpath(PAIRED/(key+".joblib"),DEST),"配对摘要":paired["权重"]})
        identity = sha256(manifest.read_bytes()).hexdigest()
        proc = subprocess.run([sys.executable,"-B",str(CLI),"--release",str(manifest),"--sha256",identity,"--input",str(source_file)],
                              stdout=subprocess.PIPE,check=True,text=True,encoding="utf-8")
        sent = json.loads(proc.stdout)
        if sent["ids"]!=service:raise ValueError("重发配方错序")
        match = {"现场":PRIMARY,"统一":"统一装机验证配对权重","全曲线":"全曲线共同映射回源",
                 "按预算固定":"统一装机验证配对权重" if size=="20" else "全曲线共同映射回源"}
        for a,b in match.items():
            if sent["分类"][a]!=paired["分类"][b] or not np.allclose(sent["浓度"][a],paired["浓度"][b],atol=1e-8,rtol=1e-12):
                raise ValueError("原始入口与开发回执不同")
        new_key = digest([key,sent])
        if not receiver.submit(new_key,sent)["accepted"]:raise ValueError("未交接")
        records[key] = {"发布SHA256":identity,"原预测":req,"重发":new_key}
    scores = {}
    for key,r in records.items():
        sent = json.loads((receiver.output/(r["重发"]+".json")).read_bytes())["payload"]
        if digest([key,sent])!=r["重发"]:raise ValueError("重发回执身份改变")
        _,gas,_ = key.split("-")
        cut = json.loads((DEST/(key+".json")).read_bytes())["阈值"]
        values = np.array([target[gas][np.flatnonzero(valid&(states==i))[0]] for i in service])
        yy = 1+(values>cut)
        scores[key] = {k:{"F1":float(f1_score(yy,v,average="macro")),"弱类召回":float(recall_score(yy,v,average=None).min()),
                          "浓度误差":float(np.mean((values-sent["浓度"][k])**2)**.5)} for k,v in sent["分类"].items()}
    means = {k:{m:float(np.mean([r[k][m] for r in scores.values()])) for m in ("F1","弱类召回","浓度误差")} for k in match}
    summary = {"自主F1":means["现场"]["F1"],"按预算固定F1":means["按预算固定"]["F1"],"统一配比F1":means["统一"]["F1"],
               "论文全曲线F1":means["全曲线"]["F1"],"每种策略交付条数":len(service)*len(records)}
    atomic_json(DEST/"scores.json",{"决策":summary,"means":means,"scores":scores,"安装":records,"服务配方数":len(service)})
    print(json.dumps({"七芯片原始入口":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as tmp:
        if (DEST/"scores.json").exists():raise ValueError("已完成原始交付")
        run(Path(tmp))
