"""相同第一年服务波形，经部署清单重发加入配方训练后的预测。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from hashlib import file_digest,sha256

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score,recall_score
from verify_year_sensor_entry import DEST as ENTRY, BASE as INITIAL, DATA, NAME, HASH, CLI, LocalStore,LocalAcceptance,atomic_json,digest
from score_mixture_response import DEST as PRIOR

DEST = PRIOR.parent/"mixture-response-entry"


def run(folder, destination=DEST, metadata_policy=False):
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    entry = json.loads((ENTRY/"scores.json").read_bytes())
    with (DATA/NAME).open("rb") as f:
        if file_digest(f,"md5").hexdigest()!=HASH:raise ValueError("原始包身份变化")
    data = loadmat(DATA/NAME,variable_names=["target","sensor"],simplify_cells=True)
    t = data["target"]
    valid = np.isfinite(t["states"]) & np.isfinite(t["acetone"]) & np.isfinite(t["toluene"])
    receiver = LocalAcceptance();receiver.output = destination/"accepted"
    events = {}
    ids_by_chip = {}
    for key, request in previous["requests"].items():
        chip,gas,size = key.split("-")
        manifest = ENTRY/(key+".json")
        if sha256(manifest.read_bytes()).hexdigest()!=entry["安装"][key]["发布SHA256"]:raise ValueError("标气安装身份变化")
        pack = json.loads(manifest.read_bytes())
        old = json.loads((PRIOR/"accepted"/(request+".json")).read_bytes())["payload"]
        if digest(old)!=request or old["上轮"]!=entry["安装"][key]["重发"]:raise ValueError("训练或服务身份变化")
        ids = old["配方"]
        raw_file = folder/(chip+".npz")
        if chip not in ids_by_chip:
            indices = np.concatenate([np.flatnonzero(valid & (t["states"]==i)) for i in ids])
            cycles = np.stack([data["sensor"][chip+f"_sensor{j}"][indices] for j in range(4)],axis=1)
            np.savez(raw_file,group=t["states"][indices].astype(int),cycles=cycles)
            ids_by_chip[chip] = ids
        if ids!=ids_by_chip[chip]:raise ValueError("重发配方变化")
        for k in ("旧权重","配对权重"):
            pack[k] = os.path.relpath(ENTRY/pack[k],destination)
        pack.update({"配方权重":os.path.relpath(PRIOR/(key+".joblib"),destination),"配方摘要":old["权重"]})
        if metadata_policy:
            pack["目标气体"] = gas
        path = destination/(key+".json")
        atomic_json(path,pack)
        sha = sha256(path.read_bytes()).hexdigest()
        proc = subprocess.run([sys.executable,"-B",str(CLI),"--release",str(path),"--sha256",sha,"--input",str(raw_file)],
                              check=True,text=True,encoding="utf-8",stdout=subprocess.PIPE)
        sent = json.loads(proc.stdout)
        if sent["ids"]!=ids:raise ValueError("配方错位")
        for a,b in (("配方现场","现场"),("配方固定","固定"),("配方论文","论文")):
            name = previous["decision"][b]
            if (sent["分类"][a]!=old["分类"][name]
                    or not np.allclose(sent["浓度"][a],old["预测"][name],atol=1e-8,rtol=1e-12)):
                raise ValueError("配方跨进程预测不同")
        if metadata_policy:
            ref = {"acetone-20":("原固定与只拟合目标气体响应各半",None),
                   "acetone-80":("只拟合目标气体响应","原论文与只拟合目标气体响应各半"),
                   "toluene-20":("原论文与在目标标气重估各成分响应各半","原论文与在目标标气重估各成分响应各半"),
                   "toluene-80":("在目标标气重估各成分响应","原论文与在目标标气重估各成分响应各半")}
            for arm,field in zip(("按气体与预算固定","按气体与预算论文"),ref[gas+"-"+size],strict=True):
                if field is None:
                    first = json.loads((INITIAL/"scores.json").read_bytes())
                    identity = first["requests"][key]
                    payload = json.loads((INITIAL/"accepted"/(identity+".json")).read_bytes())["payload"]
                    if digest(payload)!=identity:raise ValueError("首轮论文回执身份改变")
                    expected = payload["分类"][first["decision"]["论文"]]
                else:
                    expected = old["分类"][field]
                if sent["分类"][arm]!=expected:raise ValueError("固定配方实际预测未复现")
        req = digest([key,sent])
        if not receiver.submit(req,sent)["accepted"]:raise ValueError("服务未接收")
        events[key] = {"回执":req,"清单SHA256":sha,"阈值":pack["阈值"]}
    score = {}
    for key,r in events.items():
        _,gas,_ = key.split("-")
        p = json.loads((receiver.output/(r["回执"]+".json")).read_bytes())["payload"]
        if digest([key,p])!=r["回执"]:raise ValueError("回执校验不同")
        yy = 1+(np.array([t[gas][np.flatnonzero(valid&(t["states"]==i))[0]] for i in p["ids"]])>r["阈值"])
        score[key] = {k:{"F1":float(f1_score(yy,v,average="macro")),"弱类召回":float(recall_score(yy,v,average=None).min())}
                      for k,v in p["分类"].items()}
    means = {k:float(np.mean([s[k]["F1"] for s in score.values()])) for k in p["分类"]}
    summary = {"配方现场F1":means["配方现场"],"配方固定F1":means["配方固定"],"配方论文F1":means["配方论文"],
               "每种策略交付条数":sum(len(v) for v in ids_by_chip.values())*4}
    if metadata_policy:
        summary.update({"按气体与预算固定F1":means["按气体与预算固定"],"按气体与预算论文F1":means["按气体与预算论文"]})
    atomic_json(destination/"scores.json",{"实发":summary,"means":means,"scores":score,"安装":events})
    print(json.dumps({"配方完整发放":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(),TemporaryDirectory(dir=DEST) as tmp:
        if (DEST/"scores.json").exists():raise ValueError("重复交付")
        run(Path(tmp))
