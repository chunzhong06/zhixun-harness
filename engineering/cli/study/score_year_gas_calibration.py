"""在第一年原始标气文件上完成现场安装试验，第二年保持封存。"""
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"engineering/src"),str(ROOT/"engineering/cli/data")]
from prepare_sgp_year_install import DATA, NAME, HASH
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from harness.models.year_gas_calibration import GasBase, YearTools, YearInstallation, NAMES, PRIMARY, feature_cycles

DEST = ROOT/"results/experiments/local-defaults/whole-maintenance/year-gas-calibration"
GASES = ("acetone","toluene")


def measurements():
    with (DATA/NAME).open("rb") as f:
        if file_digest(f,"md5").hexdigest()!=HASH:
            raise ValueError("第一次标定MAT身份改变")
    data = loadmat(DATA/NAME,variable_names=["sensor","target"],simplify_cells=True)
    target,raw = data["target"],data["sensor"]
    codes = target["states"]
    valid = np.isfinite(codes)
    for gas in GASES:
        valid &= np.isfinite(target[gas])
    states = codes[valid].astype(int)
    if not np.array_equal(codes[valid],states):
        raise ValueError("配方身份不是整数")
    ids = sorted(set(states),key=lambda uid:sha256(f"year-gas-v1/{uid}".encode()).digest())
    take = [np.flatnonzero(states==i) for i in ids]
    devices = sorted({name.rsplit("_",1)[0] for name in raw})
    if len(devices)!=7 or len(ids)<160:
        raise ValueError("第一年芯片数或独立配方不足以符合冻结设计")
    x = {}
    for chip in devices:
        readings = np.stack([raw[f"{chip}_sensor{i}"][valid] for i in range(4)],axis=1)
        features = feature_cycles(readings)
        x[chip] = np.stack([features[idx].mean(0) for idx in take])
    y = {}
    for gas in GASES:
        concentrations = target[gas][valid]
        if any(len(np.unique(concentrations[idx]))!=1 for idx in take):
            raise ValueError("一道标气有多个目标浓度")
        y[gas] = np.array([concentrations[idx[0]] for idx in take])
    return np.array(ids),x,y


def run():
    ids,features,labels = measurements()
    source,*devices = sorted(features)
    src = features[source][:120]
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    events = {}
    for gas in GASES:
        y = labels[gas][:120]
        cut = float(np.median(y))
        base = GasBase().fit(src,y)
        for count in (20,80):
            folds = np.arange(count)%4
            bases = []
            for k in range(4):
                keep = np.ones(120,bool)
                keep[np.flatnonzero(folds==k)] = False
                bases.append(GasBase().fit(src[keep],y[keep]))
            for chip in devices:
                dest = features[chip]
                target = dest[:count]
                history = np.empty((count,len(NAMES)))
                for k in range(4):
                    test = folds==k
                    trial = YearTools().fit(bases[k],target[~test],y[:count][~test])
                    history[test] = trial.outputs(target[test])
                installed = YearTools().fit(base,target,y[:count])
                selection = YearInstallation().fit(history,y[:count],cut)
                key = f"{chip}-{gas}-{count}"
                weight = DEST/(key+".joblib")
                joblib.dump((installed,selection),weight,compress=3)
                installed,selection = joblib.load(weight)
                values = installed.outputs(dest[120:])
                p = selection.predict(values)
                # 固定端可以选任意一对已安装工具各半，并给成熟端同等权利。
                for i,left in enumerate(NAMES):
                    for j,right in enumerate(NAMES[:i]):
                        p[left+"与"+right+"各半"] = (values[:,i]+values[:,j])/2
                payload = {"源器件":source,"目标器件":chip,"气体":gas,"阈值":cut,"装机配方":ids[:count].tolist(),
                           "服务配方":ids[120:].tolist(),"浓度":{k:v.tolist() for k,v in p.items()},
                           "分类":{k:(1+(v>cut)).tolist() for k,v in p.items()},
                           "权重SHA256":sha256(weight.read_bytes()).hexdigest()}
                request = digest(payload)
                if not receiver.submit(request,payload)["accepted"]:raise ValueError("服务交付失败")
                events[key] = request
    scores = {}
    for key,request in events.items():
        sent = json.loads((receiver.output/(request+".json")).read_bytes())["payload"]
        if digest(sent)!=request or sent["服务配方"]!=ids[120:].tolist():raise ValueError("原始配方交付身份改变")
        yy = labels[sent["气体"]][120:]
        truth = 1+(yy>sent["阈值"])
        scores[key] = {k:{"F1":float(f1_score(truth,v,average="macro")),"最弱召回":float(recall_score(truth,v,average=None).min()),
                          "浓度RMSE":float(np.mean((np.array(sent["浓度"][k])-yy)**2)**.5)} for k,v in sent["分类"].items()}
    means = {k:{m:float(np.mean([s[k][m] for s in scores.values()])) for m in ("F1","最弱召回","浓度RMSE")} for k in p}
    fixed = max((k for k in p if k!=PRIMARY),key=lambda k:means[k]["F1"])
    paper = max((k for k in p if "论文" in k or k=="新芯片单独标定"),key=lambda k:means[k]["F1"])
    gap = means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    decision = {"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
                "论文F1":means[paper]["F1"],"差值":gap,"第二年可开封":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":decision,"means":means,"scores":scores,"requests":events,"有效配方":len(ids),"服务配方":len(ids)-120})
    print(json.dumps({"跨器件首年开发":decision},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有第一年开发交付")
        run()
