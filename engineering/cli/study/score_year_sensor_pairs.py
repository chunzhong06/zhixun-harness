"""比较按已知配方修正整个波形与只修正浓度误差，共用原准入、标签和预测样本。"""
from hashlib import sha256
import json
import joblib
import numpy as np
from sklearn.metrics import f1_score,recall_score

from score_year_gas_calibration import DEST as PRIOR, GASES, measurements, GasBase, YearTools, LocalAcceptance, LocalStore, atomic_json, digest
from harness.models.year_sensor_pairs import PairedYears, PairedChoice, TOOLS, PRIMARY

DEST = PRIOR.parent/"year-sensor-pairs"
ANCHOR = "两种目标监督各半"
REF = "开发最强论文固定组合"


def joint(old,pairs):
    # 上轮固定为目标岭与源误差修正各半，论文强对照另含两层续训。
    anchor = (old[:,3]+old[:,4])/2
    paper = (old[:,5]+old[:,4])/2
    return np.column_stack((anchor,pairs,paper))


def run():
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    ids,x,y = measurements()
    source,*targets = sorted(x)
    src = x[source][:120]
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    events = {}
    for gas in GASES:
        truth = y[gas][:120]
        cut = float(np.median(truth))
        for size in (20,80):
            folds = np.arange(size)%4
            bases = []
            for k in range(4):
                keep = np.ones(120,bool)
                keep[np.flatnonzero(folds==k)] = False
                bases.append(GasBase().fit(src[keep],truth[keep]))
            for chip in targets:
                key = f"{chip}-{gas}-{size}"
                request = previous["requests"][key]
                parent = json.loads((PRIOR/"accepted"/(request+".json")).read_bytes())["payload"]
                weight = PRIOR/(key+".joblib")
                if (ids[120:].tolist()!=parent["服务配方"] or ids[:size].tolist()!=parent["装机配方"]
                        or digest(parent)!=request or sha256(weight.read_bytes()).hexdigest()!=parent["权重SHA256"]):
                    raise ValueError("原交付或装机包身份变化")
                fitted, _ = joblib.load(weight)
                tx = x[chip][:size]
                held = np.empty((size,len(TOOLS)+2))
                distance = np.empty(size)
                for k in range(4):
                    test = folds==k
                    paired = PairedYears().fit(bases[k],src[:size][~test],tx[~test],truth[:size][~test])
                    old = YearTools().fit(bases[k],tx[~test],truth[:size][~test]).outputs(tx[test])
                    cols,distance[test] = paired.predict(tx[test])
                    held[test] = joint(old,cols)
                selector = PairedChoice().fit(held,distance,truth[:size],cut)
                trained = PairedYears().fit(fitted.base,src[:size],tx,truth[:size])
                package = DEST/(key+".joblib")
                joblib.dump((trained,selector),package,compress=3)
                trained,selector = joblib.load(package)
                service = x[chip][120:]
                cols,distance = trained.predict(service)
                bank = joint(fitted.outputs(service),cols)
                for j,which in ((0,"固定"),(-1,"论文")):
                    if not np.allclose(bank[:,j],parent["浓度"][previous["decision"][which]],rtol=0,atol=1e-9):
                        raise ValueError("上轮固定或论文强对照未复现")
                new = dict(zip((ANCHOR,*TOOLS,REF),bank.T,strict=True))
                new.update(selector.predict(bank,distance))
                for i,name in enumerate(TOOLS,1):
                    new["原固定各半/"+name] = (bank[:,0]+bank[:,i])/2
                    new["原论文各半/"+name] = (bank[:,-1]+bank[:,i])/2
                payload = {"服务配方":ids[120:].tolist(),"阈值":cut,"气体":gas,"上轮回执":request,
                           "浓度":{a:v.tolist() for a,v in new.items()},"分类":{a:(1+(v>cut)).tolist() for a,v in new.items()},
                           "权重":sha256(package.read_bytes()).hexdigest()}
                req = digest(payload)
                if not receiver.submit(req,payload)["accepted"]:raise ValueError("未交付")
                events[key] = req
    scores = {}
    for k,req in events.items():
        p = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest(p)!=req or p["服务配方"]!=ids[120:].tolist():raise ValueError("服务身份变化")
        values = y[p["气体"]][120:]
        yy = 1+(values>p["阈值"])
        scores[k] = {a:{"F1":float(f1_score(yy,v,average="macro")),"最弱召回":float(recall_score(yy,v,average=None).min()),
                        "浓度RMSE":float(np.mean((values-p["浓度"][a])**2)**.5)} for a,v in p["分类"].items()}
    means = {a:{m:float(np.mean([r[a][m] for r in scores.values()])) for m in ("F1","最弱召回","浓度RMSE")} for a in new}
    fixed = max((a for a in means if a!=PRIMARY),key=lambda a:means[a]["F1"])
    paper = max((a for a in means if a==REF or a.startswith("原论文各半/") or a in TOOLS),key=lambda a:means[a]["F1"])
    gap = means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    d = {"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
         "论文F1":means[paper]["F1"],"差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"requests":events,"scores":scores,"means":means})
    print(json.dumps({"同配方波形校准":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已有开发实测")
        run()
