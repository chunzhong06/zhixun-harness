"""在同一标气预算下，实验室已知的其他成分可用于训练，服务端只见传感器。"""
import json
from hashlib import sha256
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score,recall_score

from score_year_gas_calibration import DATA, NAME, GASES, measurements, GasBase,YearTools, LocalStore,LocalAcceptance,atomic_json,digest
from score_year_sensor_pairs import DEST as PRIOR, PairedYears, PairedChoice, joint
from verify_year_sensor_entry import DEST as ENTRY
from harness.models.mixture_response import NAMES, PRIMARY, fit_tools, CompositionTrial

DEST = PRIOR.parent/"mixture-response"
CHEM = ("acetone","carbonMonoxide","ethanol","ethylAcetate","formaldehyde","hydrogen","toluene","water")


def small_anchor(base, src, yy, tx, known, hold, outer_keep, cut):
    """在外层试用折内重新安装旧固定，内层验证配方也从两端训练剔除。"""
    hist = np.empty((len(known),7))
    distances = np.empty(len(known))
    folds = np.arange(len(known))%4
    def paired_table(source_model, fit, predict):
        old = YearTools().fit(source_model,tx[fit],yy[fit]).outputs(tx[predict])
        paired = PairedYears().fit(source_model,src[fit],tx[fit],yy[fit])
        values,dist = paired.predict(tx[predict])
        return joint(old,values),dist
    for j in range(4):
        test = folds==j
        keep = outer_keep.copy()
        keep[known[test]] = False
        folded = GasBase().fit(src[keep],yy[keep])
        hist[test],distances[test] = paired_table(folded,known[~test],known[test])
    table,dist = paired_table(base,known,hold)
    return PairedChoice().fit(hist,distances,yy[known],cut).predict(table,dist)["统一装机验证配对权重"]


def run():
    ids,x,y = measurements()
    source,*targets = sorted(x)
    src = x[source][:120]
    raw = loadmat(DATA/NAME,variable_names=["target"],simplify_cells=True)["target"]
    # 辅助气体只读取已知的源及装机配方（前120种），不加入服务输入。
    c = np.stack([[raw[n][np.flatnonzero(raw["states"]==uid)[0]] for n in CHEM] for uid in ids[:120]])
    if not np.isfinite(c).all():raise ValueError("标气缺少已知配方")
    parent = json.loads((ENTRY/"scores.json").read_bytes())
    receiver = LocalAcceptance()
    receiver.output = DEST/"accepted"
    events = {}
    for gas in GASES:
        index = CHEM.index(gas)
        yy = c[:,index]
        cut = np.median(yy)
        for size in (20,80):
            folds = np.arange(size)%4
            bases, kept = [], []
            for k in range(4):
                keep = np.ones(120,bool);keep[np.flatnonzero(folds==k)] = False
                kept.append(keep)
                bases.append(GasBase().fit(src[keep],yy[keep]))
            for chip in targets:
                key = f"{chip}-{gas}-{size}"
                r = parent["安装"][key]
                old = json.loads((ENTRY/"accepted"/(r["重发"]+".json")).read_bytes())["payload"]
                if digest([key,old])!=r["重发"] or old["ids"]!=ids[120:].tolist():raise ValueError("开发基准变化")
                tx = x[chip][:size]
                trial = np.empty((size,len(NAMES)+1))
                for k in range(4):
                    test = folds==k
                    keep = kept[k]
                    bank = fit_tools(src[keep],c[keep],tx[~test],c[:size][~test],index)
                    trial[test,1:] = np.column_stack([m.predict(tx[test]) for m in bank])
                    if size==80:
                        paired = PairedYears().fit(bases[k],src[:size][~test],tx[~test],yy[:size][~test])
                        trial[test,0] = paired.predict(tx[test])[0][:,3]
                    else:
                        trial[test,0] = small_anchor(bases[k],src,yy,tx,np.flatnonzero(~test),np.flatnonzero(test),keep,cut)
                calib = CompositionTrial().fit(trial,yy[:size],cut)
                fitted = fit_tools(src,c,tx,c[:size],index)
                weight = DEST/(key+".joblib")
                joblib.dump((fitted,calib),weight,compress=3)
                fitted,calib = joblib.load(weight)
                service = x[chip][120:]
                vals = np.column_stack((old["浓度"]["按预算固定"],*[m.predict(service) for m in fitted]))
                p = dict(zip(("原按预算固定",*NAMES),vals.T,strict=True))
                p.update(calib.predict(vals))
                p["原同工具论文"] = np.asarray(old["浓度"]["全曲线"])
                for j,n in enumerate(NAMES,1):
                    p["原固定与"+n+"各半"] = (vals[:,j]+vals[:,0])/2
                    p["原论文与"+n+"各半"] = (vals[:,j]+p["原同工具论文"])/2
                payload = {"配方":ids[120:].tolist(),"气体":gas,"阈值":float(cut),"上轮":r["重发"],"权重":sha256(weight.read_bytes()).hexdigest(),
                           "预测":{n:v.tolist() for n,v in p.items()},"分类":{n:(1+(v>cut)).tolist() for n,v in p.items()}}
                h = digest(payload)
                if not receiver.submit(h,payload)["accepted"]:raise ValueError("交付失败")
                events[key] = h
    scores = {}
    for key,h in events.items():
        part = json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
        if digest(part)!=h or part["配方"]!=ids[120:].tolist():raise ValueError("接收身份改变")
        target = y[part["气体"]][120:]
        yy = 1+(target>part["阈值"])
        scores[key] = {n:{"F1":float(f1_score(yy,v,average="macro")),"弱类召回":float(recall_score(yy,v,average=None).min()),
                          "浓度误差":float(np.mean((target-part["预测"][n])**2)**.5)} for n,v in part["分类"].items()}
    means = {n:{m:float(np.mean([s[n][m] for s in scores.values()])) for m in ("F1","弱类召回","浓度误差")} for n in p}
    fixed = max((n for n in means if n!=PRIMARY),key=lambda n:means[n]["F1"])
    paper = max((n for n in means if "论文" in n),key=lambda n:means[n]["F1"])
    gap = means[PRIMARY]["F1"]-max(means[n]["F1"] for n in (fixed,paper))
    d = {"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
         "论文F1":means[paper]["F1"],"差值":gap,"进入独立":gap>=.01 and means[PRIMARY]["弱类召回"]>=means[fixed]["弱类召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":scores,"requests":events})
    print(json.dumps({"混合配方响应":d},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同项已经完成")
        run()
