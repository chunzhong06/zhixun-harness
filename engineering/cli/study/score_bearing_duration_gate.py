"""从已经结束的源轴承学习每台寿命档，部署决策只使用单条振动。"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from score_bearing_log_target import DEST as PRIOR, ARCHIVE, ROOT, read, read_official_bearing, LocalStore, LocalAcceptance, atomic_json, digest

DEST = PRIOR.parent/"bearing-duration-gate"


def gate(rows, ids):
    samples = {r.sample_id:r for r in rows}
    fit = [samples[k] for k in ids]
    ends = {r.unit_id:r.time_end-1+r.label for r in fit}
    cutoff = float(np.median(list(ends.values())))
    units = sorted(ends)
    flags = np.array([int(ends[u]>cutoff) for u in units])
    x = np.asarray([r.values for r in fit])
    scale = StandardScaler().fit(x)
    z = scale.transform(x)
    records = {}
    for name,groups in (("actual",flags),("shuffled",np.roll(flags,1))):
        lookup = dict(zip(units,groups,strict=True))
        y = [lookup[r.unit_id] for r in fit]
        model = LogisticRegression(C=1.,random_state=17,class_weight="balanced",max_iter=1000).fit(z,y)
        records[name] = {"coef":model.coef_[0].tolist(),"intercept":float(model.intercept_[0])}
    return {"records":records,"center":scale.mean_.tolist(),"scale":scale.scale_.tolist(),
            "prior":float(flags.mean()),"cutoff":cutoff,"source_ids":ids}


def predict(state,x,long,short):
    z = (np.asarray(x)-state["center"])/state["scale"]
    probs = {}
    for key,info in state["records"].items():
        score = z@np.asarray(info["coef"])+info["intercept"]
        probs[key] = 1/(1+np.exp(-np.clip(score,-700,700)))
    long = np.asarray(long)
    short = np.broadcast_to(short,long.shape)
    result = {"long":long,"short":short,"hard":np.where(probs["actual"]>.5,long,short)}
    for name,weight in (("soft",probs["actual"]),("shuffled",probs["shuffled"]),("prior",state["prior"])):
        result[name] = weight*long+(1-weight)*short
    return result,probs["actual"]


def metric(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此协议已有结果")
        pkg = read_official_bearing(ARCHIVE)
        training = read(PRIOR/"training.json")
        complete = read(PRIOR/"scores.json")
        base = {r.sample_id:r for r in pkg.train}
        held_scores,correct,constants = {},[],[]
        for unit,fold in training["folds"].items():
            bundle = gate(pkg.train,fold["source_ids"])
            query = [base[i] for i in fold["held_ids"]]
            pred,prob = predict(bundle,[r.values for r in query],fold["predictions"]["minutes"],fold["median"])
            y = np.asarray([r.label for r in query])
            flag = query[0].time_end-1+query[0].label>bundle["cutoff"]
            for k,v in pred.items():
                held_scores.setdefault(k,[]).extend(((y-v)**2).tolist())
            correct.append(float(np.mean((prob>.5)==flag)))
            constants.append(float((bundle["prior"]>.5)==flag))
        held = {k:float(np.sqrt(np.mean(e))) for k,e in held_scores.items()}
        state = gate(pkg.train,training["final"]["source_ids"])
        atomic_json(DEST/"model.json",state)
        state = read(DEST/"model.json")
        official = {r.sample_id:r for r in pkg.validation}
        extra = read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
        fifth = {r["member"]:r for r in extra["rows"]}
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        requests = {}
        for site,previous in complete["requests"].items():
            old = read(PRIOR/"accepted"/(previous+".json"))["payload"]
            if digest([site,old])!=previous:
                raise ValueError("基础工具预测身份不符")
            keys = old["ids"]
            x = [fifth[k]["features"] if site=="xjtu_fifth" else official[k].values for k in keys]
            preds,prob = predict(state,x,old["predictions"]["minutes"],old["predictions"]["median"])
            payload = {"ids":keys,"model":digest(state),"previous":previous,"long_probability":prob.tolist(),
                       "predictions":{k:v.tolist() for k,v in preds.items()}}
            req = digest([site,payload])
            ack = receiver.submit(req,payload)
            if not ack["accepted"] or ack["request_id"]!=req:
                raise RuntimeError("寿命预测交接失败")
            requests[site] = req
        results = {}
        for site,req in requests.items():
            p = read(receiver.output/(req+".json"))["payload"]
            y = np.array([extra["ends"][Path(k).parent.name]-int(Path(k).stem) if site=="xjtu_fifth" else official[k].label
                          for k in p["ids"]])
            preds = p["predictions"]
            scores = {k:metric(y,v) for k,v in preds.items()}
            scores["ideal"] = float(np.sqrt(np.mean(np.minimum((y-preds["long"])**2,(y-preds["short"])**2))))
            results[site] = scores
            print(json.dumps({site:scores},ensure_ascii=False),flush=True)
        summary = {"requests":requests,"scores":results,"source_rmse":held,
                   "source_device_accuracy":float(np.mean(correct)),"constant_accuracy":float(np.mean(constants))}
        atomic_json(DEST/"scores.json",summary)
        print(json.dumps({"源留台误差":held,"源设备等权识别率":summary["source_device_accuracy"],
                          "常数识别率":summary["constant_accuracy"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
