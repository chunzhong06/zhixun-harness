"""只按整台留出的源设备选择训练目标；交付和评价均返回分钟。"""
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from score_bearing_hybrid_readout import (
    ROOT, ARCHIVE, DEST as EARLIER, read_official_bearing, statistics, read, fingerprint, LocalStore, atomic_json,
)
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance

DEST = EARLIER.parent/"bearing-log-target"


def fit(rows,folder):
    folder.mkdir(parents=True,exist_ok=True)
    sampled = []
    for unit in sorted({r.unit_id for r in rows}):
        seq = sorted((r for r in rows if r.unit_id==unit),key=lambda r:r.time_end)
        sampled.extend(seq[j] for j in np.linspace(0,len(seq)-1,8).round().astype(int))
    x = np.asarray([r.values for r in sampled])
    y = np.asarray([r.label for r in sampled])
    models = {}
    ids = {}
    for name,target in (("minutes",y),("log",np.log1p(y))):
        model = CatBoostRegressor(iterations=100,depth=6,random_seed=17,thread_count=1,
                                  verbose=False,allow_writing_files=False)
        model.fit(x,target)
        p = folder/(name+".cbm")
        model.save_model(str(p))
        models[name] = CatBoostRegressor().load_model(str(p))
        ids[name] = fingerprint(p)
    return models,{"source_ids":[r.sample_id for r in sampled],"weights":ids,
                   "median":float(np.median(y)),"mean":float(np.mean(y))}


def forecasts(models,x):
    return {"minutes":models["minutes"].predict(x,thread_count=1),
            "log":np.maximum(0,np.expm1(models["log"].predict(x,thread_count=1)))}


def evaluate_receipts(pkg,folder):
    """重用已交付预测；等距取整的源中位数与旧配方的中位数分别计分。"""
    training = read(folder/"training.json")
    earlier = EARLIER/"issued.json"
    if fingerprint(earlier)!=read(EARLIER/"scores.json")["issued_sha256"]:
        raise ValueError("旧版服务清单有变")
    manifest = read(earlier)["cases"]
    prior = read(EARLIER/"scores.json")["cases"]
    extra = read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
    official = {r.sample_id:r for r in pkg.validation}
    names = {tuple(v["ids"]):k for k,v in manifest.items()}
    scores,requests = {},{}
    for file in sorted((folder/"accepted").glob("*.json")):
        envelope = read(file)
        p = envelope["payload"]
        case = names[tuple(p["ids"])]
        if case in requests or envelope["request_id"]!=file.stem or digest([case,p])!=file.stem:
            raise ValueError("交付回执重复或摘要不符")
        if p["weight_sha256"]!=training["final"]["weights"] or p["choice"]!=training["choice"]:
            raise ValueError("交付与训练记录不符")
        requests[case]=file.stem
        units = manifest[case]["devices"]
        y = [extra["ends"][u]-int(Path(k).stem) if case=="xjtu_fifth" else official[k].label
             for u,k in zip(units,p["ids"],strict=True)]
        scores[case] = statistics(p["predictions"],y,units)
        scores[case]["published-median"] = prior[case]["arms"]["median"]
        scores[case].update({k:prior[case]["arms"][k] for k in ("root","current","hidden","phase","mix-1")})
        print(json.dumps({case:{k:v["rmse"] for k,v in scores[case].items()}},ensure_ascii=False),flush=True)
    if requests.keys()!=manifest.keys():
        raise ValueError("实际交付缺少评价设备组")
    atomic_json(folder/"scores.json",{"requests":requests,"source_choice":training["choice"],
                "source_rmse":training["held_minutes_rmse"],"factor":training["source_smearing_factor"],"scores":scores})


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("本次冻结实验已有完整结果")
        pkg = read_official_bearing(ARCHIVE)
        if (DEST/"training.json").exists() and len(list((DEST/"accepted").glob("*.json")))==3:
            evaluate_receipts(pkg,DEST)
            return
        folds,errors,ratios = {},{"minutes":[],"log":[]},[]
        for unit in sorted({r.unit_id for r in pkg.train}):
            fit_rows = [r for r in pkg.train if r.unit_id!=unit]
            held = [r for r in pkg.train if r.unit_id==unit]
            models,record = fit(fit_rows,DEST/"folds"/unit)
            pred = forecasts(models,np.asarray([r.values for r in held]))
            y = np.asarray([r.label for r in held])
            for k,p in pred.items():
                errors[k].extend(((p-y)**2).tolist())
            ratios.append(float(np.mean((y+1)/(pred["log"]+1))))
            folds[unit] = {**record,"held_ids":[r.sample_id for r in held],"predictions":{k:p.tolist() for k,p in pred.items()}}
        validation = {k:float(np.sqrt(np.mean(e))) for k,e in errors.items()}
        selected = min(validation,key=validation.get)
        factor = float(np.mean(ratios))
        models,record = fit(pkg.train,DEST/"models")
        atomic_json(DEST/"training.json",{"folds":folds,"held_minutes_rmse":validation,"choice":selected,
                    "source_smearing_factor":factor,"final":record})
        earlier = EARLIER/"issued.json"
        if fingerprint(earlier)!=read(EARLIER/"scores.json")["issued_sha256"]:
            raise ValueError("旧版服务清单有变")
        manifest = read(earlier)["cases"]
        official = {r.sample_id:r for r in pkg.validation}
        extra = read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
        fifth = {r["member"]:r for r in extra["rows"]}
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        requests = {}
        for case,old in manifest.items():
            keys = old["ids"]
            x = np.asarray([fifth[k]["features"] if case=="xjtu_fifth" else official[k].values for k in keys])
            pred = forecasts(models,x)
            pred["log-smear"] = np.maximum(0,(1+pred["log"])*factor-1)
            for k in ("median","mean"):
                pred[k] = np.full(len(x),record[k])
            payload = {"ids":keys,"choice":selected,"weight_sha256":record["weights"],
                       "predictions":{k:v.tolist() for k,v in pred.items()}}
            request = digest([case,payload])
            ack = receiver.submit(request,payload)
            if not ack["accepted"] or ack["request_id"]!=request:
                raise RuntimeError("服务交接未确认")
            requests[case] = request
        evaluate_receipts(pkg,DEST)


if __name__=="__main__":
    main()
