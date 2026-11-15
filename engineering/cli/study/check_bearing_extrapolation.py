"""只用其它公开训练设备拟合，比较单次振动测量对留出轴承剩余分钟的外推能力。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, official, read

OUT = BASE/"bearing-extrapolation"


def log_features(x):
    return np.sign(x)*np.log1p(np.abs(x))


def ridge_predict(x, y, query):
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale>1e-12, scale, 1.0)
    z = (x-center)/scale
    gram = z.T@z + len(y)*.1*np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T@(y-y.mean()))
    return np.maximum(0, y.mean()+((query-center)/scale)@coef)


def select_sensor(x,y,units):
    """内层只看外层训练设备的八点档案，每台在验证中等权。"""
    errors = np.zeros(x.shape[1])
    for u in np.unique(units):
        test = units==u
        for j in range(x.shape[1]):
            pred = ridge_predict(x[~test,j:j+1], y[~test], x[test,j:j+1])
            errors[j] += np.mean((pred-y[test])**2)
    return int(np.argmin(errors))


def summarize(records, truth):
    errors = {unit: np.array(pred)-np.array([truth[i] for i in ids]) for unit,(ids,pred) in records.items()}
    by_device = {unit:float(np.sqrt(np.mean(e*e))) for unit,e in errors.items()}
    pooled = np.concatenate(list(errors.values()))
    return {"pooled":float(np.sqrt(np.mean(pooled**2))), "device_balanced":float(np.sqrt(np.mean(np.square(list(by_device.values()))))),
            "worst_device":max(by_device.values()), "devices":by_device}


def main():
    package, checksum, batches = official("bearing")
    ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows = list(package.train)
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    X = np.array([r.values for r in rows])
    units = np.array([r.unit_id for r in rows])
    training = np.array([r.sample_id in ids for r in rows])
    if len(ids)!=72 or {r.sample_id for r in rows if r.sample_id in ids}!=ids or len(set(units))!=9:
        raise ValueError("本对照要求原登记九台设备及 72 个拟合点")
    long_units = ("Bearing3_1","Bearing3_2")
    folds = {unit:[unit] for unit in sorted(set(units))}
    folds["both-long"] = list(long_units)
    names = package.task.features
    rms = [names.index(c+".rms") for c in ("horizontal","vertical")]
    output, details = {}, {}
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"B") as trainer:
        path = Path(tmp)/"original.npz"
        path.write_bytes(package.checkpoint)
        parent = trainer.Model.load(path)
        for fold,held in folds.items():
            test = np.isin(units,held)
            fit = training & ~test
            x, query = X[fit],X[test]
            fit_ids = [r.sample_id for r,keep in zip(rows,fit,strict=True) if keep]
            y = np.array([r.label for r,keep in zip(rows,fit,strict=True) if keep])
            target_ids = [r.sample_id for r,keep in zip(rows,test,strict=True) if keep]
            if set(fit_ids).intersection(target_ids):
                raise ValueError("留出测量进入拟合库")
            lx,lq = log_features(x),log_features(query)
            start = perf_counter()
            j = select_sensor(lx,y,units[fit])
            forest = RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=17,n_jobs=1)
            pred = {"constant":np.full(len(query), y.mean()), "forest":forest.fit(x,y).predict(query),
                    "raw-ridge":ridge_predict(x,y,query), "log-ridge":ridge_predict(lx,y,lq),
                    "two-rms":ridge_predict(lx[:,rms],y,lq[:,rms]), "nested-sensor":ridge_predict(lx[:,j:j+1],y,lq[:,j:j+1]),
                    "seen-official":parent.predict(query)}
            if any(not np.isfinite(p).all() for p in pred.values()):
                raise ValueError("模型产生非有限输出")
            details[fold] = {"held":held, "fit_ids":fit_ids, "fit_max_minutes":float(y.max()), "nested_sensor":names[j],
                             "elapsed_seconds":perf_counter()-start}
            output[fold] = {"ids":target_ids,"predictions":{k:v.tolist() for k,v in pred.items()}}
    issued = OUT/"issued.json"
    atomic_json(issued,{"source":checksum,"features":names,"folds":output,
                        "fit":{k:{n:v for n,v in d.items() if n!="elapsed_seconds"} for k,d in details.items()}})
    output = read(issued)["folds"]
    truth = {r.sample_id:r.label for r in rows}
    uid = {r.sample_id:r.unit_id for r in rows}
    scores = {}
    for suite, keys in (("single",[k for k in folds if k!="both-long"]),("both-long",["both-long"])):
        results = {}
        for k in keys:
            ids_fold = output[k]["ids"]
            for arm,p in output[k]["predictions"].items():
                devices = results.setdefault(arm,{})
                for i,value in zip(ids_fold,p,strict=True):
                    group = devices.setdefault(uid[i],([],[]))
                    group[0].append(i)
                    group[1].append(value)
        scores[suite] = {arm:summarize(groups,truth) for arm,groups in results.items()}
    atomic_json(OUT/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"fit_archive_digest":digest(sorted(ids)),
                                  "fit_and_predict_seconds":{k:v["elapsed_seconds"] for k,v in details.items()}, "scores":scores})
    for k,s in scores.items():
        print(k,{a:[round(v[n],2) for n in ("pooled","device_balanced","worst_device")] for a,v in s.items()})


if __name__=="__main__":
    main()
