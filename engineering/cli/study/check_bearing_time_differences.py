"""用相邻时刻训练误差给单条轴承预测加约束，严格按整台设备隔离。"""
from hashlib import sha256

import numpy as np

from check_bearing_extrapolation import OUT, log_features, ridge_predict, summarize
from compare_group_objectives import atomic_json, digest, metric, official, read


def paired(x,y,rows,query):
    scale = x.std(0)
    scale = np.where(scale>1e-12,scale,1.)
    center = x.mean(0)
    z = (x-center)/scale
    dx,dy = [],[]
    for unit in sorted({r.unit_id for r in rows}):
        idx = sorted((i for i,r in enumerate(rows) if r.unit_id==unit),key=lambda i:rows[i].time_end)
        if len(idx)!=8 or any(rows[j].time_end<=rows[i].time_end for i,j in zip(idx,idx[1:])):
            raise ValueError("相邻差分需要每个训练设备有八个独立时刻")
        dx.extend(np.diff(z[idx],axis=0))
        dy.extend(np.diff(y[idx]))
    dx,dy = np.asarray(dx),np.asarray(dy)
    weight = len(z)/len(dx)
    gram = z.T@z + weight*(dx.T@dx) + len(z)*.1*np.eye(z.shape[1])
    coef = np.linalg.solve(gram,z.T@(y-y.mean())+weight*(dx.T@dy))
    return np.maximum(0.,y.mean()+((query-center)/scale)@coef)


def main():
    pkg,checksum,batches = official("bearing")
    fit_ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    source = OUT/"issued.json"
    prior = read(source)
    if sha256(source.read_bytes()).hexdigest()!=read(OUT/"scores.json")["issued_sha256"] or prior["source"]!=checksum:
        raise ValueError("原有对照摘要不符")
    result, metadata = {},{}
    all_rows = {r.sample_id:r for r in pkg.train}
    for fold,reference in prior["folds"].items():
        ids = reference["ids"]
        test = [all_rows[i] for i in ids]
        held = {r.unit_id for r in test}
        train = [r for r in pkg.train if r.sample_id in fit_ids and r.unit_id not in held]
        if set(ids)&{r.sample_id for r in train}:
            raise ValueError("留出测量进入训练")
        x,q = (np.asarray([r.values for r in rows]) for rows in (train,test))
        y = np.array([r.label for r in train])
        arms = {a:reference["predictions"][a] for a in ("constant","forest","raw-ridge","log-ridge")}
        for view,bx,bq in (("raw",x,q),("log",log_features(x),log_features(q))):
            if not np.allclose(ridge_predict(bx,y,bq),arms[view+"-ridge"],rtol=1e-9,atol=1e-8):
                raise ValueError("同输入普通目标未复现")
            pred = paired(bx,y,train,bq)
            if not np.isfinite(pred).all():
                raise ValueError("训练结果含非有限预测")
            arms[view+"-paired"] = pred.tolist()
        result[fold] = {"ids":ids,"predictions":arms}
        metadata[fold] = {"fit_ids":digest([r.sample_id for r in train]),"max_training_life":float(max(y)),
                          "max_prediction":{k:max(v) for k,v in arms.items()}}
    path = OUT/"time-difference"/"issued.json"
    atomic_json(path,{"source":checksum,"folds":result,"metadata":metadata})
    saved = read(path)["folds"]
    answers = {i:r.label for i,r in all_rows.items()}
    last = {u:max(r.time_end for r in all_rows.values() if r.unit_id==u) for u in {r.unit_id for r in all_rows.values()}}
    reports = {}
    for label,folds in (("single",[k for k in saved if k!="both-long"]),("both-long",["both-long"])):
        table = {}
        for arm in next(iter(saved.values()))["predictions"]:
            groups,early = {},{}
            for k in folds:
                for i,p in zip(saved[k]["ids"],saved[k]["predictions"][arm],strict=True):
                    r=all_rows[i]
                    keys,values=groups.setdefault(r.unit_id,([],[]))
                    keys.append(i)
                    values.append(p)
                    if r.time_end<=last[r.unit_id]/4:
                        early[i]=p
            table[arm] = {**summarize(groups,answers),"first_quarter":metric("bearing",answers,early)}
        reports[label]=table
        print(label,{a:tuple(round(v[k],2) for k in ("pooled","worst_device","first_quarter")) for a,v in table.items()})
    atomic_json(path.with_name("scores.json"),{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"suites":reports})


if __name__=="__main__":
    main()
