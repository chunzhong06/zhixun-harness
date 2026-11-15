"""按整台设备留出预测估计启动模型权重，现场更新只消费已失效的参考设备。"""
from collections import defaultdict
from hashlib import sha256
from itertools import groupby

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from score_completed_residual_transfer import LOCAL, SourceCombinations, atomic_json, engines, femto, official, read, xjtu
from score_ncmapss_field import MEMBER_SHA, RAW, ROOT, file_hash, flights, forest, measures
from score_whole_maintenance import paths, public_batches


PROTOCOL = ROOT/"engineering/configs/experiments/life-source-stack.json"
OUT = ROOT/"results/experiments/local-defaults/whole-maintenance/source-stack"


def blend_weight(terms):
    numerator, denominator = np.mean(terms, axis=0)
    return float(np.clip(numerator/denominator,0,1)) if denominator else .5


def calibration_term(values, y):
    # 最小化 (y - forest - w * (ridge - forest))²，返回按设备可等权合并的两项。
    direction = values[:,1] - values[:,0]
    return [float(np.mean(direction*(y-values[:,0]))), float(np.mean(direction**2))]


def trained(x,y,repair_inputs):
    repair = SourceCombinations(x) if repair_inputs else None
    features = repair.transform(x) if repair is not None else x
    trees = forest().fit(features,y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.)).fit(features,y)
    def predict(values):
        v = repair.transform(values) if repair is not None else np.asarray(values)
        return np.column_stack((trees.predict(v), ridge.predict(v)))
    return predict


def source_check(rows, repair):
    units = np.asarray([r[0] for r in rows])
    x = np.asarray([r[1] for r in rows])
    y = np.asarray([r[2] for r in rows])
    ordered = sorted(set(units),key=lambda u:sha256(f"life-stack-v1/{u}".encode()).digest())
    folds = min(len(ordered),4)
    if folds < 2:
        raise ValueError("需要独立的训练设备才能留出")
    coefficients = []
    kept = []
    for j in range(folds):
        held = ordered[j::folds]
        valid = np.isin(units,held)
        predict = trained(x[~valid],y[~valid],repair)
        v = predict(x[valid])
        coeff = [calibration_term(v[units[valid]==u],y[units==u]) for u in held]
        coefficients.extend(coeff)
        kept.append({"held":list(map(str,held)),"terms":coeff})
    return trained(x,y,repair), np.mean(coefficients,axis=0), kept


def official_inputs(task):
    pkg, _, batches = official(task)
    source = [(r.unit_id,r.values,r.label) for r in batches[0].samples if r.role=="fit"]
    current,_ = paths(task,task)
    rounds, owners = public_batches(read(current/"state.json")["samples"])
    rows = [(k,owners[k],j,x) for j,ids,values,_ in rounds for k,x in zip(ids,values,strict=True)]
    pub = ({},[],rows,lambda:{r.sample_id:r.label for r in pkg.validation},[])
    external = {"femto":femto(54),"xjtu":xjtu(pkg,LOCAL/"fifth-offset6.json")} if task=="bearing" else {"phm08":engines()}
    return source, {"official-"+task:pub, **external}


def nasa_inputs():
    if file_hash(RAW)!=MEMBER_SHA:
        raise ValueError("NASA 原件身份变化")
    with h5py.File(RAW,"r") as f:
        dev,_=flights(f,"dev")
        test,_=flights(f,"test")
    last=read(OUT.parent/"ncmapss-ds01/scores.json")
    train = [(u,x,y) for u in last["source_units"] for _,x,y in dev[u]]
    refs = {u:len(dev[u]) for u in last["reference_units"]}
    known=[]
    for u in refs:
        for i in np.linspace(0,len(dev[u])-1,5).round().astype(int):
            c,x,y = dev[u][i]
            known.append((u,f"{u}/{c}",x,y))
    service=[(f"{u}/{c}",u,c-test[u][0][0]+1,x) for u in last["service_units"] for c,x,_ in test[u]]
    return train, {"nc-ds01":(refs,known,service,lambda:{f"{u}/{c}":y for u in test for c,_,y in test[u]},[])}


def evaluate(site, stream, predict, prior, source_folds):
    ends, observations, service, get_y, _ = stream
    initial = blend_weight([prior])
    by_unit = defaultdict(list)
    for u,k,x,y in observations:
        by_unit[u].append((k,x,y))
    if set(ends) & {r[1] for r in service} or set(ends)!=set(by_unit):
        raise ValueError("现场与服务设备交叉或反馈缺行")
    ready = sorted((t,u) for u,t in ends.items())
    service.sort(key=lambda r:(r[2],r[0]))
    predictions = predict([r[3] for r in service])
    data = {a:{} for a in ("forest","ridge","half","source-stack","site-stack")}
    released, terms, events = 0, [], []
    for when, items in groupby(enumerate(service),key=lambda pair:pair[1][2]):
        while released<len(ready) and ready[released][0]<when:
            end,u = ready[released]
            obs=by_unit[u]
            if len(obs)!=5:
                raise ValueError("本轮需要每个完成设备有五条回标")
            term=calibration_term(predict([r[1] for r in obs]),np.asarray([r[2] for r in obs]))
            terms.append(term)
            events.append({"time":when,"end":end,"unit":u,"ids":[r[0] for r in obs],"term":term})
            released+=1
        w=blend_weight([prior,np.mean(terms,axis=0)]) if terms else initial
        for i,(k,_,_,_) in items:
            a,b=predictions[i]
            for arm,weight in zip(data,(0.,1.,.5,initial,w),strict=True):
                data[arm][k]=float(a+(b-a)*weight)
    file=OUT/site/"issued.json"
    atomic_json(file,{"protocol":file_hash(PROTOCOL),"source_folds":source_folds,"source_term":prior.tolist(),
                      "initial_ridge_weight":initial,"events":events,"predictions":data})
    receipt=read(file)
    pred=receipt["predictions"]
    ids=[r[0] for r in service]
    if set(ids)!=pred["half"].keys() or len(set(ids))!=len(ids):
        raise ValueError("服务行异常")
    truth=get_y()
    y=np.asarray([truth[k] for k in ids])
    owner=np.asarray([r[1] for r in service])
    table={a:measures(y,np.asarray([p[k] for k in ids]),owner) for a,p in pred.items()}
    controls=("half","forest","ridge")
    passes={a:all(table[a]["rmse"]<=.99*table[c]["rmse"] and table[a]["worst"]<=1.01*table[c]["worst"]
                  for c in controls) for a in ("source-stack","site-stack")}
    if site=="nc-ds01":
        original=read(OUT.parent/"ncmapss-ds01/scores.json")["scores"]
        if any(abs(table[a]["rmse"]-original[a]["rmse"])>1e-8 for a in controls):
            raise ValueError("NASA 固定参照在同批输入未复现")
    result={"issued_sha256":file_hash(file),"rows":len(ids),"units":len(set(owner)),
            "source_fits":2*(len(source_folds)+1),"target_model_fits":0,"used_labels":5*len(terms),
            "source_ridge_weight":initial,"last_ridge_weight":w,"scores":table,"passes":passes}
    atomic_json(file.with_name("scores.json"),result)
    print(site,round(initial,3),round(w,3),{a:round(v["rmse"],4) for a,v in table.items()},flush=True)
    return result


if __name__=="__main__":
    results={}
    for domain in ("bearing","engine","nc-ds01"):
        source,streams=nasa_inputs() if domain=="nc-ds01" else official_inputs(domain)
        predict,prior,folds=source_check(source,domain=="engine")
        for site,stream in streams.items():
            results[site]=evaluate(site,stream,predict,prior,folds)
    atomic_json(OUT/"scores.json",results)
