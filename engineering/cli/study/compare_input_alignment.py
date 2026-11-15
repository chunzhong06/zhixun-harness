"""固定官方预测器，逐轮用已到达的探索测量生成输入变换；确认后才交接服务预测。"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import (
    BASE, ROOT, PROFILES, LocalStore, _trainer, atomic_json, delivered, digest, metric,
    official, read, reference_folders, remaining_cycles, remaining_minutes, summary,
)
from engine_endpoint_check import RAW, SUBSETS, read_cmapss, test_answers

OUT = BASE/"input-alignment"
METHODS = ("identity","mean","half","agreement","coral")
MIXTURE_OUT = BASE/"input-mixture"


def covariance_factor(x, power):
    eig, vectors = np.linalg.eigh(np.cov(x,rowvar=False)+np.eye(x.shape[1]))
    return (vectors * eig**power) @ vectors.T


def group_predictions(task, values, edges):
    if task=="gas":
        return values.argmax(axis=1)
    return np.searchsorted(edges,values,side="right")


def composition_trial(method, observations, target_groups, source, source_groups, center, source_sqrt):
    """先构造只改变工况比例的源参照。未知目标组没有源样本，须退回可计算的普通对齐。"""
    if not set(target_groups)<=set(source_groups):
        return propose("coral",center,source_sqrt,observations,[]), {"choice":"unsupported-use-coral"}
    counts = np.bincount(source_groups)
    target_share = np.bincount(target_groups,minlength=len(counts))/len(target_groups)
    weights = target_share[source_groups]/counts[source_groups]
    mix = weights @ source
    observed_center = observations.mean(axis=0)
    raw = float(np.linalg.norm(observed_center-center))
    residual = float(np.linalg.norm(observed_center-mix))
    explained = float(np.linalg.norm(mix-center))
    evidence = {"original_gap":raw,"remaining_gap":residual,"composition_move":explained,
                "target_shares":target_share.tolist()}
    if method=="conditional" and residual<raw:
        if residual<=explained:
            return (None,np.zeros_like(center)),{**evidence,"choice":"identity"}
    elif method=="conditional":
        return propose("coral",center,source_sqrt,observations,[]),{**evidence,"choice":"coral"}
    denominator = 1-float(weights @ weights)
    if denominator<=1e-12:
        return propose("coral",center,source_sqrt,observations,[]), {**evidence,"choice":"insufficient-use-coral"}
    centered = source-mix
    cov = (centered.T*weights) @ centered/denominator + np.eye(len(mix))
    values, vectors = np.linalg.eigh(cov)
    square_root = (vectors*np.sqrt(values)) @ vectors.T
    return propose("coral",mix,square_root,observations,[]),{**evidence,"choice":"matched"}


def propose(method, source_mean, source_sqrt, observations, unit_groups):
    """共识按独立设备等权。缺少第二组时不提出共识动作，沿用上轮已发布的版本。"""
    if method=="coral":
        mapping = covariance_factor(observations,-.5) @ source_sqrt
        return mapping, source_mean-observations.mean(axis=0) @ mapping
    shifts = np.array([observations[group].mean(axis=0)-source_mean for group in unit_groups])
    if method=="agreement":
        if len(shifts)<2:
            return None
        lo, hi = shifts.min(axis=0), shifts.max(axis=0)
        offset = np.maximum(lo,0) + np.minimum(hi,0)
    else:
        offset = shifts.mean(axis=0) * (.5 if method=="half" else 1.)
    return None, -offset


def predict(task, model, transform, ids, x):
    if transform is not None:
        matrix, offset = transform
        z = model.normalized(x)
        shifted = z+offset if matrix is None else z@matrix+offset
        x = shifted*model.scale + model.mean
    values = model.predict(x)
    if len(values)!=len(ids) or not np.isfinite(values).all():
        raise ValueError("变换后预测无效")
    if task=="gas":
        values = values.argmax(axis=1)+1
    return dict(zip(ids,values.tolist(),strict=True))


def transfer(task, model, live, sealed, destination):
    """末轮变换直接移用。用已登记的输入摘要及同一静态权重的逐条输出复核设备身份。"""
    hist = ROOT/"results/experiments/production-selection"
    control = read(BASE/"training-coverage"/"transfer"/task/"issued.json")
    if task=="bearing":
        rows = read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
        valid = [r for r in rows if "invalid_nonfinite_values" not in r]
        ids, x = ["KAIST/"+r["file"] for r in valid], np.array([r["features"] for r in valid])
        original = read(hist/"short-window/bearing/kaist/issued.json")["predictions"]["original"]
    else:
        tracks = {s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
        units = [r for group in tracks.values() for r in group]
        ids, x = [r.unit_id for r in units], np.array([(*r.values[-1][3:],*r.values[-1][:3]) for r in units])
        original = read(hist/"engine-transfer/issued.json")["predictions"]["original"]
    if digest([ids,x.tolist()])!=control["input_sha256"]:
        raise ValueError("移用输入与既有台架或终点不符")
    outputs = {name:predict(task,model,t,ids,x) for name,t in live.items()}
    if set(original)!=set(ids) or not np.allclose([outputs["identity"][i] for i in ids],[original[i] for i in ids],rtol=1e-12,atol=1e-8):
        raise ValueError("静态权重未复现既有移用预测")
    for name in ("project","mature","simple"):
        outputs[name]=control["predictions"][name]
    folder = destination/"transfer"/task
    path = folder/"issued.json"
    atomic_json(path,{"source_issued_sha256":sha256(sealed.read_bytes()).hexdigest(),
                      "input_sha256":control["input_sha256"],"predictions":outputs})
    if task=="bearing":
        end = datetime.fromisoformat(rows[-1]["time"])
        truth = {"pooled":{i:(end-datetime.fromisoformat(r["time"])).total_seconds()/60 for i,r in zip(ids,valid,strict=True)}}
    else:
        truth = {s:test_answers(RAW,s,group) for s,group in tracks.items()}
        truth["pooled"] = {i:y for group in truth.values() for i,y in group.items()}
    scores = {name:{g:metric(task,ys,{i:p[i] for i in ys}) for g,ys in truth.items()} for name,p in outputs.items()}
    atomic_json(folder/"scores.json",{"observations":len(ids),"rmse":scores,"issued_sha256":sha256(path.read_bytes()).hexdigest()})
    print("transfer",task,{k:round(v["pooled"],4) for k,v in scores.items()})


def compare(cohort, mixture=False):
    task = {"pronostia":"bearing","phm08":"engine"}.get(cohort,cohort)
    package, checksum, _ = official(task)
    fixed, mature = reference_folders(cohort,"mature")
    samples = read(fixed/"state.json")["samples"]
    fit_ids = {i for i,r in samples.items() if r["role"]=="fit"}
    fit = [r for r in package.train if r.sample_id in fit_ids]
    if len(fit)!=len(fit_ids):
        raise ValueError("参考中心用了官方训练清单外的测量")
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    destination = MIXTURE_OUT if mixture else OUT
    out = destination/cohort
    with LocalStore(out).exclusive(), TemporaryDirectory(prefix="input-alignment-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),PROFILES[task]) as trainer:
        parent = Path(tmp)/"parent.npz"
        parent.write_bytes(package.checkpoint)
        model = trainer.Model.load(parent)
        source = model.normalized(np.array([r.values for r in fit]))
        center, square_root = source.mean(axis=0), covariance_factor(source,.5)
        methods = ("identity","coral","matched","conditional") if mixture else METHODS
        if mixture:
            base_scores = model.predict(np.array([r.values for r in fit]))
            edges = None if task=="gas" else np.unique(np.quantile(base_scores,[.25,.5,.75]))
            source_groups = group_predictions(task,base_scores,edges)
        streams = {m:{} for m in methods}
        live = dict.fromkeys(methods)
        logs = defaultdict(list)
        for ordinal in range(10):
            roles = {role:sorted(i for i,r in samples.items() if r["role"]==role and r["available_at"]==ordinal)
                     for role in ("explore","adopt","serve")}
            if any(not ids for ids in roles.values()) or any(samples[i]["label"] is not None for i in roles["serve"]):
                raise ValueError("缺少分片或未移除服务标签")
            x = {role:np.array([samples[i]["values"] for i in ids]) for role,ids in roles.items()}
            answers = {i:samples[i]["label"] for role in ("explore","adopt") for i in roles[role]}
            grouped = defaultdict(list)
            for index,i in enumerate(roles["explore"]):
                grouped[samples[i]["unit_id"]].append(index)
            groups = list(grouped.values())
            if task=="gas":
                ranked = sorted(range(len(roles["explore"])), key=lambda j:digest(roles["explore"][j]))
                groups = [ranked[::2],ranked[1::2]]
            observed = model.normalized(x["explore"])
            for method in methods:
                start = perf_counter_ns()
                composition = {}
                if method in ("matched","conditional"):
                    target_groups = group_predictions(task,model.predict(x["explore"]),edges)
                    trial,composition = composition_trial(method,observed,target_groups,source,source_groups,center,square_root)
                else:
                    trial = None if method=="identity" else propose(method,center,square_root,observed,groups)
                event = {"round":ordinal+1,"exploration_groups":len(groups),"proposed":trial is not None,"adopted":False}
                if composition:
                    event["composition"] = composition
                if trial is not None:
                    mat, off = trial
                    event.update(nonzero_offset=int(np.count_nonzero(off)),
                                 transform_sha256=digest([None if mat is None else mat.tolist(),off.tolist()]))
                    for role in ("explore","adopt"):
                        values = [metric(task, answers, predict(task,model,t,roles[role],x[role])) for t in (live[method],trial)]
                        event[role] = values
                        gain = (values[1]-values[0])*(1 if task=="gas" else -1)
                        if gain<=1e-12:
                            break
                        if role=="adopt":
                            live[method] = trial
                            event["adopted"] = True
                pred = predict(task,model,live[method],roles["serve"],x["serve"])
                if pred.keys() & streams[method].keys():
                    raise ValueError("服务测量跨轮重复")
                streams[method].update(pred)
                event["elapsed_seconds"] = (perf_counter_ns()-start)/1e9
                logs[method].append(event)
        path = out/"issued.json"
        if mixture and (prior:=read(OUT/cohort/"issued.json"))["predictions"]["coral"]!=streams["coral"]:
            raise ValueError("普通 CORAL 对照与已交预测不一致")
        if path.exists() and read(path)["predictions"]!=streams:
            raise ValueError("重复运行未复现已封存的逐条结果")
        atomic_json(path,{"package_sha256":checksum,"parent_sha256":sha256(package.checkpoint).hexdigest(),
                          "fit_sha256":digest(sorted(fit_ids)),"predictions":streams,"history":dict(logs),
                          "terminal":{k:None if t is None else [None if t[0] is None else t[0].tolist(),t[1].tolist()]
                                      for k,t in live.items()}})
        original = BASE/"original"/task if task==cohort else fixed.parent/"original"
        old = BASE/"selected"/task if task==cohort else fixed.parent/"project"
        streams.update(parent=delivered(original),old=delivered(old),mature=delivered(mature))
        ids = sorted(streams["parent"])
        if streams["identity"].keys()!=streams["parent"].keys() or not np.allclose(
                [streams["identity"][i] for i in ids],[streams["parent"][i] for i in ids],rtol=1e-12,atol=1e-8):
            raise ValueError("恒等输入未复现官方起点预测")
        truth = {r.sample_id:r.label for r in package.validation}
        if cohort=="phm08":
            truth = remaining_cycles()
        if cohort=="pronostia":
            truth = remaining_minutes("full-test-manifest.json")
        stats = summary(task,truth,streams,{i:samples[i]["unit_id"] for i in ids})
        atomic_json(out/"scores.json",{"arms":stats,"issued_sha256":sha256(path.read_bytes()).hexdigest(),
                    "cost":{m:{"proposed":sum(e["proposed"] for e in events),"adopted":sum(e["adopted"] for e in events),
                               "elapsed_seconds":sum(e["elapsed_seconds"] for e in events)} for m,events in logs.items()}})
        print(cohort,{k:round(v["score"],5) for k,v in stats.items()})
        if cohort!=task:
            transfer(task,model,live,path,destination)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("gas","bearing","engine","pronostia","phm08"))
    parser.add_argument("--mixture",action="store_true")
    args=parser.parse_args()
    compare(args.cohort,args.mixture)
