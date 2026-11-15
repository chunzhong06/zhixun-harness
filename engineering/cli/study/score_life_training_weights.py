"""配对比较同训练资料、同树数的按行或按设备加权寿命预测。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from sklearn.base import clone

from compare_fixed_tree_references import OUT, frozen, paths, tree_values
from compare_group_objectives import ROOT, atomic_json, digest, official, read, remaining_cycles, remaining_minutes, summary
from harness.models.training_tools import SklearnBackend
from score_tree_reference_transfer import suites


DEST = OUT / "device-weights"


def weighted_median(y, weights):
    order = np.argsort(y, kind="stable")
    sorted_values = y[order]
    cumulative = np.cumsum(weights[order])
    half = cumulative[-1] / 2
    pos = np.searchsorted(cumulative, half - 1e-10)
    if abs(cumulative[pos] - half) < 1e-10 and pos + 1 < len(y):
        return float((sorted_values[pos] + sorted_values[pos + 1]) / 2)
    return float(sorted_values[pos])


def train(task, pkg, package_hash, sampled_ids):
    models, constants, receipts = {}, {}, {}
    old, original = frozen(task)
    template = old["catboost"]
    backend = SklearnBackend(execution_enabled=True)
    output = DEST / task / "models"
    output.mkdir(parents=True, exist_ok=True)
    for pool in ("small", "full"):
        rows = [r for r in pkg.train if pool == "full" or r.sample_id in sampled_ids]
        if any(r.role != "fit" for r in rows):
            raise ValueError("训练候选池包含非训练行")
        x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows], dtype=float)
        units, inverse, counts = np.unique([r.unit_id for r in rows], return_inverse=True, return_counts=True)
        row_ids = digest([r.sample_id for r in rows])
        if pool == "small" and (row_ids != original["catboost"]["fit"] or np.any(counts != 8)):
            raise ValueError("小池不同于已封存的每台八条原始参照")
        for rule in ("row", "device"):
            name = f"{pool}-{rule}"
            weights = np.ones(len(rows)) if rule == "row" else len(rows) / (len(units) * counts[inverse])
            if not np.isclose(weights.sum(), len(rows)):
                raise ValueError("权重总量发生变化")
            constants[name + "-mean"] = float(np.average(y, weights=weights))
            constants[name + "-median"] = weighted_median(y, weights)
            pipeline = clone(template)
            start, cpu_start = perf_counter(), process_time()
            pipeline.fit(x, y, predictor__sample_weight=weights)
            elapsed, cpu = perf_counter() - start, process_time() - cpu_start
            data = backend.dump(pipeline)
            path = output / (name + ".joblib")
            path.write_bytes(data)
            models[name] = backend.load(path.read_bytes())
            if pool == "small" and np.max(abs(tree_values(template,task,x)-tree_values(models[name],task,x))) > 1e-8:
                raise ValueError("小池重新拟合的结果与原树不同")
            receipts[name] = {"package":package_hash, "weights":sha256(data).hexdigest(), "fit_ids":row_ids,
                "rows":len(rows), "units":len(units), "shortest_track":int(counts.min()), "longest_track":int(counts.max()),
                "row_weight_range":[float(weights.min()),float(weights.max())], "cpu_seconds":cpu,"wall_seconds":elapsed,
                "recipe":original["catboost"]["recipe"], "seed":template.named_steps["predictor"].get_params()["random_seed"]}
    atomic_json(output / "training.json", {"arms":receipts,"constants":constants})
    return models, constants


def cohorts(task,pkg):
    """答案函数仅在当前服务预测全部封存后调用。"""
    base=OUT/"teacher-readout"/task
    public_answers = {r.sample_id:r.label for r in pkg.validation}
    for name in (task, "phm08" if task=="engine" else "pronostia"):
        folder,_=paths(name,task)
        samples=read(folder/"state.json")["samples"]
        ids=[k for k,v in samples.items() if v["role"]=="serve"]
        if any(samples[k]["label"] is not None for k in ids):
            raise ValueError("服务输入携带答案")
        answers=(lambda:public_answers) if name==task else (remaining_cycles if task=="engine" else lambda:remaining_minutes("full-test-manifest.json"))
        yield name,ids,[samples[k]["values"] for k in ids],{k:samples[k]["unit_id"] for k in ids},answers,base/name
    for name,windows,groups,answers in suites(task,False):
        keys,x=zip(*(r for _,b in sorted(windows.items()) for r in b),strict=True)
        yield name,keys,x,groups,answers,base/name
    if task=="bearing":
        rows={r.sample_id:r for r in pkg.validation}
        source=read(base/"roles/issued.json")["predictions"]["current"]
        by_id={k:k.split("/")[-1] for k in source}
        yield "roles",list(source),[rows[i].values for i in by_id.values()],{k:rows[i].unit_id for k,i in by_id.items()},lambda by_id=by_id:{k:public_answers[i] for k,i in by_id.items()},base/"roles"
        source=read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
        keys=[r["member"] for r in source["rows"]]
        groups={k:Path(k).parent.name for k in keys}
        yield "new-xjtu",keys,[r["features"] for r in source["rows"]],groups,lambda:{k:source["ends"][unit]-int(Path(k).stem) for k,unit in groups.items()},base/"new-xjtu"


def score(task, cohort, forecast, destination=DEST, anchor=None):
    name,keys,x,groups,answers,base=cohort
    sealed=base/"issued.json"
    if sha256(sealed.read_bytes()).hexdigest()!=read(base/"scores.json")["issued_sha256"]:
        raise ValueError("历史参照与评分记录不符")
    old=read(sealed)["predictions"]
    p={k:old[k] for k in ("current","paper","fixed-tree","tree","teacher-ridge") if k in old}
    predicted=forecast(np.asarray(x))
    for k,values in predicted.items():
        p[k]=dict(zip(keys,values.tolist(),strict=True))
    old_tree=p.get("fixed-tree",p.get("tree"))
    tree_arm="small-row" if anchor is None else anchor
    if max(abs(old_tree[i]-p[tree_arm][i]) for i in keys)>1e-8:
        raise ValueError("输入或小池参照未复现")
    out=destination/task/name
    path=out/"issued.json"
    atomic_json(path,{"source_sha256":sha256(sealed.read_bytes()).hexdigest(),"predictions":p})
    stored=read(path)["predictions"]
    table=summary(task,answers(),{"parent":stored["current"],**stored},groups)
    for v in table.values():
        v["device_equal"]=float(np.sqrt(np.mean(np.square(list(v["group_scores"].values())))))
    atomic_json(out/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
    return {k:{metric:v[metric] for metric in ("score","device_equal","worst_group")} for k,v in table.items()}


def study(task):
    pkg,package_hash,batches=official(task)
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    models,constants=train(task,pkg,package_hash,ids)
    def predict(x):
        fixed={k:np.full(len(x),v) for k,v in constants.items()}
        return {**fixed,**{k:tree_values(model,task,x) for k,model in models.items()}}
    results={c[0]:score(task,c,predict) for c in cohorts(task,pkg)}
    atomic_json(DEST/task/"scores.json",results)
    for name,t in results.items():
        print(name,{arm:round(t[arm]["score"],2) for arm in ("current","small-row","full-row","full-device","full-device-mean","full-device-median")})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("engine","bearing"))
    study(parser.parse_args().task)
