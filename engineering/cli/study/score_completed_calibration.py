"""仅在整台历史设备已失效时拟合现场标尺，再交付给设备身份完全分开的服务台。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np
from sklearn.base import clone

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, official, read, remaining_minutes, summary
from compare_public_selection import match_bearing_window
from score_life_training_weights import cohorts
from unseen_sources import bearing_data

DEST = OUT / "completed-calibration"


def five(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.unit_id].append(row)
    selected = []
    for unit, history in sorted(grouped.items()):
        history.sort(key=lambda r: r.time_end)
        if len(history) < 5:
            raise ValueError(f"已完成设备 {unit} 的可见历史过短")
        selected.extend(history[i] for i in np.linspace(0, len(history)-1, 5).round().astype(int))
    return selected


def ridge(x, y):
    xm, ym = x.mean(axis=0), y.mean()
    xc = x-xm
    coef = np.linalg.solve(xc.T @ xc + .1*len(x)*np.eye(x.shape[1]), xc.T @ (y-ym))
    return coef, float(ym-xm@coef)


def train(rows, source_tree, trainer, parent, model_dir):
    start = process_time()
    x, y = np.array([r.values for r in rows]), np.array([r.label for r in rows])
    tree = clone(source_tree)
    tree.fit(x, y)
    model = trainer.Model.load(parent)
    z = model.normalized(x)
    coef, bias = ridge(z, y)
    h = model.forward(z)[0]
    hc, hb = ridge(h, y)
    local_mean, local_median = float(y.mean()), float(np.median(y))
    shift = float((y - tree_values(source_tree, "bearing", x)).mean())
    params = {"local-mean": (np.zeros(h.shape[1]), local_mean),
              "local-median": (np.zeros(h.shape[1]), local_median), "hidden-ridge": (hc, hb)}
    models, hashes = {}, {}
    for name, (weights, intercept) in params.items():
        native = trainer.Model.load(parent)
        native.params["w2"][:,0] = weights/native.meta["target_scale"]
        native.params["b2"][0] = (intercept-native.meta["target_mean"])/native.meta["target_scale"]
        path = model_dir / (name+".npz")
        with TemporaryDirectory(dir=model_dir) as tmp:
            candidate = Path(tmp)/path.name
            native.save(candidate)
            candidate.replace(path)
        models[name] = trainer.Model.load(path)
        if not np.allclose(models[name].predict(x), np.maximum(h@weights+intercept,0),rtol=0,atol=1e-8):
            raise ValueError("单权重重载后现场读数不符")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    def predict(values):
        a = np.asarray(values)
        return {**{k: v.predict(a) for k,v in models.items()},
                "local-tree": tree_values(tree,"bearing",a), "source-tree": tree_values(source_tree,"bearing",a),
                "tree-bias": np.maximum(tree_values(source_tree,"bearing",a)+shift,0),
                "raw-ridge": np.maximum(model.normalized(a)@coef+bias,0)}
    receipt = {"rows": [r.sample_id for r in rows], "devices": sorted({r.unit_id for r in rows}),
               "fit_hash": digest([(r.sample_id,r.values,r.label) for r in rows]), "cpu_seconds": process_time()-start,
               "models": hashes, "mean": local_mean, "median": local_median, "tree_bias": shift}
    return predict, receipt


def main():
    from dataclasses import replace
    pkg, _, batches = official("bearing")
    source_tree = frozen("bearing")[0]["catboost"]
    source_median = float(np.median([r.label for r in batches[0].samples if r.role=="fit"]))
    external, _, _ = bearing_data()
    minute = remaining_minutes()
    femto = five([replace(match_bearing_window(r),label=minute[r.sample_id]) for records in external.values() for r in records])
    banks = {"femto": femto, "xjtu": five(pkg.train)}
    allowed = {"femto": ("pronostia","new-xjtu","new24","new40"), "xjtu": ("bearing","roles","new-xjtu")}
    data = {c[0]:c for c in cohorts("bearing",pkg) if c[0] in set(allowed["femto"]+allowed["xjtu"])}
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    totals = {}
    with TemporaryDirectory() as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "B") as trainer:
        parent = Path(tmp)/"parent.npz"
        parent.write_bytes(pkg.checkpoint)
        for bank, history in banks.items():
            units = sorted({r.unit_id for r in history})
            totals[bank] = {}
            for drop in [None,*units]:
                tag = "all" if drop is None else "without-"+drop.split("/")[-1]
                output = DEST/bank/tag
                folder = output/"models"
                folder.mkdir(parents=True,exist_ok=True)
                train_set = [r for r in history if r.unit_id!=drop]
                predict, receipt = train(train_set,source_tree,trainer,parent,folder)
                atomic_json(folder/"training.json",receipt)
                totals[bank][tag] = {}
                for name in allowed[bank]:
                    _, ids, x, groups, get_answers, controls = data[name]
                    same_site = (bank=="femto") == (name in ("pronostia","new24","new40"))
                    if same_site and ({r.unit_id.split("/")[-1] for r in train_set} &
                                      {u.split("/")[-1] for u in groups.values()}):
                        raise ValueError("现场训练包含服务设备")
                    src = controls/"issued.json"
                    if sha256(src.read_bytes()).hexdigest()!=read(controls/"scores.json")["issued_sha256"]:
                        raise ValueError("源预测与既有封存身份不同")
                    old = read(src)["predictions"]
                    p = {k:old[k] for k in ("current","paper") if k in old}
                    for k,v in predict(x).items():
                        p[k] = dict(zip(ids,v.tolist(),strict=True))
                    old_tree = old.get("tree",old.get("fixed-tree"))
                    if any(abs(v-old_tree[i])>1e-8 for i,v in p["source-tree"].items()):
                        raise ValueError("同一固定树在该批输入未复现")
                    p["source-median"] = dict.fromkeys(ids,source_median)
                    path = output/name/"issued.json"
                    atomic_json(path,{"predictions":p,"fit_hash":receipt["fit_hash"]})
                    served = read(path)["predictions"]
                    table = summary("bearing",get_answers(),{"parent":served["current"],**served},groups)
                    atomic_json(path.with_name("scores.json"),{"arms":table,"issued_sha256":sha256(path.read_bytes()).hexdigest()})
                    totals[bank][tag][name] = {k:[v["score"],v["worst_group"]] for k,v in table.items()}
                    if drop is None:
                        print(bank,name,{k:round(v["score"],2) for k,v in table.items()},flush=True)
    atomic_json(DEST/"scores.json",totals)


if __name__=="__main__":
    main()
