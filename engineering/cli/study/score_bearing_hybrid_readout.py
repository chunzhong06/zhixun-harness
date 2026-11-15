"""只由源设备留出选择混合教师目标，再用官方单权重交付接口核查。

公开服务和既有第五台设备均为开发评价，答案仅在权重冻结、预测封存后计分。
每个留出折重训教师并重选隐藏表示，避免用该折答案构造教师。
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.data.official_bearing import read_official_bearing
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import LocalStore, atomic_json

PROTOCOL = ROOT / "engineering/configs/experiments/bearing-hybrid-readout.json"
ARCHIVE = ROOT / "data/official/bearing/cpu-b-rul-20261110-v1.zip"
PRIOR = ROOT / "results/experiments/local-defaults/whole-maintenance/bearing-native-choices/pool"
DEST = PRIOR.parent.parent / "bearing-hybrid-readout"
MIXES = (0., .5, 1.)


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def borders(model, x, target, tree, count):
    """用训练侧残差逐次选树切分；不向选择器传入设备身份或评价答案。"""
    features, cuts = [], []
    for axis, values in tree.get_borders().items():
        for value in values:
            cut = (value - model.mean[axis]) / model.scale[axis]
            size = int(np.sum(x[:, axis] <= cut))
            if min(size, len(x) - size) >= .01 * len(x) and -10 < cut < 10:
                features.append(axis)
                cuts.append(cut)
    if len(cuts) < count:
        raise ValueError("有效教师边界不足以填入原隐藏宽度")
    masks = x[:, np.array(features)].T <= np.array(cuts)[:, None]
    left = masks.sum(axis=1)
    residual = target - target.mean()
    used = set()
    chosen = []
    for _ in range(count):
        sums = masks @ residual
        right = residual.sum() - sums
        gains = sums**2 / left + right**2 / (len(x) - left)
        if used:
            gains[list(used)] = -np.inf
        j = int(gains.argmax())
        chosen.append((features[j], float(cuts[j])))
        used.add(j)
        residual -= np.where(masks[j], sums[j]/left[j], right[j]/(len(x)-left[j]))
    return chosen


def fit_models(rows, trainer, parent, folder):
    x = np.array([r.values for r in rows])
    y = np.array([r.label for r in rows])
    units, inverse, counts = np.unique([r.unit_id for r in rows], return_inverse=True, return_counts=True)
    weight = len(rows) / (len(units) * counts[inverse])
    small = []
    for u in units:
        ix = np.flatnonzero(np.array([r.unit_id for r in rows]) == u)
        small.extend(ix[np.linspace(0, len(ix)-1, 8).round().astype(int)])
    total = np.array([r.label + r.time_end - 1 for r in rows])
    scale = float(np.median([total[inverse == j][0] for j in range(len(units))]))
    teachers = []
    for target in (y, y / total):
        teacher = CatBoostRegressor(iterations=100, depth=6, random_seed=17, thread_count=1,
                                    verbose=False, allow_writing_files=False)
        teacher.fit(x[small], target[small])
        teachers.append(teacher)
    targets = (teachers[0].predict(x, thread_count=1),
               scale * np.clip(teachers[1].predict(x, thread_count=1), 0, 1))
    base = trainer.Model.load(parent)
    z = base.normalized(x)
    width = base.params["w1"].shape[1]
    chosen = borders(base, z, targets[0], teachers[0], width//2)
    chosen += borders(base, z, targets[1], teachers[1], width-width//2)
    models = {}
    for mix in MIXES:
        model = trainer.Model.load(parent)
        for slot, (axis, cut) in enumerate(chosen):
            model.params["w1"][:, slot] = 0
            model.params["w1"][axis, slot] = 4.
            model.params["b1"][slot] = -4 * cut
        hidden, _ = model.forward(z)
        design = np.column_stack((hidden, np.ones(len(rows))))
        target = ((1-mix)*targets[0] + mix*targets[1] - model.meta["target_mean"]) / model.meta["target_scale"]
        gram = design.T @ (weight[:, None]*design) + .001*len(rows)*np.eye(design.shape[1])
        fit = np.linalg.solve(gram, design.T @ (weight*target))
        model.params["w2"][:, 0] = fit[:-1]
        model.params["b2"][0] = fit[-1]
        # 岭回归不是 Adam 步；保留原步数和优化器历史，并在外部训练回执记实际求解。
        path = folder / f"mix-{mix:g}.npz"
        model.save(path)
        models[mix] = trainer.Model.load(path)
        np.testing.assert_allclose(models[mix].predict(x), model.predict(x), rtol=0, atol=1e-9)
    return models, {"devices": units.tolist(), "rows":len(rows), "source_scale_minutes":scale}


def statistics(predictions, truth, devices):
    y = np.asarray(truth)
    owners = np.asarray(devices)
    return {name: {"rmse":float(np.sqrt(np.mean((np.asarray(p)-y)**2))),
                   "by_device":{str(u):float(np.sqrt(np.mean((np.asarray(p)[owners==u]-y[owners==u])**2)))
                                for u in np.unique(owners)}} for name,p in predictions.items()}


def evaluate(pkg, models, selected, out, original):
    fifth = read(ROOT / "data/public/xjtu-sy/fifth-devices.json")
    public = read(PRIOR / "bearing/issued.json")
    service = set(public["ids"])
    cases = {"public": [r for r in pkg.validation if r.sample_id in service], "official_full":list(pkg.validation)}
    staged = {}
    for name, rows in cases.items():
        x = np.array([r.values for r in rows])
        staged[name] = {"ids":[r.sample_id for r in rows], "devices":[r.unit_id for r in rows],
                        "predictions":{**{f"mix-{k:g}":m.predict(x).tolist() for k,m in models.items()},
                                       "recomputed-root":original.predict(x).tolist()}}
    rows = fifth["rows"]
    staged["xjtu_fifth"] = {"ids":[r["member"] for r in rows], "devices":[Path(r["member"]).parent.name for r in rows],
                            "predictions":{f"mix-{k:g}":m.predict(np.array([r["features"] for r in rows])).tolist() for k,m in models.items()}}
    staged["xjtu_fifth"]["predictions"]["recomputed-root"] = original.predict(np.array([r["features"] for r in rows])).tolist()
    # 先封存全体预测；该边界以后才将评价标签与预测配对。
    issued = out / "issued.json"
    atomic_json(issued, {"source_selected":selected, "cases":staged})
    full_truth = {r.sample_id:r.label for r in pkg.validation}
    result = {}
    for name, payload in read(issued)["cases"].items():
        source_name = {"public":"bearing", "official_full":"roles", "xjtu_fifth":"new-xjtu"}[name]
        source = PRIOR/source_name/"issued.json"
        if fingerprint(source) != read(source.parent/"scores.json")["issued_sha256"]:
            raise ValueError("固定参照文件发生变化")
        prior = read(source)
        paired = {}
        for j, key in enumerate(prior["ids"]):
            sid = key.split("/")[-1] if name == "official_full" else key
            values = {k:v[j] for k,v in prior["predictions"].items() if k in ("root","median","hidden","phase","current")}
            if sid in paired:
                # 同一测量在不同批大小下会有浮点求和末位差；仍逐臂检查实际分钟差。
                np.testing.assert_allclose(list(paired[sid].values()),list(values.values()),rtol=0,atol=1e-7)
            paired[sid] = values
        for arm in ("root","median","hidden","phase","current"):
            payload["predictions"][arm] = [paired[key][arm] for key in payload["ids"]]
        np.testing.assert_allclose(payload["predictions"]["recomputed-root"],payload["predictions"]["root"],rtol=0,atol=1e-7)
        if name == "xjtu_fifth":
            truth = [fifth["ends"][u]-int(Path(key).stem) for key,u in zip(payload["ids"],payload["devices"],strict=True)]
        else:
            truth = [full_truth[key] for key in payload["ids"]]
        result[name] = {"rows":len(truth), "arms":statistics(payload["predictions"],truth,payload["devices"])}
    atomic_json(out/"scores.json", {"issued_sha256":fingerprint(issued), "source_selected":selected, "cases":result})
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--evaluate-existing",action="store_true",help="仅复用完整训练回执与冻结权重，不重复拟合")
    parser.add_argument("--deliver",action="store_true",help="将源端选出的权重通过原逐批交接入口实际发放")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.evaluate_existing:
        with LocalStore(out/"lock").exclusive():
            record = read(out/"training.json")
            if record["protocol_sha256"] != fingerprint(PROTOCOL) or record["package_sha256"] != fingerprint(ARCHIVE):
                raise ValueError("恢复的协议或官方训练包不符")
            pkg = read_official_bearing(ARCHIVE)
            with TemporaryDirectory(dir=out) as tmp,ZipFile(ARCHIVE) as archive,_trainer(archive,Path(tmp),"B") as trainer:
                parent = Path(tmp)/"original.npz"
                parent.write_bytes(pkg.checkpoint)
                models = {}
                for k in MIXES:
                    path = out/"models"/f"mix-{k:g}.npz"
                    if fingerprint(path) != record["weights"][f"mix-{k:g}"]:
                        raise ValueError("恢复的训练权重发生变化")
                    models[k] = trainer.Model.load(path)
                result = evaluate(pkg,models,record["source_selected"],out,trainer.Model.load(parent))
                print(json.dumps({"selected":record["source_selected"],"evaluation":{n:{a:round(v["rmse"],3) for a,v in r["arms"].items()} for n,r in result.items()}}))
            if args.deliver:
                from engineering.cli.workflow.serve_life_checkpoint import run
                selected = out/"models"/(record["source_selected"]+".npz")
                source = ROOT/"results/experiments/local-defaults/selected/bearing"
                delivery = out/"delivery"
                report = run("bearing",ARCHIVE,selected,source,delivery,10)
                issued = {}
                for event in report["accepted"]:
                    issued.update(read(delivery/"accepted"/(event["request"]+".json"))["payload"]["predictions"])
                raw = read(out/"issued.json")["cases"]["public"]
                expected = dict(zip(raw["ids"],raw["predictions"][record["source_selected"]],strict=True))
                if issued.keys() != expected.keys():
                    raise ValueError("实际交付的样本与冻结预测不一致")
                np.testing.assert_allclose([issued[k] for k in expected],list(expected.values()),rtol=0,atol=1e-7)
                truth = {r.sample_id:r.label for r in pkg.validation}
                rmse = float(np.sqrt(np.mean([(issued[k]-truth[k])**2 for k in issued])))
                atomic_json(delivery/"scores.json",{"weights_sha256":fingerprint(selected),"rows":len(issued),"rmse":rmse,
                                                      "matches_frozen_predictions":True,"new_training":False})
                print(json.dumps({"actual_delivery_rmse":rmse,"rows":len(issued),"selected":record["source_selected"]}))
        return
    if args.deliver:
        parser.error("先完成并冻结源端训练，再用 --evaluate-existing --deliver 实交")
    with LocalStore(out/"lock").exclusive():
        if (out/"scores.json").exists():
            raise ValueError("已有完整实验；请复用结果，避免覆盖或重复训练")
        started = perf_counter()
        pkg = read_official_bearing(ARCHIVE)
        units = sorted({r.unit_id for r in pkg.train})
        predictions, labels, folds = {k:[] for k in MIXES}, [], []
        with TemporaryDirectory(dir=out) as tmp, ZipFile(ARCHIVE) as archive, _trainer(archive,Path(tmp),"B") as trainer:
            parent = Path(tmp)/"original.npz"
            parent.write_bytes(pkg.checkpoint)
            for unit in units:
                fit = [r for r in pkg.train if r.unit_id != unit]
                held = [r for r in pkg.train if r.unit_id == unit]
                # 原包按独占创建保存权重；各留出折使用自己的临时目录。
                fold_folder = Path(tmp)/unit
                fold_folder.mkdir()
                models, record = fit_models(fit,trainer,parent,fold_folder)
                x = np.array([r.values for r in held])
                for k,m in models.items():
                    predictions[k].extend(m.predict(x).tolist())
                labels.extend(r.label for r in held)
                folds.append({"held_device":unit,"training":record})
            scores = {k:float(np.sqrt(np.mean((np.array(p)-np.array(labels))**2))) for k,p in predictions.items()}
            selected = min(MIXES,key=scores.get)
            folder = out/"models"
            folder.mkdir(parents=True,exist_ok=True)
            models, record = fit_models(list(pkg.train),trainer,parent,folder)
            identity = {"protocol_sha256":fingerprint(PROTOCOL), "package_sha256":fingerprint(ARCHIVE),
                        "source_selected":f"mix-{selected:g}", "source_cv_rmse":scores, "folds":folds,
                        "final":record,"weights":{f"mix-{k:g}":fingerprint(folder/f"mix-{k:g}.npz") for k in models}}
            atomic_json(out/"training.json",identity)
            results = evaluate(pkg,models,identity["source_selected"],out,trainer.Model.load(parent))
        atomic_json(out/"timing.json",{"wall_seconds":perf_counter()-started,"teacher_fits":20,"ridge_solves":30})
        print(json.dumps({"selected":identity["source_selected"], "source_cv_rmse":scores,
                          "evaluation":{n:{a:round(v["rmse"],3) for a,v in r["arms"].items()} for n,r in results.items()}},ensure_ascii=False))


if __name__ == "__main__":
    main()
