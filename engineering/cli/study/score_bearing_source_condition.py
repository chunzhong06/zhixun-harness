"""只用源设备的已知工况学习振动判据，在单条预测中检验它能否补偿官方模型的寿命偏差。"""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile
import json

import numpy as np

from compare_group_objectives import ROOT, _trainer, official
from harness.contracts import digest
from harness.runtime.storage import atomic_json
from score_life_training_weights import cohorts
from score_native_bearing_choices import DEST as PREVIOUS, fingerprint, publish


DEST = PREVIOUS.parent / "bearing-source-condition"
PROTOCOL = ROOT / "engineering/configs/experiments/bearing-source-condition.json"


def source_settings(archive, pkg):
    labels = {}
    for line in archive.read("public/train.jsonl").splitlines():
        metadata = json.loads(line)["meta"]
        unit = metadata["bearing"]
        condition = Path(metadata["source"]).parts[-3]
        if labels.setdefault(unit, condition) != condition:
            raise ValueError("单台源设备的试验工况不唯一")
    if set(labels) != {r.unit_id for r in pkg.train} or sorted(Counter(labels.values()).values()) != [3, 3, 3]:
        raise ValueError("源设备和工况分组不同")
    names = sorted(set(labels.values()))
    return {u: names.index(c) for u, c in labels.items()}, names


def device_weights(units):
    distinct, inv, count = np.unique(units, return_inverse=True, return_counts=True)
    return 1. / (len(distinct) * count[inv])


def fit_gate(z, labels, w):
    centers = np.stack([np.average(z[labels == c], axis=0, weights=w[labels == c]) for c in range(3)])
    d = z - centers[labels]
    cov = d.T @ (w[:, None]*d)
    shrink = .5*cov + .5*np.diag(np.diag(cov)) + 1e-6*np.eye(z.shape[1])
    coef = np.linalg.solve(shrink, centers.T)
    return coef, -.5*np.sum(centers * coef.T, axis=1)


def probabilities(z, gate):
    coef, bias = gate
    scores = z @ coef + bias
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def held_device_accuracy(x, units, y):
    scores = {}
    for u in np.unique(units):
        fit = units != u
        center = x[fit].mean(axis=0)
        scale = x[fit].std(axis=0)
        z = np.clip((x-center)/np.where(scale>1e-12, scale, 1.), -10, 10)
        gate = fit_gate(z[fit], y[fit], device_weights(units[fit]))
        scores[str(u)] = float(np.mean(probabilities(z[~fit], gate).argmax(axis=1) == y[~fit]))
    return scores


def train(tag, rows, settings, trainer, parent_path):
    x = np.array([r.values for r in rows])
    y = np.array([r.label for r in rows])
    units = np.array([r.unit_id for r in rows])
    conditions = np.array([settings[u] for u in units])
    w = device_weights(units)
    base = trainer.Model.load(parent_path)
    z = base.normalized(x)
    hidden, _ = base.forward(z)
    slots = np.argsort(hidden.std(axis=0) * abs(base.params["w2"][:, 0]), kind="stable")[:3]
    started = perf_counter()
    gate = fit_gate(z, conditions, w)
    residual = y - base.predict(x)
    offset = np.array([np.average(residual[conditions == c], weights=w[conditions == c]) for c in range(3)])
    constant = np.average(residual, weights=w)
    # 在同一工况的三个重复试验中使用不同的循环位移，保留类别数但破坏真工况与读数的联系。
    shuffled = {u: (settings[u] + int(u.split("_")[-1]) - 1) % 3 for u in settings}
    sham = fit_gate(z, np.array([shuffled[u] for u in units]), w)
    models, identities = {}, {}
    folder = DEST / tag / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for name, basis in (("native", gate), ("native-sham", sham), ("native-old-slots", None)):
        model = trainer.Model.load(parent_path)
        if basis is not None:
            coef, bias = basis
            contrast = 1.5 * (coef - coef.mean(axis=1, keepdims=True))
            intercept = 1.5 * (bias - bias.mean())
            raw = z @ contrast + intercept
            center = np.average(raw, axis=0, weights=w)
            scale = np.sqrt(np.average((raw - center)**2, axis=0, weights=w))
            scale = np.maximum(scale, 1e-7)
            model.params["w1"][:, slots] = contrast / scale
            model.params["b1"][slots] = intercept / scale
        h, _ = model.forward(z)
        model.params["w2"][slots, 0] = 0.
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"] - (h @ model.params["w2"])[:,0]
        a = np.column_stack((h[:, slots], np.ones(len(y))))
        ridge = np.diag([.001, .001, .001, 0.])
        fitted = np.linalg.solve(a.T @ (w[:, None]*a) + ridge, a.T @ (w * target))
        model.params["w2"][slots, 0] = fitted[:-1]
        model.params["b2"][0] = fitted[-1]
        with TemporaryDirectory(dir=folder) as tmp:
            temp = Path(tmp) / (name + ".npz")
            model.save(temp)
            path = folder / temp.name
            temp.replace(path)
        identities[name] = fingerprint(path)
        models[name] = trainer.Model.load(path)
    fit_seconds = perf_counter() - started
    audits = held_device_accuracy(x, units, conditions)
    record = {"source_rows": len(rows), "source_devices": len(settings), "fit_digest": digest([r.sample_id for r in rows]),
              "weights": identities, "source_correct_by_held_device": audits, "source_equal_accuracy": float(np.mean(list(audits.values()))),
              "offset_minutes": offset.tolist(), "global_offset": float(constant), "slots": slots.tolist(), "fit_wall_s": fit_seconds}
    atomic_json(DEST / tag / "training.json", record)
    def infer(values):
        raw = np.array(values)
        p = probabilities(base.normalized(raw), gate)
        old = base.predict(raw)
        return {**{k: m.predict(raw) for k,m in models.items()}, "soft": np.maximum(old + p @ offset, 0),
                "hard": np.maximum(old + offset[p.argmax(axis=1)], 0), "global": np.maximum(old + constant, 0)}, p.argmax(axis=1)
    return infer, offset, record


def known_condition(unit, mapping):
    # 只给独立诊断用；推理函数的唯一参数是当前行的 34 个数值。
    family = unit.split("_")[0]
    return mapping[family]


def run():
    pkg, _, batches = official("bearing")
    fit_ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    reports = {}
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "B") as trainer:
        settings, condition_names = source_settings(z, pkg)
        mapping = {u.split("_")[0]: c for u,c in settings.items()}
        if any(mapping[u.split("_")[0]] != c for u,c in settings.items()):
            raise ValueError("审计用的 XJTU 工况和来源不符")
        parent = Path(tmp)/"parent.npz"
        parent.write_bytes(pkg.checkpoint)
        for tag, sample in (("small", [r for r in pkg.train if r.sample_id in fit_ids]), ("full", list(pkg.train))):
            infer, offset, record = train(tag, sample, settings, trainer, parent)
            report = {}
            for site, ids, x, groups, answers, _ in cohorts("bearing", pkg):
                previous = PREVIOUS/"pool"/site
                if not previous.exists():
                    continue
                old = json.loads((previous/"issued.json").read_bytes())
                if list(ids) != old["ids"] or fingerprint(previous/"issued.json") != json.loads((previous/"scores.json").read_bytes())["issued_sha256"]:
                    raise ValueError("当前测量与冻结单权重候选未对齐")
                p, predicted = infer(x)
                payload = {"ids":ids, "devices":[groups[k] for k in ids], "weights":record["weights"],
                           "protocol":fingerprint(PROTOCOL), "training":fingerprint(DEST/tag/"training.json"),
                           "predictions":{**old["predictions"], **{k:v.tolist() for k,v in p.items()}}}
                if site in ("bearing","roles","new-xjtu"):
                    actual = np.array([known_condition(u, mapping) for u in payload["devices"]])
                    payload["regime_prediction"] = predicted.tolist()
                    payload["regime_true_for_grading"] = actual.tolist()
                    payload["predictions"]["known-condition-diagnostic"] = np.maximum(np.asarray(old["predictions"]["root"])+offset[actual], 0).tolist()
                table = publish(DEST/tag/site, payload, answers)
                if "regime_true_for_grading" in payload:
                    table["regime_accuracy"] = float(np.mean(actual == predicted))
                report[site] = table
            reports[tag] = {"training": record, "sites": report}
    atomic_json(DEST/"scores.json", {"protocol":fingerprint(PROTOCOL), "condition_names":condition_names,"runs":reports})
    for tag, r in reports.items():
        print(tag,round(r["training"]["source_equal_accuracy"],3),{k:[round(s["scores"][arm]["rmse"],2) for arm in
              ("current","native-old-slots","native","soft","global")] for k,s in r["sites"].items()},flush=True)


if __name__ == "__main__":
    run()
