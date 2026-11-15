"""用相同树和训练行比较幅值信息与对传感器增益不敏感的振动形状。"""
from hashlib import sha256
from time import process_time

import numpy as np
from sklearn.base import clone

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import atomic_json, digest, official
from harness.data.official_bearing import MEASURES
from harness.models.training_tools import SklearnBackend
from score_life_training_weights import cohorts, score


def views(values):
    blocks = np.asarray(values).reshape(-1, 2, len(MEASURES))
    bands = [i for i, k in enumerate(MEASURES) if k.startswith("band")]
    scaled = [MEASURES.index(k) for k in ("rms", "mean_abs", "peak_to_peak", "peak", "spectral_rms")]
    stable = [i for i in range(len(MEASURES)) if i not in scaled + bands]
    energy = blocks[:, :, bands]
    mass = energy.sum(axis=2, keepdims=True)
    proportions = np.divide(energy, mass, out=np.zeros_like(energy), where=mass>0)
    shape = np.concatenate((blocks[:, :, stable], proportions), axis=2).reshape(len(blocks), -1)
    amplitude = blocks[:, :, scaled + bands].reshape(len(blocks), -1)
    return {"shape": shape, "amplitude": amplitude}


def main():
    pkg, _, batches = official("bearing")
    small = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    rows = [r for r in pkg.train if r.sample_id in small]
    x, y = np.array([r.values for r in rows]), np.array([r.label for r in rows])
    if len(rows) != 72 or any(r.role != "fit" for r in rows):
        raise ValueError("源训练行身份不符")
    root = OUT / "gain-features"
    models, receipts = frozen("bearing")
    full = models["catboost"]
    if digest([r.sample_id for r in rows]) != receipts["catboost"]["fit"]:
        raise ValueError("各臂训练行不同")
    backend = SklearnBackend(execution_enabled=True)
    folder = root / "bearing/models"
    folder.mkdir(parents=True, exist_ok=True)
    gains = [4 if name.startswith("band") or name == "spectral_rms" else
             2 if name in ("rms", "mean_abs", "peak_to_peak", "peak") else 1 for name in MEASURES]
    doubled = x * np.tile(gains, 2)
    if not np.allclose(views(x)["shape"], views(doubled)["shape"], rtol=0, atol=1e-12):
        raise ValueError("正幅值变换后形状表示变化")
    arms, fitting = {}, {}
    for name, v in views(x).items():
        estimator = clone(full)
        start = process_time()
        estimator.fit(v, y)
        cpu = process_time() - start
        blob = backend.dump(estimator)
        path = folder / (name + ".joblib")
        path.write_bytes(blob)
        arms[name] = backend.load(path.read_bytes())
        fitting[name] = {"cpu_seconds": cpu, "features": v.shape[1], "weights": sha256(blob).hexdigest()}
    atomic_json(folder / "training.json", {"source_sha256": pkg.train_sha, "fit_ids": receipts["catboost"]["fit"], "arms": fitting})
    def forecast(values):
        inputs = views(values)
        pred = {k: tree_values(m, "bearing", inputs[k]) for k, m in arms.items()}
        return {**pred, "half": .5*(pred["shape"]+pred["amplitude"]), "full": tree_values(full, "bearing", values),
                "median": np.full(len(values), np.median(y)), "mean": np.full(len(values), np.mean(y))}
    result = {c[0]: score("bearing", c, forecast, root, "full") for c in cohorts("bearing", pkg)}
    atomic_json(root / "bearing/scores.json", result)
    print({n: {k: round(t[k]["score"], 2) for k in ("full", "shape", "amplitude", "half", "median")} for n, t in result.items()})


if __name__ == "__main__":
    main()
