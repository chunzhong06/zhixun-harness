"""在相同原始波形和权重上隔离振动快照长度，完整长窗只用于机制对照。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, frozen, paths, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, official, read
from harness.data.official_bearing import MEASURES
from harness.models.model_artifacts import ModelArtifacts
from unseen_sources import channel


def features(segment):
    return np.asarray(channel(segment[:, 0]) + channel(segment[:, 1]))


def main():
    pkg, _, batches = official("bearing")
    fit = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    median = float(np.median([r.label for r in pkg.train if r.sample_id in fit]))
    tree = frozen("bearing")[0]["catboost"]
    current, _ = paths("bearing", "bearing")
    published = read(current / "run.json")["rounds"][-1]
    if published["selected"] != read(current / published["submitted"])["payload"]["model_id"]:
        raise ValueError("源运行末版未获采用")
    receipt, blob = ModelArtifacts(current / "models").read(published["model_artifact"])
    archive, = (ROOT / "data/official/bearing").glob("*.zip")
    long_rows = {(r.unit_id, r.time_end): r.values for r in pkg.validation}
    extra = read(ROOT / "data/public/xjtu-sy/fifth-devices.json")
    for r in extra["rows"]:
        p = Path(r["member"])
        long_rows[p.parent.name, int(p.stem)] = r["features"]
    directory = OUT / "snapshot-length"
    spectral = np.array([k.startswith(("spectral_", "band", "dominant_")) for k in list(MEASURES)*2])
    rms = np.array([k == "spectral_rms" for k in list(MEASURES)*2])
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "B") as trainer:
        file = Path(tmp) / "issued.npz"
        file.write_bytes(blob)
        model = trainer.Model.load(file)
        for unit in ("Bearing1_4", "Bearing1_5"):
            source, = (ROOT / "data/public/xjtu-sy/raw").glob("*/*/" + unit)
            files = sorted(source.glob("*.csv"), key=lambda p: int(p.stem))
            end = max(r.time_end for r in pkg.validation if r.unit_id == unit) if unit == "Bearing1_4" else extra["ends"][unit]
            if not files or int(files[-1].stem) > end:
                raise ValueError("原文件测量时刻和源终点不符")
            matrix, checks, ids, hashes = {}, 0, [], []
            for p in files:
                raw = np.loadtxt(p, delimiter=",", skiprows=1)
                if raw.shape != (32768, 2) or not np.isfinite(raw).all():
                    raise ValueError("长窗原始数据尺寸不符")
                long = features(raw)
                id_ = int(p.stem)
                if (unit, id_) in long_rows:
                    if not np.allclose(long, long_rows[unit, id_], rtol=2e-6, atol=1e-9):
                        raise ValueError("重算长窗的特征与官方/已核对新设备不符")
                    checks += 1
                ids.append(id_)
                hashes.append(sha256(p.read_bytes()).hexdigest())
                variants = {"long": long}
                for name, start in (("start", 0), ("center", (32768-2560)//2), ("end", 32768-2560)):
                    values = features(raw[start:start+2560])
                    if name == "start":
                        variants["start-raw"] = values.copy()
                    values[rms] *= np.sqrt(1281 / 16385)
                    variants[name] = values
                variants["short-time"] = np.where(spectral, long, variants["start"])
                variants["short-spectrum"] = np.where(spectral, variants["start"], long)
                for name, vector in variants.items():
                    matrix.setdefault(name, []).append(vector)
            output = {}
            for k, predict in (("current", model.predict), ("tree", lambda v: tree_values(tree, "bearing", v))):
                table = {n: predict(np.asarray(m)) for n, m in matrix.items()}
                table["three"] = np.mean([table[n] for n in ("start", "center", "end")], axis=0)
                output[k] = {n: v.tolist() for n, v in table.items()}
            sealed = directory / unit / "issued.json"
            atomic_json(sealed, {"times": ids, "sha256_sources": hashes, "industrial_model": receipt["sha256"],
                                  "predictions": output, "constant": median})
            served = read(sealed)
            truth = end - np.asarray(served["times"])
            table = {k: {n: {"rmse": float(np.sqrt(np.mean((np.asarray(v)-truth)**2))),
                              "mean_abs_change": float(np.mean(np.abs(np.asarray(v)-t["long"])))}
                         for n, v in t.items()} for k, t in served["predictions"].items()}
            table["median"] = float(np.sqrt(np.mean((truth-served["constant"])**2)))
            atomic_json(sealed.with_name("scores.json"), {"rows": len(ids), "long_checked": checks, "terminal": end,
                        "arms": table, "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})
            print(unit, len(ids), {k: round(v["rmse"], 2) for k, v in table["current"].items()})


if __name__ == "__main__":
    main()
