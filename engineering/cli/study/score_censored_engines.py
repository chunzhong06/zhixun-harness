"""在真实失效时刻未公开的截断轨迹上，只计算可从持续运行记录识别的评价量。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import frozen, paths, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, official, read, summary
from harness.models.engine_setting_reference import EngineSettingReference
from harness.models.source_combinations import SourceCombinations
from harness.models.model_artifacts import ModelArtifacts
from score_engine_setting_reference import DEST
from score_life_training_weights import cohorts
from unseen_sources import PHM08_TRAIN_SHA, track_rows


FILES = {"test.txt": "0109aa91930961e86f2e972ebb2887896281c18192db70c3ce77185b080e3850",
         "final_test.txt": "8f6a857757a0bc0a2d04de0239b40e74fab72e313ade00342669c7364b475a7c"}


def signatures(v):
    # 第三项设置在两个档案可能采用不同取值。去掉后再检验相同测量。
    values = np.ascontiguousarray(np.delete(v, 2, axis=1), dtype="<f8")
    return {sha256(values[:1]).digest(), *(sha256(values[j:j+4]).digest() for j in range(len(v)-3))}


def available_tracks():
    previous = set()
    archive = ROOT / "data/public/cmapss/CMAPSSData.zip"
    if sha256(archive.read_bytes()).hexdigest() != "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f":
        raise ValueError("C-MAPSS 原档变化")
    with ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith(("train_FD", "test_FD")):
                with z.open(name) as f:
                    for _, v in track_rows(f):
                        previous.update(signatures(v))
    for name, checksum in {"train.txt": PHM08_TRAIN_SHA, **FILES}.items():
        source = ROOT / "data/public/phm08" / name
        if sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError("PHM08 原档变化")
        tracks, duplicates = {}, []
        this_file = set()
        for uid, v in track_rows(source):
            marks = signatures(v)
            if marks & previous or marks & this_file:
                duplicates.append(uid)
            else:
                tracks[uid] = v
            this_file.update(marks)
        previous.update(this_file)
        yield name, tracks, duplicates


def censored_scores(records):
    predictions = defaultdict(list)
    for r in records:
        predictions[r["device"]].append(r)
    arms = list(records[0]["pred"])
    result = {}
    for key in arms:
        totals = defaultdict(float)
        for rows in predictions.values():
            last = rows[-1]["cycle"]
            times = np.array([r["cycle"] for r in rows])
            # q 是对最后可见循环剩余寿命的估计，未知真值 d≥0，当前各点误差都是 q-d。
            q = np.array([r["pred"][key] for r in rows]) - (last - times)
            old = np.array([r["pred"]["current-raw"] for r in rows]) - (last - times)
            gap = np.maximum(0., -q)
            totals["below"] += np.count_nonzero(gap)
            totals["gap2"] += float(gap @ gap)
            totals["pairs"] += len(q) - 1
            totals["change2"] += float(np.sum(np.diff(q) ** 2))
            a, slope = float(np.mean(q*q - old*old)), float(-2*np.mean(q-old))
            if abs(a) < 1e-8 and abs(slope) < 1e-8:
                totals["tie"] += 1
            elif a < -1e-8 and slope <= 0:
                totals["better"] += 1
            elif a > 1e-8 and slope >= 0:
                totals["worse"] += 1
            else:
                totals["depends_on_endpoint"] += 1
        result[key] = {"rows_below_observed_life": int(totals["below"]),
                       "below_rate": totals["below"] / len(records), "lower_bound_rmse": float(np.sqrt(totals["gap2"] / len(records))),
                       "countdown_rmse": float(np.sqrt(totals["change2"] / totals["pairs"])),
                       **{group: int(totals[group]) for group in ("tie", "better", "worse", "depends_on_endpoint")}}
    return result


def paired_known(pkg):
    results = {}
    for name, keys, _, groups, answers, _ in cohorts("engine", pkg):
        source = DEST / name / "issued.json"
        if sha256(source.read_bytes()).hexdigest() != read(DEST / name / "scores.json")["issued_sha256"]:
            raise ValueError("配对参照来源变化")
        data = read(source)["predictions"]
        issued = {k: data[k] for k in ("current-raw", "current-mapped", "tree-raw", "tree-mapped", "paper-mapped")}
        for treatment in ("raw", "mapped"):
            issued["half-" + treatment] = {k: .5*(data["current-" + treatment][k] + data["tree-" + treatment][k]) for k in keys}
        sealed = DEST / "paired" / name / "issued.json"
        atomic_json(sealed, {"predictions": issued})
        table = summary("engine", answers(), {"parent": data["current-raw"], **read(sealed)["predictions"]}, groups)
        atomic_json(sealed.with_name("scores.json"), {"arms": table, "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})
        results[name] = {k: [v["score"], v["worst_group"]] for k, v in table.items()}
    atomic_json(DEST / "paired/scores.json", results)


def main(automatic=False):
    pkg, _, batches = official("engine")
    if not automatic:
        paired_known(pkg)
    fit = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    lookup = EngineSettingReference([r.values for r in pkg.train if r.sample_id in fit])
    if automatic:
        lookup = SourceCombinations([r.values for r in pkg.train if r.sample_id in fit])
    tree, tree_receipt = frozen("engine")
    current, others = paths("engine", "engine")
    archive, = (ROOT / "data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "C") as trainer:
        models, versions = {}, {}
        for name, folder in (("current", current), ("paper", others["paper"])):
            record = read(folder / "run.json")["rounds"][-1]
            if record["selected"] != read(folder / record["submitted"])["payload"]["model_id"]:
                raise ValueError("使用权重须为官方末轮实际采用版")
            header, blob = ModelArtifacts(folder / "models").read(record["model_artifact"])
            path = Path(tmp) / (name + ".npz")
            path.write_bytes(blob)
            models[name] = trainer.Model.load(path)
            versions[name] = header["sha256"]
        for name, tracks, excluded in available_tracks():
            folder = (DEST / "automatic" if automatic else DEST) / "censored" / Path(name).stem
            if name == "train.txt":
                atomic_json(folder / "overlap.json", {"retained": len(tracks), "excluded_devices": excluded})
                continue
            records = []
            for uid, v in tracks.items():
                idx = sorted({*range(0, len(v), 20), len(v)-1})
                x = np.column_stack((v[idx, 3:], v[idx, :3]))
                mapped = lookup.transform(x)
                pred = {}
                for treatment, features in (("raw", x), ("mapped", mapped)):
                    for arm, model in models.items():
                        pred[arm + "-" + treatment] = model.predict(features)
                    pred["tree-" + treatment] = tree_values(tree["catboost"], "engine", features)
                    pred["half-" + treatment] = .5*(pred["tree-" + treatment] + pred["current-" + treatment])
                for j, k in enumerate(idx):
                    records.append({"device": uid, "cycle": k+1, "changed": bool(np.any(x[j] != mapped[j])),
                                    "pred": {arm: float(vals[j]) for arm, vals in pred.items()}})
            sealed = folder / "issued.json"
            atomic_json(sealed, {"predictions": records, "weights": versions, "tree": tree_receipt["catboost"]["sha256"],
                                  "mapping": lookup.identity, "source_sha256": FILES[name]})
            table = censored_scores(read(sealed)["predictions"])
            atomic_json(folder / "scores.json", {"arms": table, "rows": len(records), "devices": len(tracks),
                        "excluded_devices": excluded, "changed_rows": sum(r["changed"] for r in records),
                        "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})
            print(name, len(tracks), len(records), table["current-mapped"])


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", action="store_true")
    main(parser.parse_args().automatic)
