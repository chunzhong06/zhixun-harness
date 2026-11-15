"""封存相同发动机模型在原设置与源条件设置下的服务输出，再按原答案配对评价。"""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, frozen, paths, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, official, read, remaining_cycles, summary
from harness.models.engine_setting_reference import EngineSettingReference
from harness.models.source_combinations import SourceCombinations
from harness.models.model_artifacts import ModelArtifacts
from score_tree_reference_transfer import suites


DEST = OUT / "setting-reference"


def evaluate(name, rounds, groups, answers, source, trainer, work, reference, train_settings, tree, parent, auto=None):
    current, alternatives = paths(source, "engine")
    folders = {"current": current, "paper": alternatives["paper"]}
    runs = {k: read(folder / "run.json")["rounds"] for k, folder in folders.items()}
    stores = {k: ModelArtifacts(folder / "models") for k, folder in folders.items()}
    source_folder = OUT / "teacher-readout/engine" / name
    previous = source_folder / "issued.json"
    if sha256(previous.read_bytes()).hexdigest() != read(source_folder / "scores.json")["issued_sha256"]:
        raise ValueError("冻结的原预测摘要不符")
    old = read(previous)["predictions"]
    issued, changes, versions = {}, {}, {}
    low, high, median = float(min(train_settings)), float(max(train_settings)), float(np.median(train_settings))
    elapsed = 0.
    for index, rows in sorted(rounds.items()):
        ids, x = zip(*rows, strict=True)
        x = np.asarray(x)
        start = perf_counter()
        mapped = reference.transform(x)
        elapsed += perf_counter() - start
        clipped, filled = x.copy(), x.copy()
        clipped[:, -1] = np.clip(clipped[:, -1], low, high)
        filled[:, -1] = median
        models = {"tree": lambda v: tree_values(tree, "engine", v), "original": parent.predict}
        for key in folders:
            record = runs[key][index]
            accepted = read(folders[key] / record["submitted"])["payload"]
            if accepted["model_id"] != record["selected"]:
                raise ValueError("选中权重与交接回执不符")
            header, blob = stores[key].read(record["model_artifact"])
            weights = work / (key + ".npz")
            weights.write_bytes(blob)
            models[key] = trainer.Model.load(weights).predict
            versions.setdefault(key, {})[index] = header["sha256"]
        treatments = {"raw": x, "range": clipped, "median": filled, "mapped": mapped}
        if auto is not None:
            treatments.update(auto=auto.transform(x), unseen=auto.transform(x, unseen_only=True))
        compared = treatments["auto"] if auto is not None else mapped
        changes.update(zip(ids, np.any(compared != x, axis=1).tolist(), strict=True))
        for key, predict in models.items():
            for treatment, values in treatments.items():
                pred = predict(values)
                if treatment == "raw" and key != "original":
                    old_arm = ("fixed-tree" if "fixed-tree" in old else "tree") if key == "tree" else key
                    if max(abs(v - old[old_arm][i]) for i, v in zip(ids, pred, strict=True)) > 1e-7:
                        raise ValueError("在相同服务输入上未复现旧版、树或论文的实际输出")
                arm = key + "-" + treatment
                issued.setdefault(arm, {}).update(zip(ids, pred.tolist(), strict=True))
    out = (DEST if auto is None else DEST / "automatic") / name
    sealed = out / "issued.json"
    atomic_json(sealed, {"predictions": issued, "changed": changes, "models": versions})
    recorded = read(sealed)
    forecasts = recorded["predictions"]
    truth = answers()
    table = summary("engine", truth, {"parent": forecasts["current-raw"], **forecasts}, groups)
    slices = {}
    for changed in (True, False):
        ids = [k for k, v in recorded["changed"].items() if v == changed]
        if ids:
            slices[str(changed)] = {"rows": len(ids), "rmse": {k: float(np.sqrt(np.mean([(p[i] - truth[i]) ** 2 for i in ids])))
                                                               for k, p in forecasts.items()}}
    atomic_json(out / "scores.json", {"arms": table, "by_change": slices, "match_seconds": elapsed,
                "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})
    print(name, len(changes), sum(changes.values()), {k: round(table[k]["score"], 3) for k in
          ("current-raw", "current-range", "current-median", "current-mapped", "paper-mapped", "tree-raw", "tree-mapped")})


def main(automatic=False):
    pkg, _, batches = official("engine")
    ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    x = np.asarray([r.values for r in pkg.train if r.sample_id in ids])
    reference = EngineSettingReference(x)
    auto = SourceCombinations(x) if automatic else None
    tree = frozen("engine")[0]["catboost"]
    archive, = (ROOT / "data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "C") as trainer:
        work = Path(tmp)
        parent_path = work / "parent.npz"
        parent_path.write_bytes(pkg.checkpoint)
        parent = trainer.Model.load(parent_path)
        decoded = [json.loads(path) for path in parent.meta["paths"]]
        if tuple(f"{key[1]}.{index[1]}" for key, index in decoded) != pkg.task.features:
            raise ValueError("模型特征次序不同于训练清单")
        for name in ("engine", "phm08"):
            folder, _ = paths(name, "engine")
            state = read(folder / "state.json")["samples"]
            groups, rounds = {}, {}
            for k, r in state.items():
                if r["role"] == "serve":
                    if r["label"] is not None:
                        raise ValueError("服务输入存在寿命真值")
                    groups[k] = r["unit_id"]
                    rounds.setdefault(r["available_at"], []).append((k, r["values"]))
            answers = (lambda: {r.sample_id: r.label for r in pkg.validation}) if name == "engine" else remaining_cycles
            evaluate(name, rounds, groups, answers, name, trainer, work, reference, x[:, -1], tree, parent, auto)
        for name, rounds, groups, answers in suites("engine", False):
            evaluate(name, rounds, groups, answers, "phm08", trainer, work, reference, x[:, -1], tree, parent, auto)
    output = DEST if auto is None else DEST / "automatic"
    atomic_json(output / "training.json", {"source": pkg.train_sha, "rows": len(x), "rules": [
        {"setting1": k[0], "setting2": k[1], "sensor1": k[2], "setting3": v} for k, v in sorted(reference.mapping.items())]})


def live(name, automatic=False):
    from engineering.cli.workflow.engine_reference import run_prepared
    from engineering.cli.workflow.public_maintenance import assemble_public
    from compare_public_selection import external_batches
    from compare_group_objectives import digest
    from score_local_defaults import cost
    root = ROOT / "data/official"
    output = (DEST / "automatic" if automatic else DEST) / "live" / name
    for through in (3, 10):
        harness, batches = assemble_public("engine", root, output / "regular", "first-proposal", {"mode": "rule"})
        if name == "phm08":
            batches = external_batches("engine", batches)
            harness.tools.values = {r.sample_id: r.values for b in batches for r in b.samples if r.role != "fit"}
        prior = output / "run.json"
        if through == 3 and prior.exists() and read(prior)["completed_rounds"] > 3:
            continue
        report = run_prepared(root, output, harness, batches, through, policy="source-combinations" if automatic else "source-setting")
    expected = read(DEST / name / "issued.json")
    issued = {}
    for index, item in enumerate(report["accepted"]):
        response = read(output / "accepted" / (item["request"] + ".json"))
        payload = response["payload"]
        pred = payload["predictions"]
        keys = {r.sample_id for r in batches[index].samples if r.role == "serve" and r.available_at == index}
        if (payload["batch"] != batches[index].batch_id or keys != pred.keys() or issued.keys() & keys
                or response["request_id"] != digest([report["identity"], payload])
                or item["outputs_sha256"] != digest(pred)
                or payload["decision"]["changed_rows"] != sum(expected["changed"][k] for k in keys)):
            raise ValueError("实际交接的轮次、输入或结果身份不符")
        issued.update(pred)
    target = (read(DEST / "automatic" / name / "issued.json")["predictions"]["current-auto"] if automatic else
              expected["predictions"]["current-mapped"])
    delta = max(abs(v - target[k]) for k, v in issued.items())
    if issued.keys() != target.keys() or delta > 1e-8:
        raise ValueError("真实交付与冻结模型的独立配对重算不一致")
    answers = {r.sample_id: r.label for r in harness.tools.package.validation} if name == "engine" else remaining_cycles()
    units = {r.sample_id: r.unit_id for b in batches for r in b.samples if r.role == "serve"}
    result = summary("engine", answers, {"parent": expected["predictions"]["current-raw"], "live": issued}, units)
    ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    atomic_json(output / "scores.json", {"arms": result, "paired": len(issued), "max_abs_difference": delta,
                "extra_wall_seconds": sum(x["reference_seconds"] for x in report["accepted"]),
                "regular_cost": cost(output / "regular", sha256(next((root / "engine").glob("*.zip")).read_bytes()).hexdigest(), ids),
                "run_sha256": sha256((output / "run.json").read_bytes()).hexdigest()})
    print(name, len(issued), result["live"]["score"], delta)


def verify_single_rows(name):
    from engineering.cli.workflow.engine_input_predict import infer
    output = DEST / "automatic" / ("integrated/engine" if name == "engine" else "live/phm08")
    report = read(output / "run.json")
    samples = read(output / "regular/state.json")["samples"]
    verified, max_error = 0, 0.
    for ordinal, entry in enumerate(report["accepted"], 1):
        accepted = read(output / "accepted" / (entry["request"] + ".json"))["payload"]["predictions"]
        def requests():
            for key in accepted:
                row = samples[key]
                if row["role"] != "serve" or row["label"] is not None:
                    raise ValueError("逐条推理请求必须来自无答案的服务流")
                yield {"task": "regression", "inputs": {"sensors": row["values"][:21], "settings": row["values"][21:]}}
        predicted = infer(output, ordinal, ROOT / "data/official", requests())
        for expected, result in zip(accepted.values(), predicted, strict=True):
            max_error = max(abs(expected-result["outputs"]["value"]), max_error)
            verified += 1
    if max_error > 1e-8 or len(report["accepted"]) != 10:
        raise ValueError("单条接口未复现全部轮次的真实交接")
    atomic_json(output / "single-rows.json", {"count": verified, "max_abs_difference": max_error,
                "lookup_sha256": sha256((output / "source-combinations.json").read_bytes()).hexdigest(),
                "index_sha256": sha256((output / "issued-models.json").read_bytes()).hexdigest(),
                "run_sha256": sha256((output / "run.json").read_bytes()).hexdigest()})
    print(name, verified, max_error)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--live", choices=("phm08", "engine"))
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--single", choices=("engine","phm08"))
    args = parser.parse_args()
    if args.single:
        verify_single_rows(args.single)
    elif args.live:
        live(args.live, args.automatic)
    else:
        main(args.automatic)
