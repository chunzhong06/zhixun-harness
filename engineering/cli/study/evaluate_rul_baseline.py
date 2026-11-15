"""准备或测量官方寿命起点。默认只核对输入、划分和权重，显式执行时仅作预测。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import file_digest
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.data.official_bearing import CHANNELS, MEASURES, read_official_bearing
from harness.data.official_engine import read_official_engine
from harness.data.official_rul import ARCHIVE_SHA
from harness.evaluation.metrics import prediction_score
from harness.runtime.storage import atomic_json


PACKAGES = {
    "bearing": ("cpu-b-rul-20261110-v1.zip", ARCHIVE_SHA["bearing"], read_official_bearing),
    "engine": ("cpu-c-guide-20261110-v1.zip", ARCHIVE_SHA["engine"], read_official_engine),
}


def input_paths(task):
    if task == "bearing":
        return [[["key", "features"], ["key", channel], ["key", name]] for channel in CHANNELS for name in MEASURES]
    return [[["key", name], ["index", i]] for name, count in (("sensors", 21), ("settings", 3)) for i in range(count)]


def inputs_only(task, row):
    """设备、周期、答案留在评价侧。原版程序只收到任务要求的测量值。"""
    if task == "engine":
        return {"sensors": row.values[:21], "settings": row.values[21:]}
    values = iter(row.values)
    return {"features": {channel: {name: next(values) for name in MEASURES} for channel in CHANNELS}}


def predict_offline(archive, package, task):
    """独立进程运行已固定版本的官方逐条预测入口；返回条数异常时不发布评价。"""
    with TemporaryDirectory(prefix="rul-predict-", dir=ROOT / "scratch") as temp:
        folder = Path(temp)
        for name in ("model.py", "data.py", "predict.py"):
            (folder / name).write_bytes(archive.read("baselines/" + name))
        (folder / "checkpoint.npz").write_bytes(package.checkpoint)
        payload = "".join(json.dumps({"inputs": inputs_only(task, row)}, allow_nan=False) + "\n"
                          for row in package.validation)
        env = dict(os.environ, BASELINE_CPU_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, "-B", "-s", "predict.py", "--checkpoint", "checkpoint.npz"],
                                 cwd=folder, env=env, input=payload, text=True, capture_output=True, timeout=180)
        if process.returncode:
            raise RuntimeError(f"官方预测进程失败：{process.stderr[-1000:]}")
        predictions = {}
        for row, line in zip(package.validation, process.stdout.splitlines(), strict=True):
            value = json.loads(line)["outputs"]["value"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("公开权重的预测应为非负有限数值")
            predictions[row.sample_id] = value
        return predictions


def describe(task, rows, predictions):
    paired = {r.sample_id: predictions[r.sample_id] for r in rows}
    errors = [paired[r.sample_id] - r.label for r in rows]
    return {"rows": len(rows), "rmse": prediction_score(task, rows, paired),
            "mae": math.fsum(map(abs, errors)) / len(rows), "mean_signed_error": math.fsum(errors) / len(rows),
            "over": sum(e > 0 for e in errors), "under": sum(e < 0 for e in errors)}


def longest_overestimate(rows, predictions):
    """只连接同一设备的相邻时刻，并保留首次达到最长的区间。"""
    best = {"length": 0, "start": None, "end": None}
    length, previous = 0, None
    for row in rows:
        t = row.time_start
        if predictions[row.sample_id] > row.label:
            length = length + 1 if previous == t - 1 else 1
            if length > best["length"]:
                best = {"length": length, "start": t - length + 1, "end": t}
        else:
            length = 0
        previous = t
    return best


def evaluate(package, predictions):
    rows = package.validation
    if set(predictions) != {r.sample_id for r in rows}:
        raise ValueError("评价须覆盖完整公开验证集并保留原始样本身份")
    units, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        units[row.unit_id].append(row)
        groups[row.group].append(row)
    overall = describe(package.task, rows, predictions)
    per_unit = {}
    for unit, members in units.items():
        ordered = sorted(members, key=lambda r: r.time_start)
        windows = defaultdict(list)
        for row in ordered:
            windows[(row.time_start - 1) // 50].append(row)
        per_unit[unit] = {**describe(package.task, ordered, predictions),
                          "longest_overestimate": longest_overestimate(ordered, predictions),
                          "windows": [{"start": w[0].time_start, "end": w[-1].time_end,
                                       **describe(package.task, w, predictions)} for w in windows.values()]}
    return {"overall": overall, "units": per_unit, "groups": {
        g: describe(package.task, members, predictions) for g, members in groups.items()}}


def run(task, output, *, predict=False):
    import numpy as np
    name, expected, reader = PACKAGES[task]
    with (ROOT / "data/official" / task / name).open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("官方公开寿命包版本与本次评价依据不符")
        source.seek(0)
        package = reader(source)
        with np.load(BytesIO(package.checkpoint), allow_pickle=False) as weights:
            metadata = json.loads(str(weights["metadata"].item()))
        if ([json.loads(p) for p in metadata["paths"]] != input_paths(task)
                or metadata["task"] != "regression" or metadata["profile"] != ("B" if task == "bearing" else "C")):
            raise ValueError("当前输入与初始权重的特征路径或任务不符")
        with ZipFile(source) as archive:
            official_score = json.loads(archive.read("training.json"))["validation_after"]
            record = {"package": package.package_id, "archive_sha256": expected, "validation_sha256": package.validation_sha,
                      "checkpoint_sha256": package.checkpoint_sha, "target_unit": package.task.target_unit,
                      "validation_rows": len(package.validation), "validation_units": len({r.unit_id for r in package.validation}),
                      "training_units": len({r.unit_id for r in package.train}), "feature_count": len(metadata["paths"]),
                      "official_report": official_score}
            if predict:
                start = perf_counter()
                predictions = predict_offline(archive, package, task)
                prediction_wall_seconds = perf_counter() - start
                measured = evaluate(package, predictions)
                differences = {k: measured["overall"][k] - official_score[k] for k in ("mae", "rmse")}
                record.update(evidence_kind="real_model", measured=measured, predictions=predictions,
                              prediction_wall_seconds=prediction_wall_seconds, cpu_threads=1,
                              differences_from_official=differences,
                              matches_official=all(abs(d) <= 1e-6 for d in differences.values()))
            else:
                record["evidence_kind"] = "input_preparation"
    path = output / (f"{task}-baseline.json" if predict else f"{task}-preparation.json")
    atomic_json(path, record)
    if predict and not record["matches_official"]:
        raise ValueError(f"实测基线偏离随包报告，已保留预测供核对：{path}")
    return path


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(PACKAGES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enable-real-prediction", action="store_true", help="用官方权重对完整公开验证集做一次本地 CPU 预测")
    args = parser.parse_args()
    print(run(args.task, args.output, predict=args.enable_real_prediction))
