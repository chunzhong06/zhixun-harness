"""用已发布观测比较轴承参照检索；先封存预测，再按整台设备评价。"""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys
from time import process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.contracts import digest
from harness.data.official_bearing import read_official_bearing
from harness.runtime.storage import atomic_json

CONFIG = ROOT / "engineering/configs/experiments/bearing-prefix-matching.json"
OUT = ROOT / "results/experiments/official-bearing/prefix-matching"
METHODS = ("snapshot_all", "snapshot_matched", "trend")


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def group_units(samples):
    units = defaultdict(list)
    for s in samples:
        units[s.unit_id].append(s)
    return {u: sorted(v, key=lambda s: s.time_end) for u, v in sorted(units.items())}


class ReferenceBank:
    """参照只能来自其他训练设备；同一片段的所有早期读数均取自该设备。"""

    def __init__(self, units, scale_from_checkpoint=None):
        self.rows = [s for seq in units.values() for s in seq]
        data = np.asarray([s.values for s in self.rows])
        if scale_from_checkpoint is None:
            self.mean, self.scale = data.mean(axis=0), data.std(axis=0)
            self.scale[self.scale < 1e-12] = 1
        else:
            self.mean, self.scale = scale_from_checkpoint
        self.values = self.standardize(data)
        self.times = np.asarray([s.time_end for s in self.rows])
        # 官方读取器已验证逐分钟连续。起点和排序决定引用的数组下标及并列时的顺序。
        for unit, seq in units.items():
            if any(s.time_end != i for i, s in enumerate(seq, 1)):
                raise ValueError(f"训练参照不连续：{unit}")

    def standardize(self, x):
        return np.clip((np.asarray(x) - self.mean) / self.scale, -10, 10)

    def describe(self, ids, long_minutes):
        refs = [self.rows[i] for i in ids]
        labels = [s.label for s in refs]
        return {"refs": [s.sample_id for s in refs], "median": float(np.median(labels)),
                "long": sum(y >= long_minutes for y in labels)}

    def search(self, query, past, cfg):
        count = cfg["neighbors"]
        distance = np.square(self.values - query).sum(axis=1)
        closest = np.argsort(distance, kind="stable")[:count]
        all_refs = self.describe(closest, cfg["long_minutes"])
        if len(past) < 2:
            return {m: all_refs for m in METHODS}
        # 时间从1开始，严格大于最大间隔才有同一训练设备的前序快照。
        eligible = np.flatnonzero(self.times > max(lag for lag, _ in past))
        if len(eligible) < count:
            raise ValueError("固定间隔下没有足够训练参照")
        same_distance = distance[eligible]
        trend_distance = same_distance.copy()
        current = self.values[eligible]
        for lag, point in past:
            diff = (query - point) - (current - self.values[eligible - lag])
            trend_distance += cfg["trend_weight_per_difference"] * np.square(diff).sum(axis=1)
        def describe_order(dist):
            return self.describe(eligible[np.argsort(dist, kind="stable")[:count]], cfg["long_minutes"])
        return {"snapshot_all": all_refs, "snapshot_matched": describe_order(same_distance),
                "trend": describe_order(trend_distance)}


def sequence(bank, stream, cfg, original=None):
    """stream 仅含已发放的身份、时刻、测量和轮次。查询侧没有寿命字段。"""
    result, history = [], []
    for sid, t, values, batch in stream:
        if history and t <= history[-1][1]:
            raise ValueError("已发布观测必须按同一设备时间递增")
        q = bank.standardize(values)
        matched = bank.search(q, [(t - old_t, v) for _, old_t, v in history], cfg)
        row = {"id": sid, "round": batch, "prior": [k for k, _, _ in history], "matches": matched}
        if original is not None:
            base = original[sid]
            preds = {"original": base}
            for method, match in matched.items():
                shift = max(0, match["median"] - base) if match["long"] > cfg["neighbors"] // 2 else 0
                preds[method] = base + cfg["upward_correction_fraction"] * shift
            row["predictions"] = preds
        result.append(row)
        history = (history + [(sid, t, q)])[-2:]
    return result


def measures(errors):
    return {"rmse": float(np.sqrt(np.mean(np.square(errors)))), "bias": fmean(errors)}


def score(rows, answers, model=False):
    report = {"rows": len(rows)}
    methods = ("original", *METHODS) if model else METHODS
    for method in methods:
        errors, changes = [], []
        for r in rows:
            value = r["predictions"][method] if model else r["matches"][method]["median"]
            errors.append(value - answers[r["id"]])
            if model:
                changes.append(abs(errors[-1]) - abs(r["predictions"]["original"] - answers[r["id"]]))
        stats = measures(errors)
        if method != "original":
            counts = [r["matches"][method]["long"] for r in rows]
            stats.update(any_long=sum(n > 0 for n in counts), majority_long=sum(n > 2 for n in counts))
        if model:
            stats.update(improved=sum(x < 0 for x in changes), worsened=sum(x > 0 for x in changes),
                         changed=sum(r["predictions"][method] != r["predictions"]["original"] for r in rows))
        report[method] = stats
    return report


def main():
    cfg = json.loads(CONFIG.read_bytes())
    for path, expected in cfg["inputs"].items():
        if hash_file(ROOT / path) != expected:
            raise ValueError(f"冻结输入已变化：{path}")
    archive, schedule, state_path, original_path = (ROOT / p for p in cfg["inputs"])
    # 输入的含义由既有文件的内容交叉检查。候选从始至终不更新官方模型权重。
    pkg = read_official_bearing(archive)
    plan = json.loads(schedule.read_bytes())
    original = json.loads(original_path.read_bytes())
    if original["checkpoint_sha256"].removeprefix("sha256:") != pkg.checkpoint_sha:
        raise ValueError("原版预测与官方权重来源不一致")
    units = group_units(pkg.train)
    train_y = {s.sample_id: s.label for s in pkg.train}
    folds, train_rows = {}, {}
    long_units, short_units = [], []
    start = process_time()
    for uid, samples in units.items():
        bank = ReferenceBank({u: seq for u, seq in units.items() if u != uid})
        stream = [(s.sample_id, s.time_end, s.values, None) for s in samples[::cfg["fold_query_stride_minutes"]]]
        rows = sequence(bank, stream, cfg)
        train_rows[uid] = rows
        early = [r for r in rows if train_y[r["id"]] >= cfg["long_minutes"]]
        folds[uid] = {"all": score(rows, train_y)}
        if early:
            folds[uid]["long"] = score(early, train_y)
            long_units.append(uid)
        else:
            short_units.append(uid)
    fold_cpu = process_time() - start
    short = {m: fmean(folds[u]["all"][m]["rmse"] for u in short_units) for m in METHODS}
    pass_longs = all(folds[u]["long"]["trend"]["rmse"] < folds[u]["long"]["snapshot_matched"]["rmse"]
                     for u in long_units)
    if len(long_units) != 2 or len(short_units) != 7:
        raise ValueError("整设备检验与冻结资料不一致")

    with np.load(BytesIO(pkg.checkpoint), allow_pickle=False) as weights:
        bank = ReferenceBank(units, (weights["mean"], weights["scale"]))
    start = process_time()
    samples = json.loads(state_path.read_bytes())["samples"]
    published = {}
    for uid, role in plan["roles"].items():
        stream = []
        for i, window in enumerate(plan["windows"]):
            block = sorted((s for s in samples.values() if s["unit_id"] == uid and s["role"] == role
                            and s["available_at"] == i), key=lambda s: s["time_end"])
            if digest([s["sample_id"] for s in block]) != window[role]["ids"]:
                raise ValueError("发布记录与冻结窗口不一致")
            stream.extend((s["sample_id"], s["time_end"], s["values"], i+1) for s in block)
        published[role] = sequence(bank, stream, cfg, original["predictions"])
    replay_cpu = process_time() - start

    # 三台的预测和邻居先写入同一份不可变证据，再接入答案。服务只进入结果描述。
    OUT.mkdir(parents=True, exist_ok=True)
    record = OUT / "predictions.json"
    predictions = {"config_sha256": hash_file(CONFIG), "source_sha256": hash_file(Path(__file__)),
                   "folds": train_rows, "replay": published}
    if record.exists() and json.loads(record.read_bytes()) != predictions:
        raise ValueError("已有冻结预测与本次不同，请为不同协议选择独立结果位置")
    if not record.exists():
        atomic_json(record, predictions)
    frozen = json.loads(record.read_bytes())["replay"]
    val_y = {s.sample_id: s.label for s in pkg.validation}
    explore = frozen["explore"]
    subsets = {"early": [r for r in explore if r["round"] <= 6], "last_two": [r for r in explore if r["round"] >= 9],
               "explore": explore, "last_four": [r for r in explore if r["round"] >= 7], "confirm": frozen["adopt"]}
    replay = {name: score(rows, val_y, model=True) for name, rows in subsets.items()}
    early = replay["early"]
    if early["rows"] != 48 or early["snapshot_all"]["any_long"] != 1:
        raise ValueError("未复现此前单点参照的48条观测")
    gates = {"each_long_train_unit_improves": pass_longs, "short_train_units_preserved": short["trend"] <= short["snapshot_matched"],
             "early_improves_all_controls": all(early["trend"]["rmse"] < early[m]["rmse"] for m in ("original", *METHODS[:2])),
             "end_preserved": replay["last_two"]["trend"]["rmse"] <= replay["last_two"]["original"]["rmse"],
             "confirmation_preserved": replay["confirm"]["trend"]["rmse"] <= replay["confirm"]["original"]["rmse"]}
    retain = all(gates.values())
    replay["service"] = score(frozen["serve"], val_y, model=True)
    report = {"predictions_sha256": hash_file(record), "folds": folds, "short_macro_rmse": short, "replay": replay,
              "criteria": gates, "retain_for_independent_test": retain,
              "cpu_seconds": {"folds_with_all_controls": fold_cpu, "public_replay_with_all_controls": replay_cpu}}
    # 同一预测的重放保留首次运行的计时。
    result = OUT / "result.json"
    if result.exists():
        previous = json.loads(result.read_bytes())
        if {k: v for k, v in previous.items() if k != "cpu_seconds"} != {k: v for k, v in report.items() if k != "cpu_seconds"}:
            raise ValueError("冻结评分发生变化")
    else:
        atomic_json(result, report)
    print(json.dumps({"long_units": long_units, "short_rmse": short, "criteria": gates,
                      "early": early, "cpu_seconds": report["cpu_seconds"]}))


if __name__ == "__main__":
    main()
