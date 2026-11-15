"""按实测日历先交付下一根轴承的预测，待其结束后才把误差和实测寿命开放给维护程序。"""
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from compare_fixed_tree_references import frozen
from harness.contracts import digest
from harness.models.feedback_choice import HistoricalToolChoice
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import atomic_json
from prepare_paderborn_author import DATA, ROOT, destination


OUT = ROOT / "results/experiments/local-defaults/whole-maintenance/paderborn-forward"
PROTOCOL = ROOT / "engineering/configs/experiments/paderborn-forward-maintenance.json"
CHOICES = ("phme-feature", "phme-spectrum", "with-age", "median-clock", "conditional-life")
ARCHIVE_SIZE = 4


def read(path):
    return json.loads(path.read_bytes())


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


@dataclass
class Device:
    name: str
    start: int
    end: int
    ids: list
    spectral: np.ndarray
    settings: np.ndarray
    ages: dict
    durations: dict
    audit: dict

    def inputs(self, clock):
        measured = np.column_stack((self.spectral, self.settings))
        return {"phme-feature": measured, "phme-spectrum": self.spectral,
                "with-age": np.column_stack((measured, self.ages[clock]))}


def devices():
    for row in read(DATA / "manifest.json")["files"]:
        if fingerprint(destination(row["path"])) != row["sha256"]:
            raise ValueError(f"作者文件变动 {row['path']}")
    data = []
    for i in range(2, 18):
        if i in (5, 16):
            continue
        unit = f"B{i:02}"
        operating = pd.read_csv(DATA / "op" / (unit + ".csv"))
        timestamps = pd.to_datetime(operating.iloc[:, 0], format="%d-%b-%Y %H:%M:%S")
        seconds = timestamps.to_numpy(dtype="datetime64[s]").astype(np.int64)
        raw = np.load(DATA / "fft" / (unit + ".npy"), allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != 200 or len(raw) > len(seconds):
            raise ValueError(f"振动与工况记录无法对齐 {unit}")
        intervals = np.diff(seconds)
        if np.min(intervals) <= 0:
            raise ValueError(f"实测时钟倒序 {unit}")
        active = np.r_[0, np.cumsum(np.minimum(intervals, 2 * intervals[0]))] / 60
        elapsed = (seconds - seconds[0]) / 60
        ix = np.arange(0, len(raw), 20)
        op = operating.iloc[ix, 1:].to_numpy(dtype=float)
        features = raw[ix].reshape(len(ix), 20, 10).max(axis=2)
        if not np.isfinite(features).all() or op.shape != (len(ix), 6) or not np.isfinite(op).all():
            raise ValueError(f"观测内容不完整 {unit}")
        audit = {"rows_full": len(raw), "op_rows": len(seconds), "start": str(timestamps.iloc[0]),
                 "end": str(timestamps.iloc[-1]), "max_gap_minutes": float(max(intervals) / 60),
                 "large_gaps": int(np.sum(intervals > 5 * np.median(intervals))),
                 "active_cap_seconds": int(2 * intervals[0])}
        data.append(Device(unit, int(seconds[0]), int(seconds[-1]), [f"{unit}/{j}" for j in ix],
                           features, op, {"calendar": elapsed[ix], "capped": active[ix]},
                           {"calendar": float(elapsed[-1]), "capped": float(active[-1])}, audit))
    return sorted(data, key=lambda record: record.start)


class Tools:
    def __init__(self, completed, clock, phase_template):
        self.lives = np.array([u.durations[clock] for u in completed])
        y = np.concatenate([u.durations[clock] - u.ages[clock] for u in completed])
        inputs = [u.inputs(clock) for u in completed]
        self.models = {}
        for name in CHOICES[:3]:
            model = make_pipeline(MinMaxScaler(), HistGradientBoostingRegressor(random_state=2000))
            self.models[name] = model.fit(np.vstack([items[name] for items in inputs]), y)
        normalized = np.concatenate([1 - u.ages[clock] / u.durations[clock] for u in completed])
        self.phase = clone(phase_template).fit(np.vstack([u.spectral for u in completed]), normalized)

    def predict(self, inputs, elapsed):
        forecasts = {name: np.maximum(model.predict(inputs[name]), 0.) for name, model in self.models.items()}
        forecasts["phase-tree"] = np.clip(self.phase.predict(inputs["phme-spectrum"]), 0, 1) * np.median(self.lives)
        forecasts["median-clock"] = np.maximum(np.median(self.lives) - elapsed, 0.)
        forecasts["conditional-life"] = np.array([
            np.mean(self.lives[self.lives > t] - t) if np.any(self.lives > t) else self.lives.mean() for t in elapsed
        ])
        return forecasts


def switch(losses, previous):
    if not losses:
        return previous
    means = {key: np.mean([row[key] for row in losses]) for key in CHOICES}
    best = min(CHOICES, key=means.get)
    return best if means[best] < means[previous] and means[best] <= .99 * means[previous] else previous


def loss_on_completed(record, issued, clock):
    if issued["ids"] != record.ids:
        raise ValueError("设备失效回标和当时交付的行错位")
    y = record.durations[clock] - record.ages[clock]
    return {key: float(np.mean((np.array(issued["predictions"]["update/" + key]) - y)**2)) for key in CHOICES}


def summarize(records, paths, clock):
    by_arm = {}
    for unit in records:
        batch = read(paths[unit.name])["payload"]
        y = unit.durations[clock] - unit.ages[clock]
        for arm, p in batch["predictions"].items():
            err = np.array(p) - y
            by_arm.setdefault(arm, {})[unit.name] = (float(err @ err), float(abs(err).sum()), len(err))
    rows = sum(len(u.ids) for u in records)
    table = {}
    for arm, groups in by_arm.items():
        individual = {k: float(np.sqrt(v[0] / v[2])) for k, v in groups.items()}
        table[arm] = {"rmse": float(np.sqrt(sum(v[0] for v in groups.values()) / rows)),
                      "device_mean_rmse": float(np.mean(list(individual.values()))),
                      "worst": max(individual.values()), "by_device": individual,
                      "mae": sum(v[1] for v in groups.values()) / rows}
    candidates = [a for a in table if a.startswith(("static/", "update/"))]
    fixed = min(candidates, key=lambda a: table[a]["rmse"])
    # 事后逐台择优不进入运行器，仅衡量已发候选理论上最多可以补回多少误差。
    oracle = {u.name: min(candidates, key=lambda a: by_arm[a][u.name][0]) for u in records}
    oracle_rmse = np.sqrt(sum(by_arm[a][k][0] for k, a in oracle.items()) / rows)
    gates = {a: table[a]["rmse"] <= .98 * min(table[fixed]["rmse"], table["once"]["rmse"]) and
             table[a]["worst"] <= 1.01 * table[fixed]["worst"] for a in ("ongoing", "nearest")}
    return {"scores": table, "best_fixed": fixed, "rows": rows, "oracle_rmse": float(oracle_rmse),
            "oracle_tools": oracle, "gate": gates}


def evaluate(data, clock, phase_template):
    protocol = fingerprint(PROTOCOL)
    output = OUT / clock
    endpoint = LocalAcceptance()
    endpoint.output = output / "accepted"
    initial = data[:ARCHIVE_SIZE]
    tested = data[ARCHIVE_SIZE:]
    if max(u.end for u in initial) >= min(u.start for u in tested):
        raise ValueError("启动档案在首台服务前未完整返回")
    cache = {}
    costs = []
    def trained(history):
        key = tuple(u.name for u in history)
        if key not in cache:
            start = perf_counter()
            cache[key] = Tools(history, clock, phase_template)
            costs.append({"archive": list(key), "rows": sum(len(u.ids) for u in history),
                          "fits": 4, "wall_seconds": perf_counter() - start})
        return cache[key]
    installed = trained(initial)
    initial_scales = np.vstack([u.settings[:, [2, 4]] for u in initial]).std(axis=0)
    initial_scales[initial_scales == 0] = 1.
    paths = {}
    events = []
    observed = {}
    legacy = HistoricalToolChoice(CHOICES, CHOICES[0])
    ongoing = nearest = CHOICES[0]
    once = None
    for unit in tested:
        completed = [u for u in data if u.end < unit.start]
        new = [u for u in completed if u.name in paths and u.name not in observed]
        if new:
            for done in sorted(new, key=lambda u: u.end):
                historical = read(paths[done.name])["payload"]
                observed[done.name] = loss_on_completed(done, historical, clock)
                legacy.update([observed[done.name]])
            if once is None:
                once = switch(list(observed.values()), CHOICES[0])
        ongoing = switch(list(observed.values()), ongoing)
        distances = {u.name: np.sum(((u.settings[0, [2, 4]] - unit.settings[0, [2, 4]]) / initial_scales)**2)
                     for u in completed if u.name in observed}
        neighbors = sorted(distances, key=distances.get)[:3]
        nearest = switch([observed[k] for k in neighbors], nearest)
        choices = {"legacy": legacy.active, "once": once or CHOICES[0], "ongoing": ongoing, "nearest": nearest}
        inputs, elapsed = unit.inputs(clock), unit.ages[clock]
        old = installed.predict(inputs, elapsed)
        live = trained(completed).predict(inputs, elapsed)
        predictions = {"static/" + k: v.tolist() for k, v in old.items()}
        predictions.update({"update/" + k: v.tolist() for k, v in live.items()})
        predictions.update({a: live[k].tolist() for a, k in choices.items()})
        event = {"unit": unit.name, "archive": [u.name for u in completed], "newly_released": [u.name for u in new],
                 "choices": choices, "prior_scored": list(observed), "nearest_units": neighbors}
        payload = {**event, "ids": unit.ids, "predictions": predictions}
        req = digest([protocol, clock, payload])
        receipt = endpoint.submit(req, payload)
        if not receipt["accepted"] or receipt["request_id"] != req:
            raise ValueError("交付未确认")
        paths[unit.name] = endpoint.output / (req + ".json")
        events.append({**event, "request": req})
    report = {**summarize(tested, paths, clock), "events": events, "fits": costs, "protocol_sha256": protocol,
              "manifest_sha256": fingerprint(DATA / "manifest.json"),
              "timing": {u.name: {**u.audit, "minutes": u.durations[clock]} for u in data}}
    atomic_json(output / "scores.json", report)
    print(clock, report["rows"], {a: round(v["rmse"], 2) for a, v in report["scores"].items()}, flush=True)
    return report


if __name__ == "__main__":
    trajectories = devices()
    phase = frozen("bearing")[0]["catboost"]
    for clock in ("calendar", "capped"):
        evaluate(trajectories, clock, phase)
