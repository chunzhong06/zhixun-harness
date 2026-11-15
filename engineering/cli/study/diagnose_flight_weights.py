"""从已交接的预测解释组合的得失，真值只在交付完成后用于算分。"""
import json
from pathlib import Path

import h5py
import numpy as np

from score_flight_range_choice import DEST, dataset_path, read_flights
from harness.contracts import digest
from harness.models.flight_range_choice import NAMES
from harness.runtime.storage import atomic_json


def summarize(target, heads, issued, constant):
    lower, upper = heads.min(axis=1), heads.max(axis=1)
    reachable = np.clip(target, lower, upper)
    dynamic_error = (issued - target) ** 2
    fixed_error = (constant - target) ** 2
    return {
        "飞行数": len(target),
        "五种都高估": int((target < lower).sum()),
        "五种都低估": int((target > upper).sum()),
        "答案在五种预测范围内": int(((lower <= target) & (target <= upper)).sum()),
        "知道答案才可达的组合误差下界": float(np.sqrt(np.mean((target - reachable) ** 2))),
        "现场组合误差": float(np.sqrt(dynamic_error.mean())),
        "冻结历史均值权重误差": float(np.sqrt(fixed_error.mean())),
        "逐航班调整改善次数": int((dynamic_error < fixed_error).sum()),
        "逐航班调整退步次数": int((dynamic_error > fixed_error).sum()),
        "逐航班调整相同次数": int((dynamic_error == fixed_error).sum()),
    }


def diagnose(directory: Path, site: str, scores: dict, frozen: dict):
    key = scores["request"]
    payload = json.loads((directory / "accepted" / (key + ".json")).read_bytes())["payload"]
    if digest([site, payload]) != key:
        raise ValueError("逐航班诊断与已发预测的身份不符")
    keys = [tuple(k) for k in payload["ids"]]
    with h5py.File(dataset_path(site)) as data:
        truth = read_flights(data, "test", answers=True)
    if len(keys) != len(truth) or set(keys) != set(truth):
        raise ValueError("诊断与已发送的航班不一致")
    owner = np.asarray([u for u, _ in keys])
    target = np.asarray([truth[k] for k in keys])
    columns = payload["predictions"]
    heads = np.column_stack([columns[name] for name in NAMES])
    order = frozen["顺序"]
    constant = np.column_stack([columns[name] for name in order]) @ frozen["固定权重"]["现场权重固定为历史均值"]
    issued = np.asarray(columns["估错软组合"])
    if "事前主方案" in columns:
        np.testing.assert_allclose(issued, columns["事前主方案"])
        np.testing.assert_allclose(constant, columns["现场权重固定为历史均值"])
    totals = summarize(target, heads, issued, constant)
    np.testing.assert_allclose(totals["现场组合误差"], scores["scores"]["估错软组合"]["rmse"])
    groups = {}
    for unit in np.unique(owner):
        pick = owner == unit
        groups[str(unit)] = summarize(target[pick], heads[pick], issued[pick], constant[pick])
    return {"回执": key, "全部航班": totals, "各台": groups}


if __name__ == "__main__":
    frozen = json.loads((DEST / "fixed-controls.json").read_bytes())
    output = {}
    for directory in (DEST, DEST / "forward"):
        scores = json.loads((directory / "scores.json").read_bytes())
        for site, cohort in scores.items():
            output[site] = diagnose(directory, site, cohort, frozen)
    atomic_json(DEST / "forward" / "weight-diagnostics.json", output)
