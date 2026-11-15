"""检验第三方树的追加收益是否集中在输入确实修复的观测，按设备进行配对重抽。"""
from hashlib import sha256

import numpy as np

from compare_group_objectives import atomic_json, official, read, summary
from score_censored_engines import censored_scores
from score_engine_setting_reference import DEST
from score_life_training_weights import cohorts


def paired_intervals(truth, predictions, groups):
    units = sorted(set(groups.values()))
    indexes = {k: i for i, k in enumerate(units)}
    keys = list(predictions["select"])
    units_by_row = np.asarray([indexes[groups[k]] for k in keys])
    counts = np.bincount(units_by_row)
    target = np.array([truth[k] for k in keys])
    draw = np.random.default_rng(17).integers(0, len(units), size=(10000, len(units)))
    n = counts[draw].sum(axis=1)
    errors = {}
    for arm, forecasts in predictions.items():
        sse = np.bincount(units_by_row, weights=(target - [forecasts[k] for k in keys]) ** 2)
        errors[arm] = np.sqrt(sse[draw].sum(axis=1) / n)
    return {k: np.quantile(errors[l]-errors[r], [.025, .5, .975]).tolist() for k, l, r in (
        ("auto-minus-old", "auto", "old"), ("auto-minus-paper", "auto", "paper"),
        ("auto-minus-tree", "auto", "tree"), ("select-minus-auto", "select", "auto"),
        ("select-minus-tree", "select", "tree"))}


def main():
    pkg, _, _ = official("engine")
    base = DEST / "automatic"
    for name, keys, _, groups, answers, _ in cohorts("engine", pkg):
        src = base / name / "issued.json"
        if sha256(src.read_bytes()).hexdigest() != read(src.with_name("scores.json"))["issued_sha256"]:
            raise ValueError("冻结对照内容不同")
        stored = read(src)
        pred = stored["predictions"]
        a, t = pred["current-auto"], pred["tree-auto"]
        # 独立查两种修复器在此数据上的等效性，混合条件只取自动修复器的实际输入差。
        if a != pred["current-mapped"] or t != pred["tree-mapped"]:
            raise ValueError("自动恢复与已核查的逐行修复标志不匹配")
        flags = stored["changed"]
        half = {i: .5*(a[i]+t[i]) for i in keys}
        predictors = {"old": pred["current-raw"], "auto": a, "paper": pred["paper-auto"], "tree": t,
                      "half": half, "select": {i: half[i] if flags[i] else a[i] for i in keys},
                      "opposite": {i: a[i] if flags[i] else half[i] for i in keys}}
        folder = base / "select" / name
        sealed = folder / "issued.json"
        atomic_json(sealed, {"predictions": predictors})
        forecasts = read(sealed)["predictions"]
        truth = answers()
        table = summary("engine", truth, {"parent": forecasts["old"], **forecasts}, groups)
        contrasts = paired_intervals(truth, forecasts, groups)
        atomic_json(folder / "scores.json", {"arms": table, "paired_device_intervals": contrasts,
                    "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})
        print(name, round(table["select"]["score"], 3), round(table["select"]["worst_group"], 3), contrasts["select-minus-auto"])
    for name in ("test", "final_test"):
        src = base / "censored" / name / "issued.json"
        if sha256(src.read_bytes()).hexdigest() != read(src.with_name("scores.json"))["issued_sha256"]:
            raise ValueError("新设备回放的冻结预测不同")
        records = read(src)["predictions"]
        for r in records:
            p = r["pred"]
            p["select"] = p["half-mapped"] if r["changed"] else p["current-mapped"]
            p["opposite"] = p["current-mapped"] if r["changed"] else p["half-mapped"]
        folder = base / "select" / name
        sealed = folder / "issued.json"
        atomic_json(sealed, {"predictions": records})
        atomic_json(folder / "scores.json", {"arms": censored_scores(read(sealed)["predictions"]),
                    "issued_sha256": sha256(sealed.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
