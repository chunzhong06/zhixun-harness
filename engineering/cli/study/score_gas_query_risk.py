"""以完整确认回执为准比较四条既有气体流，来源等权计分。"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from engineering.cli.workflow.gas_query_stream import run
from harness.contracts import digest
from harness.methods.feedback_query_risk import POLICIES
from harness.runtime.storage import LocalStore, atomic_json

BASE = ROOT / "data/prepared/gas"
DEST = ROOT / "results/experiments/local-defaults/whole-maintenance/gas-query-risk"
STREAMS = {"全通道": "temperature-trials/full", "换件": "temperature-trials/replacement",
           "甲烷背景": "dynamic-mixtures/methane", "一氧化碳背景": "dynamic-mixtures/co"}
SEEDS = (17, 47)


def read(path):
    return json.loads(path.read_bytes())


def score(folder, report, labels, classes):
    y, predictions, keys = [], [], []
    for event in report["accepted"]:
        key = event["request"]
        item = read(folder / "accepted" / (key + ".json"))
        p = item["payload"]
        if digest([report["identity"], p]) != key or item["request_id"] != key:
            raise ValueError("预测交接身份改变")
        keys.extend(p["ids"])
        y.extend(labels[k] for k in p["ids"])
        predictions.extend(p["prediction"])
    if len(set(keys)) != len(keys) or set(keys) != set(labels):
        raise ValueError("评分未覆盖完整服务流")
    k = list(range(1, classes + 1))
    return {"F1": float(f1_score(y, predictions, labels=k, average="macro", zero_division=0)),
            "最差召回": float(recall_score(y, predictions, labels=k, average=None, zero_division=0).min()),
            "测量": len(keys), "回标": report["returned"]}


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已完成开发验收")
        results = {}
        for name, relative in STREAMS.items():
            manifest = BASE / relative / "stream.json"
            spec = read(manifest)
            rows = {}
            for method in POLICIES:
                for seed in SEEDS:
                    folder = DEST / name / method / str(seed)
                    report = run(manifest, folder, method, 8, seed)
                    labels = read(manifest.parent / "evaluation.json")
                    rows.setdefault(method, []).append(score(folder, report, labels, spec["classes"]))
            results[name] = rows
            print(json.dumps({name: {k: float(np.mean([r["F1"] for r in v])) for k, v in rows.items()}},
                             ensure_ascii=False), flush=True)
        mean = {m: {metric: float(np.mean([np.mean([e[metric] for e in r[m]]) for r in results.values()]))
                    for metric in ("F1", "最差召回")} for m in POLICIES}
        candidates = ("历史辨别力", "历史局部错误")
        winner = max(candidates, key=lambda k: mean[k]["F1"])
        fixed = max((k for k in POLICIES if k not in (*candidates, "BADGE")), key=lambda k: mean[k]["F1"])
        gain = mean[winner]["F1"] - mean[fixed]["F1"]
        decision = {"候选": winner, "固定": fixed, "对最强固定差值": gain,
                    "对论文差值": mean[winner]["F1"] - mean["BADGE"]["F1"],
                    "可进独立": min(gain, mean[winner]["F1"]-mean["BADGE"]["F1"]) >= .01
                                and mean[winner]["最差召回"] >= mean[fixed]["最差召回"]-.02}
        atomic_json(DEST / "scores.json", {"streams": results, "means": mean, "decision": decision})
        print(json.dumps({"汇总": mean, "取舍": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
