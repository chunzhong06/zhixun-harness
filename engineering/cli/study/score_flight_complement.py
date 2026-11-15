"""在完整机队留出下比较冻结的单航班读法；预测时只传入已真实发出的模型输出。"""
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.contracts import digest
from harness.methods.flight_complement import FlightComplement
from harness.models.flight_range_choice import NAMES
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from score_ds02_forward import read_flights
from score_ncmapss_field import measures, file_hash

BASE = ROOT / "results/experiments/local-defaults/whole-maintenance"
DEST = BASE / "flight-sequence-complement"
PAPER = "Darrah双向序列本地适配"


def read(path):
    return json.loads(path.read_bytes())


def receipt(folder, site, scores):
    key = scores["request"]
    payload = read(folder / "accepted" / (key + ".json"))["payload"]
    if digest([site, payload]) != key:
        raise ValueError("已发预测身份改变")
    return payload


def history():
    ranges = BASE / "flight-range-choice"
    paper = BASE / "flight-load-neighbors/paper"
    old = read(ranges / "scores.json")
    previous = read(ranges / "forward/scores.json")
    sequences = read(paper / "scores.json")["sites"]
    cohorts = {}
    for site, summary in {**old, **previous}.items():
        folder = ranges if site in old else ranges / "forward"
        payload = receipt(folder, site, summary)
        published = receipt(paper, site, sequences[site]) if site in old else payload
        if payload["ids"] != published["ids"]:
            raise ValueError("论文与森林的航班不同")
        keys = [tuple(k) for k in payload["ids"]]
        path, = (ROOT / "data/public/ncmapss").glob(f"N-CMAPSS_{site}*.h5")
        with h5py.File(path) as data:
            answers = read_flights(data, "test", answers=True)
        if len(keys) != len(answers) or set(keys) != set(answers):
            raise ValueError("交付与作者答案无法对齐")
        truth = np.array([answers[k] for k in keys])
        columns = payload["predictions"]
        pred = np.column_stack([*(columns[name] for name in NAMES), published["predictions"][PAPER]])
        cohorts[site] = (keys, pred, truth, columns)
    return cohorts


def training(cohorts, held=None):
    source = {k: v for k, v in cohorts.items() if k != held}
    pred = np.vstack([v[1] for v in source.values()])
    truth = np.concatenate([v[2] for v in source.values()])
    sites = np.concatenate([np.repeat(k, len(v[2])) for k, v in source.items()])
    return FlightComplement(pred, truth, sites)


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("该实验已经完成")
        cohorts = history()
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        scores = {}
        for site, (keys, pred, truth, columns) in cohorts.items():
            model = training(cohorts, held=site)
            out = model.forecast_all(pred)
            out["前次现场规则"] = np.array(columns["估错软组合"])
            for i, name in enumerate(NAMES):
                out[name] = pred[:, i]
            payload = {"ids": [list(k) for k in keys], "predictions": {k: v.tolist() for k, v in out.items()}}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("开发留组交接失败")
            delivered = receipt(DEST, site, {"request": key})
            owner = np.array([k[0] for k in keys])
            metrics = {k: measures(truth, np.array(v), owner) for k, v in delivered["predictions"].items()}
            scores[site] = {"request": key, "scores": metrics, "rows": len(keys)}
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in scores.values()]))
                 for k in metrics}
        choices = ("航班互补调权", "现场与六工具固定各半")
        best = min(choices, key=means.get)
        fixed = min((k for k in means if k not in (*choices, "前次现场规则")), key=means.get)
        gain = 1 - means[best] / means[fixed]
        selection = {"现场规则": best, "最强固定": fixed, "开发降幅": gain,
                     "值得独立验证": gain >= .01 and means[best] < means["固定用论文"]}
        model = training(cohorts)
        joblib.dump(model, DEST / "choice.joblib")
        atomic_json(DEST / "scores.json", {"sites": scores, "means": means, "decision": selection,
                                          "artifact": file_hash(DEST / "choice.joblib")})
        print(json.dumps({"留机队真实预测": means, "取舍": selection}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
