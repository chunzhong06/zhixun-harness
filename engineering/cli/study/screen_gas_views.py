"""按登记目录筛选候选；拟合只用官方训练行，比较只用既有探索行。"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
import numpy as np
from harness.data.official_gas import read_official_gas
from harness.evaluation.gas_pilot import partition
from harness.evaluation.metrics import prediction_score, error_profile
from harness.models.gas_views import fit_view, predict_view
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import atomic_json


def screen(archive, config, out):
    if sha256(archive.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("官方气体包版本不符")
    protocol = json.loads(config.read_bytes())
    package = read_official_gas(archive)
    batches, _ = partition(package, protocol)
    rows = [s for b in batches for s in b.samples if s.role == "explore"]
    x = np.asarray([r.values for r in rows])
    train = np.asarray([r.values for r in package.train])
    labels = np.asarray([r.label-1 for r in package.train])
    with np.load(BytesIO(package.checkpoint), allow_pickle=False) as parent:
        mean, scale = parent["mean"], parent["scale"]
        z = np.clip((x - mean)/scale, -10, 10)
        logits = np.tanh(z @ parent["w1"] + parent["b1"]) @ parent["w2"] + parent["b2"]
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
    baseline = prob.argmax(axis=1)+1
    truth = np.asarray([r.label for r in rows])

    def assess(pred):
        mapped = {r.sample_id: int(v) for r, v in zip(rows, pred, strict=True)}
        return {"macro_f1": prediction_score(package.task, rows, mapped),
                "classes": error_profile(package.task, rows, mapped)["classes"],
                "corrected": int(np.sum((baseline != truth) & (pred == truth))),
                "new_errors": int(np.sum((baseline == truth) & (pred != truth)))}

    result = {"protocol_sha256": sha256(config.read_bytes()).hexdigest(), "package_sha256": PACKAGE_SHA,
              "source_rows": len(train), "explore_ids": [r.sample_id for r in rows],
              "baseline": assess(baseline), "methods": {}, "selected": None}
    best = result["baseline"]["macro_f1"]
    for method in protocol["screen"]:
        started = perf_counter()
        view = fit_view(train, labels, method, mean, scale)
        preds = predict_view(view, x, mean, scale, prob).argmax(axis=1)+1
        entry = assess(preds)
        entry["seconds"] = perf_counter() - started
        result["methods"][method] = entry
        if entry["macro_f1"] > best:
            best, result["selected"] = entry["macro_f1"], method
    atomic_json(out, result)
    print(json.dumps({k: v["macro_f1"] for k,v in result["methods"].items()}))
    print("selected", result["selected"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen(args.package, args.protocol, args.output)
