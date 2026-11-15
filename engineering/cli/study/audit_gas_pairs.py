"""只读复核在用版本、全局对照与局部改判。对照完毕后才打开已有开发留出。"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"engineering/src"))
import numpy as np
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import prediction_score, error_profile
from harness.models.gas_views import fit_view, predict_view
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import atomic_json


def read(p):
    return json.loads(p.read_bytes())


def audit(folder, package_path):
    plan = read(folder/"protocol.json")
    if sha256(package_path.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("气体包版本不符")
    package = read_official_gas(package_path)
    by_id = {r.sample_id:r for r in package.validation}
    held = [by_id[s] for s in plan["holdout"]]
    control = folder.parent/plan["protocol"]["controls_directory"]
    baseline = read(control/"static/result.json")
    original = baseline["full_validation"]["predictions"]

    def compare(rows, predictions):
        old = {r.sample_id:original[r.sample_id] for r in rows}
        corrected = sum(old[r.sample_id] != r.label == predictions[r.sample_id] for r in rows)
        harm = sum(old[r.sample_id] == r.label != predictions[r.sample_id] for r in rows)
        return {"count":len(rows), "baseline_macro_f1":prediction_score(package.task, rows, old),
                "macro_f1":prediction_score(package.task, rows, predictions), "corrected":corrected, "new_errors":harm,
                "classes":error_profile(package.task, rows, predictions)["classes"]}

    comparisons = {}
    for arm in plan["protocol"]["arms"]:
        result = read(folder/arm/"result.json")
        state = read(folder/arm/"state.json")
        later = {}
        decisions = []
        published = False
        for r in result["rounds"]:
            bid = r["round"]
            if published:
                appeared = state["operations"][f"{bid}:explore:{r['incumbent']}"]["value"]["predictions"]
                assert set(appeared) == set(plan["rounds"][bid]["explore"]) and later.keys().isdisjoint(appeared)
                later.update(appeared)
            if r["incumbent"] != r["selected"]:
                decisions.append({"round":bid, "pair":r["candidate"]["gas_pair"], "confirmation":r["confirmation"],
                                  "exploration":r["candidate"]["current_scores"]})
                published = True
        pred = result["holdout"]["predictions"]
        summary = compare(held, pred)
        assert summary["macro_f1"] == result["holdout"]["macro_f1"]
        summary.update(decisions=decisions, later=compare([by_id[s] for s in later], later) if later else None,
                       maintenance_seconds=result["maintenance_wall_seconds"],
                       fit_seconds=sum(r["candidate"]["adapter_fit_seconds"] for r in result["rounds"] if r["candidate"]),
                       new_fits=sum(1 for r in result["rounds"] if r["candidate"] and not r["candidate"]["reused"]))
        comparisons[arm] = summary
    # 全局参照使用同样的邻居、表示与概率配比，唯一去掉的条件是可改判的方向。
    x = np.asarray([r.values for r in held])
    with np.load(BytesIO(package.checkpoint), allow_pickle=False) as m:
        mean, scale = m["mean"], m["scale"]
        hidden = np.tanh(np.clip((x-mean)/scale,-10,10) @ m["w1"]+m["b1"])
        logits = hidden@m["w2"]+m["b2"]
        probs = np.exp(logits - logits.max(axis=1,keepdims=True))
        probs /= probs.sum(axis=1,keepdims=True)
    assert (probs.argmax(1)+1).tolist() == [original[r.sample_id] for r in held]
    for method in ("shape-blend", "shape-knn"):
        bank = fit_view([r.values for r in package.train],[r.label-1 for r in package.train],method,mean,scale)
        pred = predict_view(bank,x,mean,scale,probs).argmax(1)+1
        comparisons["global-"+method] = compare(held,{r.sample_id:int(v) for r,v in zip(held,pred,strict=True)})
    atomic_json(folder/"comparison.json", comparisons)
    for k,v in comparisons.items():
        print(k, round(v["macro_f1"],5), v["corrected"], v["new_errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    args = parser.parse_args()
    audit(args.run, args.package)
