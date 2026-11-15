"""用相同样本比较单次与连续维护，单独登记第一次修好的样本有没有被第二次改坏。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"engineering/src"))
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import prediction_score
from harness.runtime.storage import atomic_json


def load(path):
    return json.loads(path.read_bytes())


def comparison(rows, reference, final, original, task):
    if set(reference) != set(final) or set(final) != {r.sample_id for r in rows}:
        raise ValueError("比较需要逐条配对")
    fixed = broken = prior_gains = lost = 0
    for row in rows:
        sid = row.sample_id
        old_ok = reference[sid] == row.label
        new_ok = final[sid] == row.label
        prior_good = old_ok and original[sid] != row.label
        fixed += new_ok and not old_ok
        broken += old_ok and not new_ok
        prior_gains += prior_good
        lost += prior_good and not new_ok
    return {"rows":len(rows), "prior_macro_f1":prediction_score(task, rows, reference),
            "new_macro_f1":prediction_score(task, rows, final), "corrected":fixed, "new_errors":broken,
            "previously_corrected":prior_gains, "lost_previous_corrections":lost}


def audit(folder, control, archive):
    package = read_official_gas(archive)
    by_id = {r.sample_id:r for r in package.validation}
    plan = load(folder/"protocol.json")
    baseline = load(control/"result.json")
    base_state = load(control/"state.json")
    original = load(ROOT/"results/experiments/official-gas/pilot-1/static/result.json")["full_validation"]["predictions"]
    ref_held = baseline["holdout"]["predictions"]
    arms = {}
    for arm in plan["protocol"]["arms"]:
        state = load(folder/arm/"state.json")
        result = load(folder/arm/"result.json")
        target = result["holdout"]["predictions"]
        summary = comparison([by_id[s] for s in ref_held],ref_held,target,original,package.task)
        adoptions = []
        review = []
        after, control_after = {}, {}
        online, online_control = {}, {}
        for r, base in zip(result["rounds"],baseline["rounds"],strict=True):
            bid = r["round"]
            if bid != base["round"]:
                raise ValueError("单次和连续维护的轮次不一致")
            key = f"{bid}:explore:"
            cand_pred = state["operations"][key+r["incumbent"]]["value"]["predictions"]
            base_pred = base_state["operations"][key+base["incumbent"]]["value"]["predictions"]
            assert set(cand_pred) == set(base_pred) == set(plan["rounds"][bid]["explore"])
            assert online.keys().isdisjoint(cand_pred)
            online.update(cand_pred)
            online_control.update(base_pred)
            if len(adoptions) == 2:
                after.update(cand_pred)
                control_after.update(base_pred)
                batch_review = comparison([by_id[s] for s in cand_pred],base_pred,cand_pred,original,package.task)
                review.append({"round":bid, **batch_review})
            if r["incumbent"] != r["selected"]:
                adoptions.append({"round":bid, "pair":r["candidate"]["gas_pair"], "confirmation":r["confirmation"],
                                  "exploration":r["candidate"]["current_scores"], "sha256":r["candidate"]["checkpoint_sha256"]})
        summary["adoptions"] = adoptions
        summary["online"] = comparison([by_id[s] for s in online], online_control, online, original, package.task)
        summary["after_second"] = comparison([by_id[s] for s in after],control_after,after,original,package.task) if after else None
        summary["post_second_rounds"] = review
        summary["class_results"] = result["holdout"]["classes"]
        summary["new_fits"] = sum(1 for r in result["rounds"] if r["candidate"] and not r["candidate"]["reused"])
        arms[arm] = summary
    atomic_json(folder/"comparison.json",arms)
    for arm,v in arms.items():
        print(arm, v["new_macro_f1"], v["corrected"], v["new_errors"], v["lost_previous_corrections"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--control", type=Path, default=ROOT/"results/experiments/official-gas/pilot-4/pair-shape-blend")
    p.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    args = p.parse_args()
    audit(args.run, args.control, args.package)
