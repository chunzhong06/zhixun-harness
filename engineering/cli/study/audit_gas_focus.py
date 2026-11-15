"""复查官方气体开发对照：测量空间的近邻、当前轮候选，以及复用前后的逐条决定。"""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.data.official_gas import LABELS, read_official_gas
from harness.runtime.storage import atomic_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root, archive):
    plan = load(root / "protocol.json")
    prior = root.parent / plan["protocol"]["controls_directory"]
    baseline = load(prior / "static/result.json")
    package = read_official_gas(archive)
    explore_ids = {sid for batch in plan["rounds"].values() for sid in batch["explore"]}
    explore = [r for r in package.validation if r.sample_id in explore_ids]
    assert explore_ids.isdisjoint(plan["holdout"]) and len(explore) == len(explore_ids)
    def labels(rows):
        return np.array([r.label for r in rows])
    with np.load(BytesIO(package.checkpoint), allow_pickle=False) as model:
        def normalized(rows):
            x = np.array([row.values for row in rows])
            return np.clip((x - model["mean"]) / model["scale"], -10, 10)
        def predict(x):
            hidden = np.tanh(x @ model["w1"] + model["b1"])
            return (hidden @ model["w2"] + model["b2"]).argmax(axis=1) + 1
        train_x, explore_x = normalized(package.train), normalized(explore)
        train_y, explore_y = labels(package.train), labels(explore)
        train_pred, explore_pred = predict(train_x), predict(explore_x)
    assert explore_pred.tolist() == [baseline["full_validation"]["predictions"][r.sample_id] for r in explore]
    diagnosis = {}
    for label in (LABELS.index("Acetaldehyde")+1, LABELS.index("Ethanol")+1):
        # 最近邻仅作描述，固定训练集均值、尺度及裁剪；不拟合验证行，也不使用行号特征。
        closest = [train_y[np.argmin(np.sum((train_x-x)**2, axis=1))] for x in explore_x[explore_y == label]]
        def counts(values):
            return dict(Counter(LABELS[int(v)-1] for v in values))
        diagnosis[LABELS[label-1]] = {"train_predictions": counts(train_pred[train_y == label]),
            "explore_predictions": counts(explore_pred[explore_y == label]), "nearest_training_labels": counts(closest)}

    # 按源数据已有的特征定义分组，查看同一类在不同测量下是否更接近原训练资料。
    # 这些结果用于决定下一项调查，既不选线上模型，也不读取决定采用用的样本。
    views = {"amplitude": (0, 1), "rise": (2, 3, 4), "decay": (5, 6, 7)}
    view_counts = {}
    for name, positions in views.items():
        mask = np.isin(np.arange(train_x.shape[1]) % 8, positions)
        sources = train_x[:, mask]
        neighbours = np.array([train_y[np.argmin(np.sum((sources-x)**2, axis=1))] for x in explore_x[:, mask]])
        view_counts[name] = {LABELS[label-1]: {
            "correct": int(np.sum((neighbours == label) & (explore_y == label))),
            "false_positives": int(np.sum((neighbours == label) & (explore_y != label))),
            "support": int(np.sum(explore_y == label))} for label in (4, 1)}

    evidence, rounds_by_arm = {}, {}
    for name in ("focused", "focused-reuse"):
        state = load(root / name / "state.json")
        result = load(root / name / "result.json")
        rounds, changes = [], Counter()
        for row in result["rounds"]:
            bid, candidate = row["round"], row["candidate"]
            def scores(mid):
                return state["operations"][f"{bid}:explore:{mid}"]["value"]
            before, after = scores(row["incumbent"]), scores(candidate["model"])
            assert set(before["predictions"]) == set(after["predictions"]) == set(plan["rounds"][bid]["explore"])
            for sid, old in before["predictions"].items():
                target = state["samples"][sid]["label"]
                new = after["predictions"][sid]
                if old != new:
                    changes[(LABELS[target-1], LABELS[old-1], LABELS[new-1])] += 1
            rounds.append({"candidate": candidate["checkpoint_sha256"], "parent": candidate["parent_sha256"],
                "epoch_draws": candidate["original_draws"], "current_scores": [before["score"], after["score"]],
                "current_predictions": after["predictions"], "confirmation": row["confirmation"],
                "adopted": row["selected"] != row["incumbent"], "decision": row["decision"]})
        rounds_by_arm[name] = rounds
        evidence[name] = {"epochs": result["total_epochs"], "heldout_classes": result["holdout"]["classes"],
            "heldout_macro_f1": result["holdout"]["macro_f1"], "heldout_accuracy": result["holdout"]["accuracy"],
            "explore_prediction_changes": [{"truth": a, "before": b, "after": c, "count": n}
                for (a,b,c),n in sorted(changes.items())], "adoptions": sum(r["adopted"] for r in rounds),
            "reuses": sum(bool(r["candidate"]["reused"]) for r in result["rounds"]),
            "distinct_candidates": len({r["candidate"] for r in rounds}),
            "maintenance_seconds": result["maintenance_wall_seconds"]}
    first, cached = [load(root / n / "result.json") for n in ("focused", "focused-reuse")]
    identical = (rounds_by_arm["focused"] == rounds_by_arm["focused-reuse"]
                 and first["holdout"]["predictions"] == cached["holdout"]["predictions"])
    return {"baseline_diagnosis": diagnosis, "nearby_by_feature_family": view_counts,
            "arms": evidence, "reused_matches_retraining": identical}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    args = parser.parse_args()
    report = summarize(args.run, args.package)
    atomic_json(args.run / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
