"""复核同一公开十轮中候选复用前后的决定、预测及实际费用，只读两个运行目录。"""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.evaluation.round_evidence import round_evidence
from harness.data.official_bearing import read_official_bearing
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.storage import atomic_json


def read(path):
    return json.loads(path.read_bytes())


def same(left, right, location):
    if left != right:
        raise ValueError("配对不一致：" + location)


def cost(folder):
    jobs = [read(p) for p in (folder / "jobs").glob("*/state.json")]
    if not jobs or any(j["status"] != "completed" for j in jobs):
        raise ValueError("物理作业回执不完整")
    trained = {p.parent.name for p in (folder / "jobs").glob("*/training.json")}
    return {"physical_fits": len(trained), "physical_jobs": len(jobs),
            "training_job_cpu_seconds": sum(j["cpu_seconds"] for j in jobs if j["request_id"] in trained),
            "all_worker_cpu_seconds": sum(j["cpu_seconds"] for j in jobs)}


def comparison_view(folder, arm):
    state = read(folder / arm / "workflow/state.json")
    models = ModelArtifacts(folder / "models")

    @lru_cache(maxsize=None)
    def weights(model_id):
        # 跨轮复用会改变候选的显示名称。实际比较原件内容，读取时核对清单与完整权重。
        ref = state["models"][model_id]["artifact"]
        manifest, _ = models.read(ref)
        return manifest["sha256"]

    def scored(batch, purpose, mid):
        value = state["operations"][f"{batch}:{purpose}:{mid}"]["value"]
        return {k: value[k] for k in ("sample_versions", "predictions", "score", "score_definition")
                if k in value}

    rounds = []
    for i in range(1, 11):
        batch = f"round-{i}"
        row = state["completed"][batch]
        decision = row["decision"]
        current, nominee, selected = (decision[k] for k in ("current_model", "nominee", "selected_model"))
        trace = round_evidence(state, batch)
        # 预算及费用证据因复用而变化；比较计划的实际动作和理由，引用在各自原上下文中核对。
        actions = [{k: v for k, v in action.items() if k != "evidence_refs"} for action in trace["planned_actions"]]
        for action in trace["planned_actions"]:
            for ref in action["evidence_refs"]:
                same(state["planning_evidence"][ref]["scope"]["batch_id"], batch, f"{arm}/{batch}/提议依据")
        trials = []
        for trial in row["trials"]:
            mid = trial["model_id"]
            same(trial["status"], "completed", f"{arm}/{batch}/候选状态")
            trials.append({"parent": weights(state["models"][mid]["parent_id"]), "candidate": weights(mid),
                           "training_ids": trial["training_ids"], "evaluation": scored(batch, "explore", mid)})
        confirmation = None
        if decision.get("candidate_score") is not None:
            confirmation = [scored(batch, "adopt", mid) for mid in (current, nominee)]
        rounds.append({"batch": batch, "input": row["input_digest"], "actions": actions,
            "incumbent": weights(current), "nominee": weights(nominee), "selected": weights(selected),
            "issue": trace["observation"].get("device_bias_issue"),
            "explore": None if arm == "static" else scored(batch, "explore", current),
            "trials": trials, "confirmation": confirmation, "reason": decision["reason"],
            "outcome": trace["outcome"], "gain": decision.get("improvement"), "service": row["predictions"]})
    executions = [op for op in state["operations"].values() if op.get("execution")]
    builds = [op["execution"] for op in executions if op["name"].startswith("build:")]
    return rounds, {"assessed": sum(len(r["trials"]) for r in rounds), "logical_fits": len(builds),
        "training_job_cpu_seconds": sum(op["cpu_seconds"] for op in builds),
        "all_worker_cpu_seconds": sum(op["execution"]["cpu_seconds"] for op in executions)}


def audit(control, replay, frozen):
    for name, checksum in frozen["control_files_sha256"].items():
        same(sha256((control / name).read_bytes()).hexdigest(), checksum, name)
    same(read(control/"protocol.json"), read(replay/"protocol.json"), "分组与训练预算")
    old, new = (read(p/"comparison.json") for p in (control, replay))
    report = {"control": frozen["control"], "replay": replay.relative_to(ROOT).as_posix(),
              "physical": {"before": cost(control), "after": cost(replay)}, "arms": {}}
    for arm in old["arms"]:
        before, old_cost = comparison_view(control, arm)
        after, new_cost = comparison_view(replay, arm)
        for left, right in zip(before, after, strict=True):
            same(left, right, arm + "/" + left["batch"])
        for key in ("overall", "per_device", "worst_round_rmse", "source_digest"):
            same(old["arms"][arm][key], new["arms"][arm][key], arm+"/"+key)
        report["arms"][arm] = {"before": old_cost, "after": new_cost, "rmse": new["arms"][arm]["overall"]["rmse"],
            "service_predictions_matched": sum(len(r["service"]) for r in after),
            "rounds": [{"round": r["batch"], "outcome": r["outcome"], "service_rows": len(r["service"]),
                        "paired_fields_equal": True} for r in after]}
    return report


def bias_evidence(control, archive):
    """用训练集与已揭晓的探索观测诊断偏差；服务标签不参与选题或寻找参照。"""
    import numpy as np
    same(sha256(archive.read_bytes()).hexdigest(), read(control/"protocol.json")["archive"], "官方包")
    package = read_official_bearing(archive)
    state = read(control/"fixed_confirmed/workflow/state.json")
    with ZipFile(archive) as z, np.load(BytesIO(z.read("checkpoint.npz")), allow_pickle=False) as weights:
        mean, scale = weights["mean"], weights["scale"]
    # 沿用原模型的输入尺度。五近邻只描述历史测量的覆盖，不参与本轮维护或改判。
    bank = np.clip((np.array([s.values for s in package.train])-mean)/scale, -10, 10)
    long_life = np.array([s.label >= 1000 for s in package.train])
    rounds = []
    for i in range(1, 11):
        b = f"round-{i}"
        row = state["completed"][b]
        def predictions(mid):
            return state["operations"][f"{b}:explore:{mid}"]["value"]["predictions"]
        old = predictions(row["decision"]["current_model"])
        new = predictions(row["trials"][0]["model_id"])
        errors, long_neighbors = [], []
        for sid, y in old.items():
            sample = state["samples"][sid]
            errors.append(y-sample["label"])
            q = np.clip((np.array(sample["values"])-mean)/scale, -10, 10)
            closest = np.argsort(((bank-q)**2).sum(axis=1), kind="stable")[:5]
            long_neighbors.append(int(long_life[closest].sum()))
        rounds.append({"round": b, "observations": len(old), "underestimated": sum(e<0 for e in errors),
                       "mean_bias_minutes": fmean(errors), "squared_error_sum": sum(e*e for e in errors),
                       "candidate_mean_shift_minutes": fmean(new[k]-old[k] for k in old),
                       "queries_with_long_life_neighbor": sum(n>0 for n in long_neighbors)})
    return {"rounds": rounds, "long_life_threshold_minutes": 1000,
            "early_six_squared_error_share": sum(r["squared_error_sum"] for r in rounds[:6]) /
                                              sum(r["squared_error_sum"] for r in rounds),
            "long_training_rows_by_unit": dict(Counter(r.unit_id for r in package.train if r.label>=1000))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/experiments/fit-reuse")
    study = parser.parse_args().study.resolve()
    protocol = read(study/"protocol.json")
    control = ROOT/protocol["control"]
    report = audit(control, study/"bearing", protocol)
    filename = read(control/"run-definition.json")["package"]
    report["bias_evidence"] = bias_evidence(control, ROOT/"data/official/bearing"/filename)
    atomic_json(study/"paired-audit.json", report)
    print(json.dumps(report["physical"], ensure_ascii=False))
