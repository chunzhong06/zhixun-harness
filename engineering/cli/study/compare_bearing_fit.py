"""对比已冻结的两份轴承训练清单。默认只核对材料，实测最多新增一个单步候选。"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.run_rul_maintenance import PACKAGES, PROTOCOL, partition, describe
from engineering.cli.study.run_rul_series import SERIES, windows
from harness.contracts import ActionPlan, digest
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import RulPilotTools, read_parent
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.storage import LocalStore, atomic_json

PREVIOUS = ROOT / "results/experiments/fit-reuse/bearing"
OUTPUT = ROOT / "results/experiments/official-bearing/fit-manifest"
EXPECTED_IDS = ("aedadc9e2b94377991b4090ff0321e3c3f17cf5f3df8f1958ae17ae50d6f2178",
                "84cd05df86c5eb94b1a62b8d6839373de1437d76957fc156d144781d329e3589")


def load(path):
    return json.loads(path.read_bytes())


def freeze(path, value):
    """重入时验证原决定，防止已有权重或评价被配给新的协议。"""
    if path.exists():
        if load(path) != value:
            raise ValueError(f"已冻结的文件内容不同：{path}")
    else:
        atomic_json(path, value)


def length_sample(rows, budget):
    by_device = defaultdict(list)
    for row in rows:
        by_device[row.unit_id].append(row)
    if min(map(len, by_device.values())) < 2 or not 2 * len(by_device) <= budget <= len(rows):
        raise ValueError("预算或训练轨迹不满足每台至少保留首末两条的要求")
    total = len(rows)
    quota = {u: max(2, len(group) * budget // total) for u, group in by_device.items()}
    # 名额只依赖完整训练轨迹的长度；离目标份额同样远时按设备身份固定次序。
    def surplus(u):
        return total * quota[u] - budget * len(by_device[u])
    while sum(quota.values()) < budget:
        u = min((u for u in quota if quota[u] < len(by_device[u])), key=lambda u: (surplus(u), u))
        quota[u] += 1
    while sum(quota.values()) > budget:
        u = min((u for u in quota if quota[u] > 2), key=lambda u: (-surplus(u), u))
        quota[u] -= 1
    selected = []
    for u, count in sorted(quota.items()):
        ordered = sorted(by_device[u], key=lambda row: row.time_end)
        selected.extend(ordered[i * (len(ordered) - 1) // (count - 1)] for i in range(count))
    return tuple(selected)


def prior_predictions(artifact):
    """复用同一权重已完成的预测；只有旧作业未覆盖的观测才再推理。"""
    known = {}
    for path in (PREVIOUS / "jobs").glob("*/input.json"):
        payload = load(path)
        request = payload["request"]
        if request["kind"] != "predict" or request["artifact"] != artifact:
            continue
        receipt = load(path.with_name("state.json"))
        if (receipt["status"] != "completed" or receipt["request_id"] != digest(request)
                or digest(payload["rows"]) != request["inputs_digest"]
                or set(receipt["value"]) != {r["sample_id"] for r in payload["rows"]}):
            raise ValueError("旧候选预测与输入回执不匹配")
        for sid, value in receipt["value"].items():
            if known.setdefault(sid, value) != value:
                raise ValueError("同一旧模型的预测记录不一致")
    return known


def fill_predictions(tools, artifact, rows, predictions):
    missing = [{"sample_id": r.sample_id, "values": list(r.values)} for r in rows if r.sample_id not in predictions]
    if missing:
        predictions.update(tools.predict(artifact, missing))


def run(output, real):
    once, series = load(PROTOCOL), load(SERIES)
    filename, checksum, reader = PACKAGES["bearing"]
    archive = ROOT / "data/official/bearing" / filename
    baseline = ROOT / "results/experiments/official-bearing/baseline/bearing-baseline.json"
    if (sha256(archive.read_bytes()).hexdigest() != checksum
            or sha256(baseline.read_bytes()).hexdigest() != once["baseline_sha"]["bearing"]):
        raise ValueError("官方模型包或基线记录已改变")
    package = reader(archive)
    original = load(baseline)["predictions"]
    rows, roles = partition(package, once)
    equal = rows["fit"]
    length = length_sample(package.train, len(equal))
    lists = [sorted(r.sample_id for r in fit) for fit in (equal, length)]
    hashes = tuple(sha256(("\n".join(ids) + "\n").encode()).hexdigest() for ids in lists)
    if hashes != EXPECTED_IDS:
        raise ValueError("清单与上一轮已冻结的两套身份不符")
    batches = windows(package, roles, length, series, "bearing")
    groups = defaultdict(list)
    for b in batches:
        for row in b.samples:
            if row.role == "explore":
                name = "early" if b.ordinal < 6 else "late" if b.ordinal >= 8 else "transition"
                groups[name].append(row)
            elif row.role in ("adopt", "serve"):
                groups[row.role].append(row)
    source = load(PREVIOUS / "fixed_confirmed/workflow/state.json")
    old_ref = source["models"][source["completed"]["round-1"]["trials"][0]["model_id"]]["artifact"]
    old_models = ModelArtifacts(PREVIOUS / "models")
    context = {"package_sha256": checksum, "checkpoint_sha256": package.checkpoint_sha}
    old_manifest, _, depth = read_parent(old_models, old_ref, context)
    limits = ExecutionLimits(series["worker_wall_seconds"], series["worker_cpu_seconds"], series["worker_memory_bytes"])
    new_runner = LocalJobRunner(output / "jobs", ModelArtifacts(output / "models"), limits)
    new_tools = RulPilotTools("bearing", archive, package, length, (), original, new_runner,
                             batches=batches, reuse_identical_fit=True)
    old_tools = RulPilotTools("bearing", archive, package, equal, (), None,
                              LocalJobRunner(output / "jobs", old_models, limits), batches=batches, reuse_identical_fit=True)
    parent = new_tools.initial()
    action = ActionPlan("recent", epochs=1)
    if depth != 1 or digest(old_manifest["request"]) != old_tools.fit_identity(package.task, action, equal, parent):
        raise ValueError("原候选与本次父权重、清单或训练环境不一致")
    protocol = {"archive": checksum, "baseline_sha256": once["baseline_sha"]["bearing"],
                "parent": parent, "fixed_candidate": old_ref, "fit_ids": lists, "max_new_fits": 1,
                "worker": new_tools.worker_sha, "device_roles": roles,
                "groups": {k: [r.sample_id for r in members] for k, members in groups.items()},
                "acceptance": "前六轮低估幅度和RMSE均下降，末两轮平均高估与RMSE及短确认设备RMSE均不增加，相对原版；其余两轮单列。选择冻结后查服务设备。"}
    freeze(output / "protocol.json", protocol)
    if not real:
        return
    candidate = new_tools.build(package.task, "round-1", action, length, parent=parent)
    train_files = list((output / "jobs").glob("*/training.json"))
    if len(train_files) != 1:
        raise ValueError("实测未取得唯一训练回执")
    training = load(train_files[0])
    if training["train_rows"] != 72 or training["final_steps"] - training["initial_steps"] != 1:
        raise ValueError("训练用量与冻结协议不符")
    predictions = {"original": original, "equal": prior_predictions(old_ref), "length": {}}
    reused = set(predictions["equal"])
    def measure(names):
        members = [r for k in names for r in groups[k]]
        fill_predictions(old_tools, old_ref, members, predictions["equal"])
        fill_predictions(new_tools, candidate, members, predictions["length"])
        return {k: {arm: describe(package.task, groups[k], pred) for arm, pred in predictions.items()} for k in names}
    scores = measure(("early", "transition", "late", "adopt"))
    early, late, short = scores["early"], scores["late"], scores["adopt"]
    checks = {
        "early_underestimate": max(0, -early["length"]["mean_signed_error"]) < max(0, -early["original"]["mean_signed_error"]),
        "early_rmse": early["length"]["rmse"] < early["original"]["rmse"],
        "late_overestimate": max(0, late["length"]["mean_signed_error"]) <= max(0, late["original"]["mean_signed_error"]),
        "late_rmse": late["length"]["rmse"] <= late["original"]["rmse"],
        "short_rmse": short["length"]["rmse"] <= short["original"]["rmse"]}
    selected = "length" if all(checks.values()) else "original"
    freeze(output / "decision.json", {"protocol": digest(protocol), "candidate": candidate,
                                     "scores": scores, "checks": checks, "selected": selected})
    # 服务阶段使用已冻结的选择。服务答案仅在已发预测与反事实预测全部取得后按 ID 计分。
    service_truth = {r.sample_id: r for r in package.validation if roles[r.unit_id] == "serve"}
    for arm, tool, ref in (("equal", old_tools, old_ref), ("length", new_tools, candidate)):
        fill_predictions(tool, ref, groups["serve"], predictions[arm])
    service_rows = [service_truth[r.sample_id] for r in groups["serve"]]
    scores["serve"] = {arm: describe(package.task, service_rows, pred) for arm, pred in predictions.items()}
    observations = [r for name in ("early", "transition", "late", "adopt", "serve") for r in groups[name]]
    receipts = [load(p) for p in (output / "jobs").glob("*/state.json")]
    if any(r["status"] != "completed" for r in receipts):
        raise ValueError("存在未完成作业")
    freeze(output / "result.json", {"protocol": digest(protocol), "scores": scores, "selected": selected,
           "selected_service": scores["serve"][selected], "checks": checks, "physical_fits": len(train_files),
           "old_predictions_reused": sum(r.sample_id in reused for r in observations),
           "training_cpu_seconds": load(train_files[0].with_name("state.json"))["cpu_seconds"],
           "all_job_cpu_seconds": sum(r["cpu_seconds"] for r in receipts),
           "predictions": {r.sample_id: {arm: pred[r.sample_id] for arm, pred in predictions.items()} for r in observations}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="在核对成功后运行一个真实轴承候选")
    args = parser.parse_args()
    with LocalStore(OUTPUT).exclusive():
        run(OUTPUT, args.run)
