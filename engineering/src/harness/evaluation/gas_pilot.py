"""官方气体本地对照。先冻结样本用途；留出答案只用于结束后的共同评价。"""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
from zipfile import ZipFile

from harness.contracts import ActionPlan, BatchContext, EngineConfig, digest
from harness.data.official_gas import LABELS, read_official_gas
from harness.evaluation.metrics import error_profile, prediction_score
from harness.evaluation.round_evidence import round_evidence
from harness.methods.planning.gas_continuation import FirstWindowPairPlanner, FocusedGasPlanner, GasContinuationPlanner, GasPairPlanner, gas_issue
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import OfficialGasTools, PACKAGE_SHA
from harness.runtime.engine import Harness
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.storage import LocalStore, atomic_json
from harness.integrations.official_driver import bind_driver


def partition(package, protocol, order=None):
    groups = defaultdict(list)
    for row in package.validation:
        groups[row.label].append(row)
    rounds = [[] for _ in range(protocol["rounds"])]
    held = []
    for label, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: digest([protocol["seed"], r.sample_id]))
        count = max(1, round(len(rows)*protocol["holdout_fraction"]))
        held.extend(ordered[:count])
        for index, row in enumerate(ordered[count:]):
            ordinal = (index//2 + label) % len(rounds)
            role = "explore" if index % 2 == 0 else "adopt"
            rounds[ordinal].append(replace(row, role=role, available_at=ordinal))
    if order is not None:
        if sorted(order) != list(range(1, len(rounds)+1)):
            raise ValueError("回放顺序应覆盖每个原分组且只用一次")
        rounds = [[replace(row, available_at=position) for row in rounds[original-1]]
                  for position, original in enumerate(order)]
    batches = tuple(BatchContext(package.task.task_id, f"round-{i+1}", i,
                    tuple(package.train if i == 0 else ()) + tuple(rows)) for i, rows in enumerate(rounds))
    return batches, tuple(held)


class FixedGasPlanner:
    identity = "official-gas-fixed-three-epochs-v1"

    def propose(self, context):
        facts = {f["kind"]: f for f in context["facts"]["evidence"]}
        ref = facts["performance"]["id"]
        action = ActionPlan("recent", reason="固定配方从当前检查点续训三遍", epochs=3, evidence_refs=(ref,))
        return {"actions": [action.to_record()], "stop": False}


def evaluate_checkpoint(package, tools, artifact, rows):
    inputs = tuple({"sample_id": r.sample_id, "values": r.values} for r in rows)
    predictions = tools.predict(artifact, inputs)
    profile = error_profile(package.task, rows, predictions)
    return {"macro_f1": prediction_score(package.task, rows, predictions), "accuracy": profile["accuracy"],
            "classes": [{**r, "name": LABELS[r["label"]-1]} for r in profile["classes"]],
            "predictions": predictions, "count": len(rows)}


def run_arm(path, protocol_file, output, arm, order_name=None, *, driver=None):
    """每组独立保存状态与权重；恢复必须使用相同协议、模型包和分组。"""
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    order = None
    if "orders" in protocol or order_name is not None:
        if order_name not in protocol.get("orders", {}) or order_name not in {"reverse", "rotate"}:
            raise ValueError("须显式选择冻结的回放顺序")
        order = protocol["orders"][order_name]
        output = output / order_name
    with LocalStore(output / arm / "pilot-lock").exclusive():
        return _run_arm(path, protocol, output, arm, order, driver)


def _run_arm(path, protocol, output, arm, order, driver=None):
    if arm not in protocol.get("arms", ["static", "fixed", "adaptive"]):
        raise ValueError("比较组与冻结协议不一致")
    if protocol["max_candidate_epochs"] != 3:
        raise ValueError("不兼容的预算")
    if sha256(path.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("实验固定的官方包已改变")
    package = read_official_gas(path)
    batches, held = partition(package, protocol, order)
    folder = output / arm
    driver = driver or {"mode": "rule"}
    query_replay = arm == "query-replay"
    observed_query = arm in {"query-current", "query-first"}
    query_arm = query_replay or observed_query
    if driver["mode"] != "rule" and (query_arm or arm in {"static", "fixed", "sequence-first-window"}):
        raise ValueError("固定对照保留原规则；替换驱动应使用自主比较组")
    plan = {"protocol": protocol, "package_sha256": PACKAGE_SHA, "training": package.train_sha256,
            "validation": package.validation_sha256, "holdout": [r.sample_id for r in held],
            "rounds": {b.batch_id: {role: [r.sample_id for r in b.samples if r.role == role]
                       for role in ("explore", "adopt")} for b in batches}}
    with LocalStore(output / "protocol-lock").exclusive():
        manifest = output / "protocol.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
            raise ValueError("输出目录已有不同的实验协议")
        atomic_json(manifest, plan)
    if observed_query:
        # 先锁定原来的十段编号，再取维护前缀；留出集与下一段不会重新划分。
        prefix = protocol.get("maintenance_prefix", len(batches))
        if type(prefix) is not int or not 1 <= prefix <= len(batches):
            raise ValueError("维护前缀超出已冻结的批次")
        batches = batches[:prefix]
    focused = arm in {"focused", "focused-reuse"}
    pair = arm.removeprefix("pair-") if arm.startswith("pair-") else None
    sequence = arm.removeprefix("sequence-") if arm.startswith("sequence-") else None
    if arm == "sequence-first-window":
        sequence = "protected"
    if sequence:
        pair = "shape-blend"
    if arm == "sequence-first-window":
        # 中断恢复时只重读首轮已提交的计划输入，后续轮次仍按原先冻结的清单执行。
        planner = FirstWindowPairPlanner.restore(pair,LocalStore(folder).load(),batches[0].batch_id)
    elif pair:
        planner = GasPairPlanner(pair)
    elif focused:
        planner = FocusedGasPlanner()
    elif query_arm:
        from harness.models.gas_fit_queries import GasQueryReplayPlanner
        planner = GasQueryReplayPlanner()
    elif arm == "adaptive":
        planner = GasContinuationPlanner()
    else:
        planner = FixedGasPlanner()
    planner = bind_driver(planner, folder, **driver)
    saved_result = folder / "result.json"
    if saved_result.exists():
        saved = json.loads(saved_result.read_text(encoding="utf-8"))
        if (saved["protocol_digest"] != digest(plan) or saved["arm"] != arm
            or len(saved["rounds"]) != len(batches)
            or saved.get("driver_identity") != (planner.identity if driver["mode"] != "rule" else None)):
            raise ValueError("已完成结果与当前任务或驱动不匹配")
        return saved
    if "controls_directory" in protocol:
        parent = output.parent.parent if order is not None else output.parent
        control = parent / protocol["controls_directory"]
        prior = json.loads((control / "protocol.json").read_text(encoding="utf-8"))
        expected_rounds = prior["rounds"] if order is None else {
            f"round-{i}":prior["rounds"][f"round-{original}"] for i, original in enumerate(order, 1)}
        if expected_rounds != plan["rounds"] or any(prior[k] != plan[k] for k in plan if k not in ("protocol","rounds")) or any(
                prior["protocol"][k] != protocol[k] for k in ("max_candidate_epochs", "worker_wall_seconds",
                    "worker_cpu_seconds", "worker_memory_bytes", "seed", "rounds", "holdout_fraction")):
            raise ValueError("共同对照的数据、批次或预算不一致")
        for name, expected in protocol["controls_sha256"].items():
            raw = (control / name / "result.json").read_bytes()
            if sha256(raw).hexdigest() != expected or json.loads(raw)["protocol_digest"] != digest(prior):
                raise ValueError("复用的对照结果已改变")
    elif arm != "static":
        baseline = json.loads((output / "static" / "result.json").read_text(encoding="utf-8"))
        if baseline["protocol_digest"] != digest(plan) or not baseline["baseline_reproduced"]:
            raise ValueError("须先复现同协议的官方起点")
    limits = ExecutionLimits(protocol["worker_wall_seconds"], protocol["worker_cpu_seconds"], protocol["worker_memory_bytes"])
    artifacts = ModelArtifacts(folder / "models")
    runner = LocalJobRunner(folder / "jobs", artifacts, limits)
    if observed_query:
        from harness.models.gas_observed_queries import ObservedGasQueryTools
        tool_type = ObservedGasQueryTools
        if protocol.get("reuse_query_weights") is True:
            from harness.models.gas_query_reuse import ReusableGasQueryTools
            tool_type = ReusableGasQueryTools
        tools = tool_type(path, package, batches, runner, folder, arm.removeprefix("query-"))
    elif query_replay:
        from harness.models.gas_fit_queries import GasQueryTools
        tools = GasQueryTools(path, package, batches, runner, protocol["gas_fit_queries"])
    else:
        tools = OfficialGasTools(path, package, batches, runner, focus=focused, reuse=bool(pair) or arm == "focused-reuse",
                                 pair_method=pair, sequence=sequence,
                                 pair_only=bool(pair) and protocol.get("action_scope") == "protected_pair_only")
    # 每项筛选只训一遍；多项回测合计与原每轮三遍上限一致。
    config = EngineConfig(execution_mode="real_model", recent_cap=len(package.train),
                          max_trials=protocol["max_candidate_epochs"] if query_replay else 1,
                          policy="plan_feedback", maintenance_trigger="never" if arm == "static" else "every_batch",
                          worker_cpu_seconds=limits.cpu_seconds, worker_memory_bytes=limits.memory_bytes)
    engine = Harness(package.task, config, tools, planner, LocalStore(folder), allow_real_execution=True)
    started = time.perf_counter()
    state = engine.run(batches)
    elapsed = time.perf_counter() - started
    failed = [k for k, v in state["operations"].items() if v["status"] == "failed"]
    if failed or len(state["completed"]) != len(batches):
        raise ValueError(f"运行未完整完成，请查操作回执：{failed}")
    initial = next(m for m in state["models"].values() if m["parent_id"] is None)
    result = {"protocol_digest": digest(plan), "arm": arm, "maintenance_wall_seconds": elapsed, "rounds": [],
              "model_experiment": True, "total_epochs": 0, "training_seconds": 0., "training_worker_cpu_seconds": 0.}
    if driver["mode"] != "rule":
        result["driver_identity"] = planner.identity
    if arm == "static":
        reproduction = evaluate_checkpoint(package, tools, initial["artifact"], package.validation)
        with ZipFile(path) as archive:
            expected = json.loads(archive.read("training.json"))["validation_after"]
        result["full_validation"] = reproduction
        result["provided_scores"] = expected
        result["baseline_reproduced"] = all(abs(reproduction[k]-v) <= protocol["baseline_tolerance"] for k, v in expected.items())
        if not result["baseline_reproduced"]:
            raise ValueError(f"起点成绩与随包记录不符：{reproduction['macro_f1']}，{expected}")

    for batch in batches:
        bid = batch.batch_id
        done = state["completed"][bid]
        decision = done["decision"]
        ctx = next((e["context"] for e in state["events"] if e["key"] == f"{bid}:plan_context:0"), None)
        entry = {"round": bid, "incumbent": decision["current_model"], "selected": done["model_id"],
                 "decision": decision["reason"], "issue": gas_issue(ctx) if ctx else None, "candidate": None,
                 "round_evidence": round_evidence(state, bid),
                 "confirmation": [decision.get("baseline_score"), decision.get("candidate_score")]}
        if query_arm:
            entry["backtested_candidates"] = []
        for trial in done["trials"]:
            mid = trial.get("model_id")
            if not mid:
                continue
            ref = state["models"][mid]["artifact"]
            manifest, _ = artifacts.read(ref)
            job = folder / "jobs" / ref["key"]
            trained = json.loads((job / "training.json").read_text(encoding="utf-8"))
            receipt = json.loads((job / "state.json").read_text(encoding="utf-8"))
            parent = state["models"][entry["incumbent"]]["artifact"]
            if receipt["status"] != "completed" or (
                    manifest["request"]["parent"] != parent and not (
                    observed_query and protocol.get("reuse_query_weights") is True
                    and artifacts.read(manifest["request"]["parent"])[0]["sha256"] == artifacts.read(parent)[0]["sha256"])):
                raise ValueError("候选与实际生效父版本不一致")
            epochs = trained["final_epochs"] - trained["initial_epochs"]
            reused = any(e["kind"] == "fit_reused" and e["batch_id"] == bid and e["model_id"] == mid
                         for e in state["events"])
            if reused:
                epochs = 0
            before = state["operations"].get(f"{bid}:explore:{entry['incumbent']}", {}).get("value", {})
            after = state["operations"].get(f"{bid}:explore:{mid}", {}).get("value", {})
            entry["candidate"] = {"model": mid, "parent_sha256": trained["parent_checkpoint_sha256"],
                                  "checkpoint_sha256": trained["checkpoint_sha256"], "epochs": epochs,
                                  "exploration": [trained["validation_before"], trained["validation_after"]],
                                  "training_seconds": 0. if reused else trained["training_seconds"]}
            if focused or pair:
                entry["candidate"].update(reused=reused, focus_label=trained.get("focus_label"),
                    original_draws=trained.get("sampling_history", []), current_scores=[before.get("score"), after.get("score")])
                if reused:
                    # 本轮复用没有逐遍评价，原作业的探索分数不能当作本轮证据。
                    entry["candidate"].pop("exploration")
            if trained.get("gas_pair"):
                entry["candidate"].update(gas_pair=trained["gas_pair"],
                    adapter_fit_seconds=0. if reused else trained["adapter_fit_seconds"])
            if query_arm:
                if reused:
                    # 训练回执来自第一次构建；本轮提案来源及成绩以本轮观察为准。
                    entry["candidate"].pop("exploration", None)
                    entry["candidate"].update(reused=True, current_scores=[before.get("score"), after.get("score")],
                                              current_proposal=tools.records[bid]["proposal"])
                entry["backtested_candidates"].append({**entry["candidate"], "query": trained["gas_fit_query"],
                                                        "macro_f1": trial["exploration_score"]})
            result["total_epochs"] += epochs
            if not reused:
                result["training_seconds"] += trained["training_seconds"]
                result["training_worker_cpu_seconds"] += receipt["cpu_seconds"]
        if query_arm:
            # 最后尝试者未必胜出。这里只登记交给独立确认的那个候选。
            nominee = decision["nominee"]
            entry["candidate"] = next((c for c in entry["backtested_candidates"] if c["model"] == nominee), None)
            # 采用决定已冻结后再汇总确认类别，不让这批答案参与拟案。
            confirmation = [row for row in batch.samples if row.role == "adopt" and row.available_at == batch.ordinal]
            entry["confirmation_pool_by_class"] = {label: sum(row.label == i for row in confirmation)
                                                   for i, label in enumerate(LABELS, 1)}
        result["rounds"].append(entry)
    if result["total_epochs"] > protocol["rounds"]*protocol["max_candidate_epochs"]:
        raise ValueError("实际训练超过冻结预算")
    # 该组完整决策冻结后才评分；评价结果不会进入计划器或下一轮的采用资料。
    served = {}
    for r in result["rounds"]:
        mid = r["selected"]
        if mid not in served:
            served[mid] = evaluate_checkpoint(package, tools, state["models"][mid]["artifact"], held)
        r["heldout_macro_f1"] = served[mid]["macro_f1"]
    result["holdout"] = served[state["current"]]
    result["holdout_by_model"] = {mid: {k:v for k,v in score.items() if k != "predictions"}
                                  for mid, score in served.items()}
    atomic_json(folder / "result.json", result)
    print(f"{arm} 完成，留出 Macro-F1 {result['holdout']['macro_f1']:.6f}，续训 {result['total_epochs']} 遍")
    return result
