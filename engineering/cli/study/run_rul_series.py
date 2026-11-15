"""两项公开寿命任务的十轮回放。准备模式只加载权重与编制清单，实测另有显式入口。"""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from engineering.cli.study.run_rul_maintenance import ROOT, PACKAGES, PROTOCOL, partition, describe
from harness.contracts import ActionPlan, BatchContext, EngineConfig, digest
from harness.evaluation.final_answers import prepare_answers, score_final
from harness.evaluation.round_evidence import round_evidence
from harness.integrations.official_driver import bind_driver
from harness.methods.planning.rul_continuation import RulDevicePlanner
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import RulPilotTools, execute_official_job
from harness.runtime.engine import Harness
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.handoff import OfficialBatchEndpoint, run_stream
from harness.runtime.storage import LocalStore, atomic_json

SERIES = ROOT / "engineering/configs/experiments/official-rul-series.json"


def windows(package, assignment, fit, spec, task_name):
    """已完结档案可按全程分段。服务设备按当前时刻发放，切点独立于它的最终寿命。"""
    count, cap = spec["rounds"], spec["archive_points_per_device_per_round"]
    width = spec["service_window_width"][task_name]
    by_unit = defaultdict(list)
    for row in package.validation:
        by_unit[row.unit_id].append(row)
    groups = [[] for _ in range(count)]
    for unit, members in sorted(by_unit.items()):
        ordered = sorted(members, key=lambda r: r.time_end)
        role = assignment[unit]
        if role == "serve":
            for row in ordered:
                i = min((row.time_end - 1) // width, count - 1)
                groups[i].append(replace(row, role=role, label=None, available_at=i))
            continue
        for i in range(count):
            block = ordered[i * len(ordered) // count:(i + 1) * len(ordered) // count]
            size = min(cap, len(block))
            if size < 4:
                raise ValueError("档案分段无法提供辨认持续偏差所需的四次观测")
            for j in range(size):
                row = block[j * (len(block) - 1) // (size - 1)]
                groups[i].append(replace(row, role=role, available_at=i))
    return tuple(BatchContext(package.task.task_id, f"round-{i+1}", i,
                              (fit if i == 0 else ()) + tuple(block)) for i, block in enumerate(groups))


class RegularUpdate:
    identity = "rul-every-round-shared-confirmation-v1"

    def propose(self, context):
        fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
        plan = ActionPlan("recent", epochs=1, reason="按固定日程从在用版本续训一遍，使用共同确认条件。",
                          evidence_refs=(fact["id"],))
        return {"actions": [plan.to_record()], "stop": False}


def inspect_handoff(task_name, archive, package, fit, original, batches, limits):
    """把上次实测的原版和候选分别作为示例父权重，走正式工作者的加载和选数路径。"""
    previous = ROOT / f"results/experiments/official-{task_name}/maintenance-1"
    saved = json.loads((previous / "result.json").read_bytes())
    report = json.loads((previous / "jobs" / saved["candidate"]["key"] / "training.json").read_bytes())
    models = ModelArtifacts(previous / "models")
    with TemporaryDirectory(dir=ROOT / "scratch", prefix="rul-parent-") as tmp:
        work = Path(tmp)
        runner = LocalJobRunner(work, models, limits)
        tools = RulPilotTools(task_name, archive, package, fit, (), original, runner, batches=batches)
        readings = {}
        for label, key, round_index in (("original", "parent", 0), ("candidate", "candidate", 1)):
            batch = batches[round_index]
            req = tools.continuation_request(package.task, batch.batch_id, ActionPlan("recent", epochs=1),
                                             fit, saved[key])
            receipt = execute_official_job(work, {"request": req, "backend": tools.worker}, models, inspect_resume=True)
            prefix = "initial" if label == "original" else "final"
            expected_sha = report["parent_checkpoint_sha256" if label == "original" else "checkpoint_sha256"]
            if (receipt["epochs"] != report[prefix + "_epochs"] or receipt["steps"] != report[prefix + "_steps"]
                    or "sha256:" + receipt["parent_sha256"] != expected_sha
                    or "sha256:" + receipt["train_sha256"] != report["train_sha256"]):
                raise ValueError("正式接续入口与已发布的真实训练记录不符")
            readings[label] = receipt
    first, second = readings.values()
    if (first["depth"], second["depth"]) != (0, 1) or first["explore_sha256"] == second["explore_sha256"]:
        raise ValueError("两轮应分别绑定原版、候选和各自的探索档案")
    for k in ("mean", "scale"):
        if first["tensors"][k] != second["tensors"][k]:
            raise ValueError("接续时丢失了原标准化状态")
    return readings


def evaluate(package, batches, state, answers):
    """只用各版本在其生效时实际发出的预测；逐轮和逐台误差保留各自的样本分母。"""
    final = score_final(package.task, state, answers)
    if final["status"] != "completed":
        raise ValueError("实际发出的服务预测未完整覆盖冻结范围")
    predicted = {r["sample_id"]: r["prediction"] for r in final["records"]}
    served = {r.sample_id: r for r in package.validation if r.sample_id in predicted}
    by_device = defaultdict(list)
    for row in served.values():
        by_device[row.unit_id].append(row)
    devices = {uid: describe(package.task, rows, predicted) for uid, rows in by_device.items()}
    rounds = []
    previous = next(mid for mid, m in state["models"].items() if m["parent_id"] is None)
    for batch in batches:
        result = state["completed"][batch.batch_id]
        decision = result["decision"]
        if decision["current_model"] != previous or result["model_id"] != decision["selected_model"]:
            raise ValueError("发生跨轮跳版")
        for trial in result["trials"]:
            if trial["status"] != "completed" or state["models"][trial["model_id"]]["parent_id"] != previous:
                raise ValueError("未取得已确认父版本上的完整候选")
        rows = [served[s.sample_id] for s in batch.samples if s.role == "serve"]
        reused = sum(e["kind"] == "fit_reused" and e.get("batch_id") == batch.batch_id for e in state["events"])
        rounds.append({"batch": batch.batch_id, "current": previous, "selected": result["model_id"],
                       "decision": decision, "candidates": len(result["trials"]),
                       "reused_fits": reused, "round_evidence": round_evidence(state, batch.batch_id),
                       "service": describe(package.task, rows, predicted)})
        previous = result["model_id"]
    return {"overall": describe(package.task, list(served.values()), predicted),
            "per_device": devices, "device_equal_rmse": fmean(v["rmse"] for v in devices.values()),
            "rounds": rounds, "worst_round_rmse": max(r["service"]["rmse"] for r in rounds),
            "candidate_evaluations": sum(r["candidates"] for r in rounds),
            "candidate_epochs": sum(r["candidates"] - r["reused_fits"] for r in rounds),
            "source_digest": final["source_digest"]}


def run_arm(package, batches, fit, folder, tools, arm, limits, *, driver=None, real=False):
    """固定与自主策略只改变何时试改；交接、探索提名与跨设备确认沿用同一实现。"""
    planner = (bind_driver(RulDevicePlanner(), folder, **(driver or {"mode": "rule"}))
               if arm == "harness" else RegularUpdate())
    config = EngineConfig(execution_mode=tools.evidence_kind, policy="plan_feedback", recent_cap=len(fit),
                          maintenance_trigger="never" if arm == "static" else "every_batch",
                          max_trials=1, max_plan_calls=1, worker_cpu_seconds=limits.cpu_seconds,
                          worker_memory_bytes=limits.memory_bytes)
    engine = Harness(package.task, config, tools, planner, LocalStore(folder / "workflow"), allow_real_execution=real)
    state, handoff = run_stream(engine, OfficialBatchEndpoint(batches, folder / "accepted"), folder / "handoff")
    if len(handoff["accepted"]) != len(batches) or any(op["status"] != "completed" for op in state["operations"].values()):
        raise ValueError(f"{arm} 的维护或提交未完成，需核对作业回执")
    return planner.identity


def run(task_name, output, *, real=False, driver=None, data_root=None):
    directory = ROOT / f"results/experiments/official-{task_name}"
    baseline_path = directory / "baseline" / f"{task_name}-baseline.json"
    once = json.loads(PROTOCOL.read_bytes())
    spec = json.loads(SERIES.read_bytes())
    if spec["max_candidate_epochs_per_round"] != 1 or spec["arms"] != ["static", "fixed_confirmed", "harness"]:
        raise ValueError("该对照需要相同的一遍训练预算以及三个预定策略")
    filename, checksum, reader = PACKAGES[task_name]
    archive = (data_root or ROOT / "data/official") / task_name / filename
    if sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError("寿命任务官方包已变化")
    package = reader(archive)
    original = None
    if baseline_path.is_file() or not real:
        baseline = baseline_path.read_bytes()
        if sha256(baseline).hexdigest() != once["baseline_sha"][task_name]:
            raise ValueError("已登记的原版预测文件发生变化")
        original = json.loads(baseline)["predictions"]
    rows, assignment = partition(package, once)
    fit = rows["fit"]
    batches = windows(package, assignment, fit, spec, task_name)
    limits = ExecutionLimits(spec["worker_wall_seconds"], spec["worker_cpu_seconds"], spec["worker_memory_bytes"])
    sets, released = [], set()
    for batch in batches:
        ids = {s.sample_id for s in batch.samples}
        if len(ids) != len(batch.samples) or ids & released:
            raise ValueError("十轮输入有重复身份")
        released.update(ids)
        scope = {}
        for role in ("explore", "adopt", "serve"):
            members = [s for s in batch.samples if s.role == role]
            if not members or any(assignment[s.unit_id] != role or (s.label is None) != (role == "serve")
                                  or s.available_at != batch.ordinal for s in members):
                raise ValueError("公开档案的当轮用途或答案状态不一致")
            scope[role] = {"rows": len(members), "devices": len({s.unit_id for s in members}),
                           "ids": digest([s.sample_id for s in members])}
        sets.append({"batch": batch.batch_id, **scope})
    ids = {s.sample_id for b in batches for s in b.samples if s.role == "serve"}
    if ids != {r.sample_id for r in package.validation if assignment[r.unit_id] == "serve"}:
        raise ValueError("服务输出遗漏了冻结设备或时刻")
    frozen = {"definition": spec, "archive": checksum, "baseline": once["baseline_sha"][task_name],
              "training": digest(sorted(s.sample_id for s in fit)), "training_count": len(fit),
              "roles": assignment, "windows": sets}
    if original is None:
        # 新机器可直接使用官方原权重逐批预测。该模式的费用与复用既有预测分开比较。
        frozen.update(baseline=package.checkpoint_sha, original_predictions="runtime_official_checkpoint")
    output = Path(output)
    with LocalStore(output).exclusive():
        path = output / "protocol.json"
        if path.exists() and json.loads(path.read_bytes()) != frozen:
            raise ValueError("目录已经用于不同的寿命回放")
        atomic_json(path, frozen)
        if not real:
            readings = inspect_handoff(task_name, archive, package, fit, original, batches, limits)
            atomic_json(output / "prepared.json", {"task": task_name, "protocol_digest": digest(frozen),
                        "loaded": readings, "fits": 0, "predictions": 0, "service_calls": 0})
            return output / "prepared.json"
        runner = LocalJobRunner(output / "jobs", ModelArtifacts(output / "models"), limits)
        tools = RulPilotTools(task_name, archive, package, fit, (), original, runner, batches=batches,
                              reuse_identical_fit=(task_name == "bearing"))
        drivers = {arm: run_arm(package, batches, fit, output / arm, tools, arm, limits, driver=driver, real=True)
                   for arm in spec["arms"]}
        # 三种策略都已提交全部预测，才把服务标签交给评价器。
        answers = prepare_answers(package.task, batches, [{"sample_id": r.sample_id, "label": r.label}
                                  for r in package.validation if r.sample_id in ids])
        results = {}
        for arm in spec["arms"]:
            state = LocalStore(output / arm / "workflow").load()
            result = evaluate(package, batches, state, answers)
            counts = [r["candidates"] for r in result["rounds"]]
            if (any(n > 1 for n in counts) or arm == "fixed_confirmed" and counts != [1] * len(batches)
                    or arm == "static" and any(counts)):
                raise ValueError("候选数量与该组的预算或固定日程不一致")
            ops = [op["execution"] for op in state["operations"].values() if op.get("execution")]
            result["logical_worker_cpu_seconds"] = sum(r["cpu_seconds"] for r in ops)
            result.update(driver=drivers[arm], workflow=state["fingerprint"])
            results[arm] = result
        jobs = [json.loads(p.read_bytes()) for p in (output / "jobs").glob("*/state.json")]
        atomic_json(output / "comparison.json", {"protocol_digest": digest(frozen), "arms": results,
                    "physical_fits": len(list((output / "jobs").glob("*/training.json"))),
                    "physical_worker_cpu_seconds": sum(j["cpu_seconds"] for j in jobs),
                    "failed_jobs": [j["request_id"] for j in jobs if j["status"] == "failed"]})
        return output / "comparison.json"
