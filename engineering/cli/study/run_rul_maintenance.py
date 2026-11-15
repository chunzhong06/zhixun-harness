"""冻结按整台设备隔离的寿命维护对照；显式执行才启动官方 CPU 续训。"""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.evaluate_rul_baseline import PACKAGES, describe
from harness.contracts import ActionPlan, BatchContext, EngineConfig, digest
from harness.methods.planning.rul_continuation import RulDevicePlanner, device_biases
from harness.integrations.official_driver import bind_driver
from harness.models.model_artifacts import ModelArtifacts
from harness.models.engine_manifests import FIRST, LAST, EngineFitPlanner, EngineFitTools, fit_lists, ids_sha
from harness.models.official_rul_cpu import RulPilotTools
from harness.runtime.engine import Harness
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.storage import LocalStore, atomic_json

PROTOCOL = ROOT / "engineering/configs/experiments/official-rul-device-maintenance.json"


def partition(package, protocol):
    """按先定设备身份分配用途，划分不读取预测或寿命答案。训练时可取完整训练轨迹。"""
    by_unit = defaultdict(list)
    for row in package.train:
        by_unit[row.unit_id].append(row)
    fit = []
    for group in by_unit.values():
        ordered = sorted(group, key=lambda r: r.time_end)
        count = min(protocol["training_observations_per_device"], len(ordered))
        indices = [i * (len(ordered) - 1) // (count - 1) for i in range(count)] if count > 1 else [0]
        fit.extend(ordered[i] for i in indices)
    strata = defaultdict(set)
    for row in package.validation:
        stratum = row.unit_id.split(":")[0] if package.task.target_unit == "cycles" else "bearing"
        strata[stratum].add(row.unit_id)
    assignment = {}
    for stratum, units in strata.items():
        if len(units) < 3:
            raise ValueError(f"{stratum} 缺少分开探索、确认和服务所需的设备")
        for index, uid in enumerate(sorted(units, key=lambda u: digest([protocol["seed"], u]))):
            assignment[uid] = ("explore", "adopt", "serve")[index % 3]
    if set(assignment) & set(by_unit):
        raise ValueError("验证设备越入训练集")
    rows = {"fit": tuple(fit), "explore": [], "adopt": [], "serve": []}
    for row in package.validation:
        role = assignment[row.unit_id]
        if role == "serve" or (row.time_end - 1) % protocol["development_observation_stride"] == 0:
            rows[role].append(replace(row, role=role))
    return {k: tuple(v) for k, v in rows.items()}, assignment


def runtime_batch(task, rows):
    # 工作流收到每台服务设备的第一条无答案观测；决定冻结后，全部服务观测再用同一版本评价。
    live = tuple(replace(r, label=None) for r in rows["serve"] if r.time_end == 1)
    return BatchContext(task.task_id, "maintenance-1", 0, rows["fit"] + rows["explore"] + rows["adopt"] + live)


def compare(task, rows, old, new):
    units = defaultdict(list)
    for row in rows:
        units[row.unit_id].append(row)
    before, after = describe(task, rows, old), describe(task, rows, new)
    per_device = {u: {"original": describe(task, member, old), "candidate": describe(task, member, new)}
                  for u, member in units.items()}
    return {"original": before, "candidate": after, "devices": per_device,
            "device_equal_mean_rmse": {arm: fmean(v[arm]["rmse"] for v in per_device.values())
                                       for arm in ("original", "candidate")}}


def engine_windows(task, rows, old, new):
    """分别保留目标时段和两侧表现，先用于确认，再用同一口径评价已冻结决定。"""
    windows = {}
    for name, lower, upper in (("early", 1, FIRST - 1), ("target", FIRST, LAST), ("later", LAST + 1, float("inf"))):
        subset = tuple(r for r in rows if lower <= r.time_end <= upper)
        if subset:
            windows[name] = compare(task, subset, old, new)
    return windows


def run(task_name, output, real=False, *, driver=None, data_root=None, fit_policy=None, input_reader=None):
    """input_reader 只用于检查模式，代替能启动训练进程的运行器。旧协议沿用原计划器。"""
    if fit_policy is not None and (task_name != "engine" or (driver and driver["mode"] != "rule")):
        raise ValueError("本次清单对照用于发动机单次维护的本地计划器")
    if input_reader is not None and (real or fit_policy is None):
        raise ValueError("只读核验须指定发动机选样策略并关闭真实执行")
    protocol = json.loads(PROTOCOL.read_bytes())
    name, checksum, read = PACKAGES[task_name]
    archive = (data_root or ROOT / "data/official") / task_name / name
    saved = ROOT / f"results/experiments/official-{task_name}/baseline/{task_name}-baseline.json"
    baseline_bytes = saved.read_bytes()
    if sha256(archive.read_bytes()).hexdigest() != checksum or sha256(baseline_bytes).hexdigest() != protocol["baseline_sha"][task_name]:
        raise ValueError("原版预测记录或官方输入发生变化")
    package = read(archive)
    baseline = json.loads(baseline_bytes)
    original = baseline["predictions"]
    if not baseline["matches_official"] or baseline["checkpoint_sha256"] != package.checkpoint_sha:
        raise ValueError("原版基线来源不符")
    rows, roles = partition(package, protocol)
    frozen = {"protocol": protocol, "archive_sha256": checksum, "devices": roles,
              "roles": {k: {"count": len(v), "sample_ids_sha256": digest([r.sample_id for r in v])} for k,v in rows.items()},
              "source": "previously_inspected_public_development"}
    with LocalStore(output).exclusive():
        driver = driver or {"mode": "rule"}
        rule = RulDevicePlanner() if fit_policy is None else EngineFitPlanner(fit_policy)
        planner = bind_driver(rule, output, **driver)
        if fit_policy is not None:
            frozen["fit_policy"] = rule.identity
            frozen["fit_lists"] = {name: ids_sha(row.sample_id for row in members)
                                   for name, members in fit_lists(rows["fit"]).items()}
        if driver["mode"] != "rule":
            frozen["driver_identity"] = planner.identity
        manifest = output / "protocol.json"
        if manifest.exists() and json.loads(manifest.read_bytes()) != frozen:
            raise ValueError("该目录已有其他冻结清单")
        atomic_json(manifest, frozen)
        discovery = device_biases(rows["explore"], original)
        atomic_json(output / "observation.json", {"model": package.checkpoint_sha, "source": "saved_real_predictions",
                    "devices": discovery, "role": "discovery_only"})
        if not real and input_reader is None:
            return manifest
        result_path = output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            if result["protocol_digest"] != digest(frozen):
                raise ValueError("实测结果与冻结输入不符")
            return result_path
        runner_type = input_reader or LocalJobRunner
        runner = runner_type(output / "jobs", ModelArtifacts(output / "models"),
                                ExecutionLimits(90, 60, 768 * 1024**2))
        if fit_policy is None:
            tools = RulPilotTools(task_name, archive, package, rows["fit"], rows["explore"], original, runner)
        else:
            tools = EngineFitTools(archive, package, rows["fit"], rows["explore"], original, runner)
        config = EngineConfig(policy="plan_feedback", execution_mode="real_model", recent_cap=len(rows["fit"]), max_trials=1)
        main = Harness(package.task, config, tools, planner, LocalStore(output / "workflow"),
                       allow_real_execution=True)
        batch = runtime_batch(package.task, rows)
        if input_reader is not None:
            return runner.inspect(main, batch, tools, rows["fit"])
        state = main.run((batch,))
        if any(v["status"] != "completed" for v in state["operations"].values()
               if v["name"].startswith("plan:")):
            raise ValueError("寿命驱动调用未完成，不能将运行失败记作主动保持")
        done = state["completed"][batch.batch_id]
        decision = done["decision"]
        if any(t["status"] not in ("completed",) for t in done["trials"]):
            raise ValueError("维护候选执行不完整，需核对作业回执")
        # 共同固定对照只构建一个候选。自主侧跳过试验时，其反事实成本单独归给固定组。
        initial = tools.initial()
        candidates = [state["models"][t["model_id"]]["artifact"] for t in done["trials"] if t.get("model_id")]
        if fit_policy is not None and not candidates:
            raise ValueError("选样对照未产生登记候选，禁止另行补训")
        candidate = candidates[0] if candidates else tools.build(package.task, batch.batch_id, ActionPlan("recent", epochs=1),
                                                                rows["fit"], parent=initial)
        adopted = state["models"][decision["selected_model"]]["artifact"]
        confirm = rows["adopt"]
        confirm_inputs = [{"sample_id": r.sample_id, "values": list(r.values)} for r in confirm]
        confirm_predictions = tools.predict(candidate, confirm_inputs)
        confirmed = compare(package.task, confirm, original, confirm_predictions)
        fixed_adopts = confirmed["original"]["rmse"] - confirmed["candidate"]["rmse"] > 1e-12
        policies = {"static": "original", "fixed": "candidate",
                    "fixed_confirmed": "candidate" if fixed_adopts else "original",
                    "harness": "candidate" if adopted == candidate else "original"}
        selection_path = output / "selected.json"
        selection = {"protocol_digest": digest(frozen), "policies": policies, "confirmation": confirmed}
        if fit_policy is not None:
            selection["confirmation_windows"] = engine_windows(package.task, confirm, original, confirm_predictions)
        if selection_path.exists() and json.loads(selection_path.read_bytes()) != selection:
            raise ValueError("冻结后的选择与重新读取的结果不一致")
        atomic_json(selection_path, selection)
        held = rows["serve"]
        inputs = [{"sample_id": r.sample_id, "values": list(r.values)} for r in held]
        updated = tools.predict(candidate, inputs)
        fixed_original = {r.sample_id: original[r.sample_id] for r in held}
        chosen = updated if adopted == candidate else fixed_original
        if adopted not in (candidate, initial) or any(abs(chosen[sid] - v) > 1e-8 for sid,v in done["predictions"].items()):
            raise ValueError("实际发出的服务预测与冻结决定不一致")
        paired = compare(package.task, held, fixed_original, updated)
        jobs = [json.loads(p.read_bytes()) for p in (output / "jobs").glob("*/state.json")]
        if any(j["status"] not in ("completed", "failed") for j in jobs):
            raise ValueError("比较还有未结束的作业")
        # 当前比较所需的结果已分别核验。先前失败的工作进程单列，并计入总费用。
        failures = [{k: j.get(k) for k in ("request_id", "failure", "error_message", "cpu_seconds")}
                    for j in jobs if j["status"] == "failed"]
        report = {"protocol_digest": digest(frozen), "decision": decision, "candidate": candidate,
                  "prior_failed_jobs": failures,
                  "parent": initial, "physical_fits": len(list((output / "jobs").glob("*/training.json"))),
                  "policies": policies, "logical_fit_counts": {"static": 0, "fixed": 1, "fixed_confirmed": 1,
                                                               "harness": len(candidates)},
                  "shared_jobs_cpu_seconds": sum(j["cpu_seconds"] for j in jobs),
                  "service_comparison": paired, "harness_service": describe(package.task, held, chosen),
                  "candidate_service_predictions": updated, "service_rows": len(held), "plan": done["trials"]}
        if fit_policy is not None:
            report["fit_policy"] = fit_policy
            report["service_windows"] = engine_windows(package.task, held, fixed_original, updated)
        if report["physical_fits"] != 1:
            raise ValueError("共享预算必须对应唯一的一次续训")
        atomic_json(result_path, report)
        return result_path


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(PACKAGES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-fit-policy", choices=("fixed", "observed"), help="选择首次发动机维护的等量取样策略")
    parser.add_argument("--enable-real-maintenance", action="store_true", help="运行一次本地官方候选续训及比较")
    args = parser.parse_args()
    print(run(args.task, args.output, args.enable_real_maintenance, fit_policy=args.engine_fit_policy))
