"""把三个公开任务交给同一驱动与逐批接收入口。演练读取官方输入，候选预测预先固定。"""
from hashlib import file_digest
import json
import os
from pathlib import Path

from engineering.cli.study.run_rul_maintenance import PACKAGES, PROTOCOL, partition, run as run_rul
from harness.contracts import EngineConfig, digest
from harness.data.official_gas import read_official_gas
from harness.evaluation.final_answers import prepare_answers, score_final
from harness.integrations.official_driver import bind_driver
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.methods.planning.gas_continuation import GasContinuationPlanner, GasPairPlanner
from harness.methods.planning.rul_continuation import RulDevicePlanner
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.engine import Harness
from harness.runtime.handoff import OfficialBatchEndpoint, run_stream
from harness.runtime.storage import LocalStore, atomic_json
from harness.testing.gas_continuation import GasRehearsalTools, rehearsal_inputs
from harness.testing.rul_continuation import RulReplayTools, device_stream

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/official"
GAS_FILE = "cpu-a-guide-20261110-v1.zip"


def rehearsal(name, output, mode, *, limit=10, data_root=DATA):
    if mode not in {"rule", "mock"} or not 1 <= limit <= 10:
        raise ValueError("演练仅使用本地规则或离线服务，轮数在1到10之间")
    archive, expected = (data_root / name / GAS_FILE, PACKAGE_SHA) if name == "gas" else (
        data_root / name / PACKAGES[name][0], PACKAGES[name][1])
    with archive.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("当前文件与已核对的官方任务包不符")
    if name == "gas":
        package = read_official_gas(archive)
        batches, tables = rehearsal_inputs(package)
        tools = GasRehearsalTools(package, tables, output / "models", batches)
        rule, fit = GasContinuationPlanner(), package.train
    else:
        package = PACKAGES[name][2](archive)
        roles, assignment = partition(package, json.loads(PROTOCOL.read_bytes()))
        fit = roles["fit"]
        batches, tables = device_stream(package, fit, assignment)
        tools = RulReplayTools(package, fit, tables, output / "models", check_devices=True)
        rule = RulDevicePlanner()
    planner = bind_driver(rule, output, mode=mode)
    config = EngineConfig(policy="plan_feedback", recent_cap=len(fit), max_trials=1,
                          max_plan_calls=1)
    engine = Harness(package.task, config, tools, planner, LocalStore(output / "workflow"))
    endpoint = OfficialBatchEndpoint(batches, output / "accepted", limit)
    state, stream = run_stream(engine, endpoint, output / "handoff")
    rounds = []
    previous = next(mid for mid, model in state["models"].items() if model["parent_id"] is None)
    for ack in stream["accepted"]:
        bid = ack["batch_id"]
        done = state["completed"][bid]
        decision = done["decision"]
        if decision["current_model"] != previous or decision["selected_model"] != done["model_id"]:
            raise ValueError("确认版本与实际上轮交接不符")
        envelope = json.loads((endpoint.output / (ack["request_id"] + ".json")).read_bytes())["payload"]
        if envelope["decision"] != decision or envelope["predictions"] != done["predictions"]:
            raise ValueError("本地接收器与工作流的实际提交不一致")
        for trial in done["trials"]:
            if trial["status"] != "completed" or state["models"][trial["model_id"]]["parent_id"] != previous:
                raise ValueError("候选未完成或没有从在用版本开始")
            artifact, _ = tools.artifacts.read(state["models"][trial["model_id"]]["artifact"])
            if artifact["request"]["parent"] != state["models"][previous]["artifact"]:
                raise ValueError("产物内部记录的父检查点不符")
        contexts = [e["context"] for e in state["events"] if e["key"] == f"{bid}:plan_context:0"]
        plan_op = state["operations"].get(f"{bid}:plan:0")
        if len(contexts) != 1 or not plan_op or plan_op["status"] != "completed":
            raise ValueError(f"{bid} 缺少成功的当轮驱动记录：{(plan_op or {}).get('error')}")
        calls = plan_op.get("service_calls", [])
        if mode == "mock":
            exchanges = [entry for entry in planner.audit if entry["batch_id"] == bid]
            if (len(exchanges) != 1 or len(calls) != 1 or exchanges[0]["status"] != "completed"
                or exchanges[0]["real_model_called"] or exchanges[0]["request_id"] != calls[0]["request_id"]
                or exchanges[0]["plan"] != plan_op["value"] or exchanges[0]["context_digest"] != digest(contexts[0])):
                raise ValueError("驱动请求、返回动作与当轮账目不一致")
        rounds.append({"batch": bid, "current": previous, "selected": done["model_id"],
                       "actions": plan_op["value"], "candidates": [t["model_id"] for t in done["trials"]],
                       "demo_units": state["ledger"][bid]["demo_units"], "prediction_digest": digest(done["predictions"]),
                       "service_calls": len(calls), "mock_tokens": sum(c["usage"]["total_tokens"] for c in calls),
                       "receipt_id": ack["request_id"],
                       "driver_request": calls[0]["request_id"] if calls else None})
        previous = done["model_id"]
    receipt = {"task": name, "mode": mode, "input": endpoint.identity, "driver": planner.identity,
               "model_experiment": False, "rounds": rounds, "completed": len(rounds) == len(batches)}
    if receipt["completed"]:
        # 服务答案此时才进入评价入口。气体另外有按原演练协议在后续轮到达的反馈。
        ids = {s.sample_id for b in batches for s in b.samples if s.role == "serve"}
        answers = [{"sample_id": s.sample_id, "label": s.label} for s in package.validation if s.sample_id in ids]
        final = score_final(package.task, state, prepare_answers(package.task, batches, answers))
        if final["status"] != "completed":
            raise ValueError("实际接收的服务输出与独立答案没有完整对应")
        receipt["scored_outputs"] = final["predicted_samples"]
        if name != "gas":
            receipt["devices"] = {role: sorted({s.unit_id for b in batches for s in b.samples if s.role == role})
                                  for role in ("explore", "adopt", "serve")}
    atomic_json(output / "stream-review.json", receipt)
    return receipt


def compare_drivers(task, folder, data_root=DATA):
    """比较直接规则与经服务协议返回的同一个决定；从第三轮恢复时原回执应逐字保持。"""
    direct = rehearsal(task, folder / "rule", "rule", data_root=data_root)
    prefix = rehearsal(task, folder / "mock", "mock", limit=3, data_root=data_root)
    mock = rehearsal(task, folder / "mock", "mock", data_root=data_root)
    if mock["rounds"][:len(prefix["rounds"])] != prefix["rounds"] or mock["input"] != direct["input"]:
        raise ValueError("恢复更改了既有交接，或两组起始输入不相同")
    exchanges = json.loads((folder / "mock/driver-exchanges.json").read_bytes())["exchanges"]
    if len(exchanges) != 10 or len({e["request_id"] for e in exchanges}) != 10:
        raise ValueError("十轮演练应恰好各调用一次离线驱动")
    if not mock["completed"] or mock["scored_outputs"] != direct["scored_outputs"]:
        raise ValueError("两个入口没有在相同服务范围完成评价")
    common = ("batch", "current", "selected", "actions", "candidates", "demo_units", "prediction_digest")
    for a, b in zip(direct["rounds"], mock["rounds"], strict=True):
        if any(a[k] != b[k] for k in common):
            raise ValueError(f"{a['batch']} 的规则与服务传输结果不一致")
    # 取真实运行当时已保存的输入逐个过协议，核对原始决定。此步不加载工业模型。
    base = ROOT / "results/experiments" / f"official-{task}"
    history = base / ("pilot-11/sequence-protected/state.json" if task == "gas" else "maintenance-1/workflow/state.json")
    saved = json.loads(history.read_bytes())
    rule = GasPairPlanner("shape-blend") if task == "gas" else RulDevicePlanner()
    replay = bind_driver(rule, None, mode="mock")
    reproduced = []
    for event in saved["events"]:
        if event["kind"] == "plan_context":
            context = {**event["context"], "allowed_recipes": {"recent": [0]}}
            operation = saved["operations"][event["key"].replace(":plan_context:", ":plan:")]
            if operation["status"] != "completed" or replay.propose(context) != operation["value"]:
                raise ValueError(f"{event['key']} 与冻结实测的改法不一致")
            reproduced.append({"batch": context["batch_id"], "plan": operation["value"],
                               "source_context": digest(event["context"])})
    report = {"task": task, "basis": "official_input_with_predefined_predictions",
              "input": mock["input"], "direct_driver": direct["driver"], "offline_driver": mock["driver"],
              "shared_decisions_and_predictions": True, "service_outputs": mock["scored_outputs"],
              "device_roles": mock.get("devices"), "rounds": mock["rounds"],
              "archived_decisions": reproduced, "archive": history.relative_to(ROOT).as_posix(),
              "mock_tokens": sum(row["mock_tokens"] for row in mock["rounds"]),
              "real_calls": 0, "actual_fits": 0}
    atomic_json(folder / "comparison.json", report)
    return report


def main(arguments):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("gas", "bearing", "engine", "all"))
    parser.add_argument("--data-root", type=Path, default=DATA, help="只读的三个官方包目录，含 gas、bearing、engine 子目录")
    parser.add_argument("--output", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--driver", choices=("rule", "mock", "service"), default="mock")
    choice.add_argument("--compare-drivers", action="store_true", help="在两个子目录验证同一规则及中断接续")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--rul-series", action="store_true", help="按寿命十轮对照协议运行；默认只核对输入和真实父权重")
    parser.add_argument("--enable-real-execution", action="store_true")
    parser.add_argument("--enable-service-network", action="store_true")
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(arguments)
    if args.task == "all":
        # 镜像默认使用三个正式输入和预写预测核对交接；权限与预算按单项显式执行入口控制。
        if args.enable_real_execution or args.enable_service_network or args.model_config or args.driver == "service" or args.rul_series or args.compare_drivers:
            parser.error("all 用于一次离线验收三项正式输入；实测和对照分别指定任务")
        results = {}
        for task in ("gas", "bearing", "engine"):
            results[task] = rehearsal(task, args.output / task, args.driver,
                                      limit=10 if args.stop_after is None else args.stop_after, data_root=args.data_root)
        atomic_json(args.output / "three-task-receipt.json", {
            "complete": all(r["completed"] for r in results.values()), "model_experiment": False,
            "tasks": {k: {"input": r["input"], "driver": r["driver"], "rounds": len(r["rounds"]),
                          "outputs_scored": r.get("scored_outputs")} for k, r in results.items()}})
        print(args.output / "three-task-receipt.json")
        return
    if args.compare_drivers:
        if args.enable_real_execution or args.enable_service_network or args.model_config or args.stop_after is not None or args.rul_series:
            parser.error("驱动比较使用固定十轮离线条件")
        compare_drivers(args.task, args.output, args.data_root)
        print(args.output / "comparison.json")
        return
    net = args.driver == "service"
    if net != args.enable_service_network or net != bool(args.model_config) or (net and not args.enable_real_execution):
        parser.error("真实驱动须同时启用正式工具、网络和核定模型配置")
    if args.rul_series and (args.task == "gas" or args.stop_after is not None):
        parser.error("寿命十轮对照须使用轴承或发动机完整任务")
    if not args.enable_real_execution and not args.rul_series:
        review = rehearsal(args.task, args.output, args.driver, limit=10 if args.stop_after is None else args.stop_after,
                           data_root=args.data_root)
        print(f"{args.task} 已确认 {len(review['rounds'])} 轮离线交接。")
        return
    if args.stop_after is not None:
        parser.error("局部交接选项用于工程演练；实测遵守各任务冻结协议")
    driver = {"mode": args.driver}
    if net:
        driver.update(config=PyTrioConfig(**json.loads(args.model_config.read_bytes())),
                      allow_network=True, api_key=os.environ.get("PYTRIO_API_KEY"))
    if args.rul_series:
        from engineering.cli.study.run_rul_series import run
        print(run(args.task, args.output, real=args.enable_real_execution, driver=driver, data_root=args.data_root))
    elif args.task == "gas":
        from harness.evaluation.gas_pilot import run_arm
        protocol = ROOT / "engineering/configs/experiments/official-gas-pilot.json"
        run_arm(args.data_root / "gas" / GAS_FILE, protocol, args.output, "static")
        run_arm(args.data_root / "gas" / GAS_FILE, protocol, args.output, "adaptive", driver=driver)
    else:
        print(run_rul(args.task, args.output, real=True, driver=driver, data_root=args.data_root))
