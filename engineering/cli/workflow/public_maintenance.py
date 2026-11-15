"""队员用公开原包启动一条真实维护流程。训练、决定及交出预测共用同一个 Harness。"""
import argparse
from dataclasses import replace
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys

from engineering.cli.study.run_rul_maintenance import PACKAGES, PROTOCOL, ROOT, partition as partition_rul
from engineering.cli.study.run_rul_series import RegularUpdate, SERIES, windows
from harness.contracts import EngineConfig, digest
from harness.data.official_gas import LABELS, read_official_gas
from harness.evaluation.gas_pilot import partition as partition_gas
from harness.evaluation.round_evidence import round_evidence
from harness.integrations.official_driver import bind_driver
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.methods.planning.gas_continuation import FirstWindowPairPlanner, GasPairPlanner
from harness.methods.planning.rul_continuation import RulDevicePlanner
from harness.methods.planning.error_slice import ErrorSlicePlanner
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import OfficialGasTools, PACKAGE_SHA
from harness.models.official_rul_cpu import RulPilotTools
from harness.models.readout_training import BiasRulTools
from harness.models.rul_selection import CoverageRulTools, SelectableRulTools
from harness.models.trajectory_training import POLICIES as TRAJECTORY_POLICIES, TrajectoryRulTools
from harness.runtime.engine import Harness
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.handoff import OfficialBatchEndpoint, run_stream
from harness.runtime.storage import LocalStore, atomic_json

TASKS = ("gas", "bearing", "engine")
COVERAGE_POLICIES = {"coverage-mixed":"mixed","coverage-targeted":"targeted","coverage-uniform":"uniform"}


class FirstRoundProposal:
    """首批使用现场提案，后续沿固定策略维护。首批身份来自输入流，恢复时保持不变。"""
    def __init__(self, first, fixed, first_batch):
        self.first, self.fixed, self.first_batch = first, fixed, first_batch
        self.identity = digest(["live-first-then-fixed-v1", first.identity, fixed.identity, first_batch])
        if hasattr(first, "audit"):
            self.audit = first.audit
            self.set_runtime_timeout = first.set_runtime_timeout

    def uses_service(self, batch_id):
        return batch_id == self.first_batch and hasattr(self.first, "audit")

    def propose(self, context):
        # 气体固定策略要用首批观察建立排序，即使首批交由另一个提案者决定。
        fallback = self.fixed.propose(context)
        return self.first.propose(context) if context["batch_id"] == self.first_batch else fallback


def gas_service_windows(batches, held):
    """固定分片在维护前形成；服务测量不含答案。运行与事后逐版核对共用此分法。"""
    ordered = sorted(held, key=lambda row: digest(row.sample_id))
    return tuple(replace(batch, samples=batch.samples + tuple(
        replace(row, role="serve", label=None, available_at=batch.ordinal)
        for row in ordered[batch.ordinal::len(batches)])) for batch in batches)


def assemble_public(name, data_root, folder, policy, driver, *, select_training_rows=False, gas_partition=None):
    """只有官方包和源代码是必要输入。服务观测在进入运行状态前移除答案。"""
    if policy=="bearing-stump" and name!="bearing":
        raise ValueError("轴承残差树只用于轴承任务")
    if name == "gas":
        filename, checksum, reader = "cpu-a-guide-20261110-v1.zip", PACKAGE_SHA, read_official_gas
    else:
        filename, checksum, reader = PACKAGES[name]
    archive = data_root / name / filename
    with archive.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != checksum:
            raise ValueError(f"{name} 原包与登记版本不符")
    package = reader(archive)
    if name == "gas":
        protocol = json.loads((ROOT/"engineering/configs/experiments/official-gas-pilot.json").read_bytes())
        partitioner=partition_gas if gas_partition is None else gas_partition
        batches, held = partitioner(package, protocol)
        batches = gas_service_windows(batches, held)
        planner, fit = GasPairPlanner("shape-blend"), package.train
        if policy in ("selected","fixed","first-proposal","readout-bias") or policy in COVERAGE_POLICIES or policy in TRAJECTORY_POLICIES:
            planner = FirstWindowPairPlanner.restore("shape-blend",LocalStore(folder).load(),batches[0].batch_id)
    else:
        protocol = json.loads(SERIES.read_bytes())
        rows, assignment = partition_rul(package, json.loads(PROTOCOL.read_bytes()))
        fit = rows["fit"]
        batches = windows(package, assignment, fit, protocol, name)
        planner = RegularUpdate() if policy in ("fixed","first-proposal","readout-bias","bearing-stump") or policy in TRAJECTORY_POLICIES or (policy=="selected" and name=="engine") else RulDevicePlanner()
    limits = ExecutionLimits(protocol["worker_wall_seconds"], protocol["worker_cpu_seconds"], protocol["worker_memory_bytes"])
    runner = LocalJobRunner(folder/"jobs", ModelArtifacts(folder/"models"), limits)
    if name == "gas":
        tools = OfficialGasTools(archive, package, batches, runner, reuse=True, pair_method="shape-blend",
                                 sequence="protected" if policy in ("selected","fixed","error-slice","first-proposal","readout-bias") or policy in COVERAGE_POLICIES or policy in TRAJECTORY_POLICIES else None)
    else:
        kwargs = {"batches":batches, "reuse_identical_fit":name == "bearing"}
        if policy in COVERAGE_POLICIES:
            tools = CoverageRulTools(name, archive, package, fit, (), None, runner,
                                     allocation=COVERAGE_POLICIES[policy], **kwargs)
        elif policy == "readout-bias":
            tools = BiasRulTools(name, archive, package, fit, (), None, runner, **kwargs)
        elif policy == "bearing-stump":
            from harness.models.bearing_stumps import StumpTools
            tools = StumpTools(name, archive, package, fit, (), None, runner, **kwargs)
        elif policy in TRAJECTORY_POLICIES:
            tools = TrajectoryRulTools(name, archive, package, fit, (), None, runner,
                                       recipe=TRAJECTORY_POLICIES[policy], **kwargs)
        else:
            tool_type = SelectableRulTools if select_training_rows or policy in ("selected","error-slice","first-proposal") else RulPilotTools
            tools = tool_type(name, archive, package, fit, (), None, runner, **kwargs)
    if policy == "error-slice" or (policy in COVERAGE_POLICIES and name != "gas"):
        if driver["mode"] != "rule":
            raise ValueError("按当批错误排序是本地提案策略，不能同时绑定模型提案器")
        planner = ErrorSlicePlanner()
    config = EngineConfig(execution_mode="real_model", policy="plan_feedback", recent_cap=len(fit),
                          maintenance_trigger="never" if policy == "original" else "every_batch",
                          max_plan_calls=1, max_trials=1, worker_cpu_seconds=limits.cpu_seconds,
                          worker_memory_bytes=limits.memory_bytes)
    if policy == "first-proposal":
        first = bind_driver(ErrorSlicePlanner(), folder, **driver)
        planner = FirstRoundProposal(first, planner, batches[0].batch_id)
    else:
        planner = bind_driver(planner, folder, **driver)
    engine = Harness(package.task, config, tools, planner,
                     LocalStore(folder), allow_real_execution=True)
    return engine, batches


def run_task(name, data_root, folder, policy, limit, driver, *, select_training_rows=False):
    engine, batches = assemble_public(name, data_root, folder, policy, driver, select_training_rows=select_training_rows)
    return run_prepared(name,folder,policy,limit,driver,engine,batches)


def local_policy(folder, requested, *, default="selected"):
    """新目录使用实测推荐；已开始的目录恢复最初的策略，包括没有策略标记的旧运行。"""
    marker = folder/"public-policy.json"
    if marker.exists():
        previous = json.loads(marker.read_bytes())["policy"]
    elif (folder/"run.json").exists():
        previous = json.loads((folder/"run.json").read_bytes())["policy"]
    elif (folder/"state.json").exists():
        state = LocalStore(folder).load()
        previous = "original" if state["config"]["maintenance_trigger"]=="never" else "rule"
    else:
        previous = None
    if previous is not None and requested is not None and previous!=requested:
        raise ValueError("该目录已用另一策略运行；比较不同策略请使用单独目录")
    chosen = requested or previous or default
    if not marker.exists():
        atomic_json(marker,{"policy":chosen})
    return chosen


def run_prepared(name, folder, policy, limit, driver, engine, batches, *, count_rejected_proposals=False):
    """公开包与外部对照均用同一版本交接、服务提交和回执核验。"""
    endpoint = OfficialBatchEndpoint(batches, folder/"accepted", limit)
    saved = LocalStore(folder/"handoff").load()
    if saved and len(saved["accepted"]) > limit:
        raise ValueError("该目录已交接更多轮次。缩短回放请使用新的输出目录。")
    state, stream = run_stream(engine, endpoint, folder/"handoff")
    rejected = []
    for op in state["operations"].values():
        if op["status"] == "completed":
            continue
        if (count_rejected_proposals and op["status"] == "failed" and op["name"] == "plan:0"
                and op.get("error_code") == "invalid_action_context"):
            rejected.append(op["batch_id"])
            continue
        raise RuntimeError(f"{name} 有未成功的作业，详情见 {folder / 'jobs'}")
    entries = []
    previous = next(mid for mid, model in state["models"].items() if model["parent_id"] is None)
    service_groups = {b.batch_id:{s.group for s in b.samples if s.role == "serve" and s.available_at == b.ordinal}
                      for b in batches}
    for ack in stream["accepted"]:
        bid = ack["batch_id"]
        result = state["completed"][bid]
        evidence = round_evidence(state, bid, service_groups=service_groups[bid])
        if bid in rejected and (evidence["current"] != evidence["selected"] or evidence["trials"]):
            raise ValueError("被拒的唯一提案不能产生候选或改变在用版本")
        submitted = folder/"accepted"/(ack["request_id"]+".json")
        payload = json.loads(submitted.read_bytes())["payload"]
        if (evidence["current"] != previous or payload["model_id"] != evidence["selected"] or
                payload["decision"] != result["decision"] or payload["predictions"] != result["predictions"] or
                not payload["predictions"]):
            raise ValueError(f"{name} {bid} 的版本、决定和提交内容不一致")
        previous = evidence["selected"]
        sid, value = next(iter(payload["predictions"].items()))
        entries.append({"batch":bid, **evidence, "model_artifact":state["models"][previous]["artifact"],
                        "predictions":len(payload["predictions"]), "first_prediction":{"id":sid, "value":value},
                        "submitted":submitted.relative_to(folder).as_posix()})
    # 支持从已完成的轮次继续。离线实验记录只有在事件序列没有改变时才复用。
    key = digest([state["fingerprint"], [a["request_id"] for a in stream["accepted"]]])
    run = folder/"run.json"
    prior = json.loads(run.read_bytes()) if run.is_file() else {}
    tracking = folder/"swanlab"/key
    verified = tracking/"verification.json"
    if prior.get("receipt_key") != key or not verified.is_file() or not json.loads(verified.read_bytes()).get("verified"):
        subprocess.run([sys.executable, "-B", str(ROOT/"engineering/cli/verify/verify_offline_tracking.py"),
                        "--run", str(folder), "--output", str(tracking)], check=True)
    report = {"task":name, "policy":policy, "driver":driver["mode"], "evidence_kind":state["evidence_kind"], "receipt_key":key,
              "completed_rounds":len(entries), "total_rounds":len(batches), "rounds":entries,
              "output_labels":list(LABELS) if name == "gas" else None,
              "target_unit":engine.task.target_unit, "swanlab":verified.relative_to(folder).as_posix()}
    if count_rejected_proposals:
        report["rejected_proposals"] = rejected
    atomic_json(run, report)
    return report


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*TASKS, "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT/"data/official")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=range(1,11), default=10,
                        help="运行至第几轮；扩大同目录的轮数会从上次决定接续")
    parser.add_argument("--policy", choices=("rule","original","fixed","selected","error-slice","first-proposal","readout-bias","bearing-stump",*COVERAGE_POLICIES,*TRAJECTORY_POLICIES),
                        help="新本地发动机运行默认 first-proposal，气体和轴承默认 selected。可指定 first-proposal 只在首批现场提案或 readout-bias 测寿命输出偏差校正。")
    parser.add_argument("--model-config", type=Path, help="指定服务配置；与显式网络开关一起使用")
    parser.add_argument("--enable-service-network", action="store_true")
    parser.add_argument("--proposer",choices=("default","simple","self-healing"),default="default",
                        help="选择项目规则、简单提案或按误差诊断提案的论文适配")
    parser.add_argument("--proposal-inbox",type=Path,help="本地模型提案的请求和回复目录")
    parser.add_argument("--select-training-rows",action="store_true",help="寿命续训允许按训练池分位数选样")
    parser.add_argument("--gas-feedback",type=int,help="本地气体每轮交付后返回的标签数，启用按混淆率合并的预测")
    parser.add_argument("--gas-total-feedback",type=int,help="本地气体整条流的回标总额，在每轮上限内尽早使用")
    from harness.models.field_classification import MODES
    from harness.models.site_training import MODES as TRAINING_MODES
    from harness.models.served_scale_gas import MODE as SERVED_SCALE
    from harness.models.coverage_scale_gas import MODE as COVERAGE_SCALE
    parser.add_argument("--gas-field-mode",choices=(*MODES,*TRAINING_MODES,SERVED_SCALE,COVERAGE_SCALE),help="气体现场选版或指定固定工具，需配置每轮回标")
    parser.add_argument("--engine-input-repair",action="store_true",help="用源训练输入学习组合，纠正唯一可还原的不一致读数")
    engine_mode=parser.add_mutually_exclusive_group()
    engine_mode.add_argument("--engine-early-reference",action="store_true",help="发动机用源设备开工读数作工况参照，含原输入修复")
    engine_mode.add_argument("--engine-source-choice",action="store_true",help="先用完整源设备留出选择发动机工具，再按固定权重交付")
    engine_mode.add_argument("--engine-full-condition",action="store_true",help="用源设备开工工况处理全部源记录，从原权重续训发动机预测器")
    engine_mode.add_argument("--engine-continued-condition",action="store_true",help="从十遍发动机全量工况权重再续训五十遍")
    engine_mode.add_argument("--engine-training-inquiry",choices=("adaptive","fixed-order"),
                             help="由真实先导结果安排下一场训练对照，或使用固定试验顺序")
    parser.add_argument("--bearing-phase",action="store_true",help="轴承使用仅由源训练建立的阶段树和固定寿命尺度")
    parser.add_argument("--bearing-checkpoint",type=Path,help="轴承实验分支交付冻结的赛方格式权重；路径与摘要写入恢复配置")
    args = parser.parse_args(arguments)
    if args.bearing_checkpoint and (args.bearing_phase or args.task not in ("bearing","all") or
            args.policy not in (None,"selected") or args.proposer!="default" or args.model_config or
            args.enable_service_network or args.select_training_rows):
        parser.error("轴承冻结权重需与本地 selected 来源配合，且不同时使用阶段树")
    if args.bearing_phase and (args.task not in ("bearing","all") or args.policy not in (None,"selected") or
            args.proposer!="default" or args.model_config or args.enable_service_network or args.select_training_rows):
        parser.error("轴承阶段预测需在 bearing 或 all 的本地 selected 运行中使用")
    if (args.engine_input_repair or args.engine_early_reference or args.engine_source_choice or args.engine_full_condition or args.engine_continued_condition or args.engine_training_inquiry) and (args.task not in ("engine","all") or args.policy not in (None,"first-proposal") or
            args.proposer!="default" or args.model_config or args.enable_service_network or args.select_training_rows):
        parser.error("发动机输入修复需与本地首轮维护配合，可在 engine 或 all 运行")
    if args.gas_total_feedback is not None and (args.gas_total_feedback<0 or args.gas_feedback is None):
        parser.error("气体总回标额需要非负整数及每轮回标上限")
    if args.gas_field_mode is not None and args.task not in ("gas","all"):
        parser.error("气体现场选版需要气体任务")
    if args.gas_feedback is not None and (args.gas_feedback<0 or args.task not in ("all","gas") or
            args.policy not in (None,"selected") or args.proposer!="default" or args.model_config or args.enable_service_network):
        parser.error("气体回标适用于本地 selected 气体运行，反馈数量须非负")
    if args.policy=="bearing-stump" and args.task!="bearing":
        parser.error("bearing-stump 需要指定 bearing 任务")
    if args.enable_service_network != bool(args.model_config) or (args.enable_service_network and args.policy == "original"):
        parser.error("服务驱动需要配置文件、网络开关和启用维护的策略")
    driver = {"mode":"rule"}
    model_proposal = args.proposer in ("simple","self-healing")
    guidance = "self_healing" if args.proposer=="self-healing" else None
    if args.proposal_inbox is not None:
        if not model_proposal or args.model_config or args.policy=="original":
            parser.error("本地提案目录需要模型提案、启用维护且不使用远程服务")
        driver={"mode":"local-proposal","simple_proposal":True,"inbox":args.proposal_inbox.resolve(),
                "proposal_guidance":guidance}
    if model_proposal and not (args.proposal_inbox or args.model_config):
        parser.error("模型提案需要本地提案目录或已配置的模型服务")
    if args.model_config:
        service = PyTrioConfig(**json.loads(args.model_config.read_bytes()))
        api_key = os.environ.get("PYTRIO_API_KEY")
        if service.base_url.rstrip("/") != PyTrioConfig.base_url or not api_key or not api_key.strip():
            parser.error("需要已核对的比赛服务地址和账户凭据")
        driver = {"mode":"service", "config":service, "allow_network":True, "api_key":api_key,
                  "simple_proposal":model_proposal,"proposal_guidance":guidance}
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        parser.error("公开输入和运行输出需要独立目录")
    tasks = TASKS if args.task == "all" else (args.task,)
    with LocalStore(output/"public-run-lock").exclusive():
        receipt = {}
        for name in tasks:
            folder=output/name
            checkpoint_config=folder/"checkpoint-config.json"
            if name=="bearing" and (args.bearing_checkpoint is not None or checkpoint_config.exists()):
                from engineering.cli.workflow.serve_life_checkpoint import run as deliver_checkpoint
                # 配置先于运行持久化；恢复时按实际摘要核对，防止同目录悄悄换权重。
                previous=json.loads(checkpoint_config.read_bytes()) if checkpoint_config.exists() else None
                weights=args.bearing_checkpoint.resolve() if args.bearing_checkpoint else Path(previous["weights"])
                with weights.open("rb") as stream:
                    checksum=file_digest(stream,"sha256").hexdigest()
                config={"weights":str(weights),"sha256":checksum}
                if previous is not None and config!=previous:
                    parser.error("该目录已冻结另一份轴承权重或权重内容已变")
                if args.bearing_phase or args.policy not in (None,"selected") or driver["mode"]!="rule" or args.select_training_rows:
                    parser.error("轴承冻结权重恢复需保持本地 selected 来源")
                local_policy(folder,"native-fixed")
                atomic_json(checkpoint_config,config)
                source=folder/"regular"
                run_task("bearing",data_root,source,"selected",args.rounds,driver)
                archive=data_root/"bearing"/PACKAGES["bearing"][0]
                report=deliver_checkpoint("bearing",archive,weights,source,folder,args.rounds)
                served=sum(r["rows"] for r in report["accepted"])
                receipt[name]={"rounds":len(report["accepted"]),"report":"bearing/run.json","policy":"native-fixed",
                               "predictions":served,"weights_sha256":checksum}
                print(f"bearing 已按冻结单权重交付 {served} 条预测。")
                continue
            observer_config=folder/"feedback-budget.json"
            feedback=args.gas_feedback
            total_feedback=args.gas_total_feedback
            field_mode=args.gas_field_mode
            if name=="gas" and observer_config.exists():
                previous=json.loads(observer_config.read_bytes())
                configured=previous["feedback_per_round"]
                if feedback is not None and feedback!=configured:
                    parser.error("该运行目录的气体回标预算与本次请求不同")
                if total_feedback is not None and total_feedback!=previous.get("total_feedback"):
                    parser.error("该运行目录的气体回标总额与本次请求不同")
                feedback=configured
                total_feedback=previous.get("total_feedback")
                if field_mode is not None and field_mode!=previous.get("field_mode"):
                    parser.error("当前目录已固定另一气体选版方式")
                field_mode=previous.get("field_mode")
            if name=="gas" and field_mode is not None and feedback is None:
                parser.error("气体现场选版需要 --gas-feedback 或已有回标运行目录")
            if name=="gas" and feedback is not None:
                from engineering.cli.workflow import gas_public_feedback
                if driver["mode"]!="rule" or args.policy not in (None,"selected"):
                    parser.error("已有气体回标运行需要本地 selected 策略")
                local_policy(folder,"observer")
                config={"feedback_per_round":feedback}
                if field_mode is not None:
                    config["field_mode"]=field_mode
                if total_feedback is not None:
                    config["total_feedback"]=total_feedback
                atomic_json(observer_config,config)
                report=gas_public_feedback.run(data_root,None,folder,17,feedback,args.rounds,total_feedback,field_mode=field_mode)
                emitted=0
                for event in report["accepted"]:
                    payload=json.loads((folder/"accepted"/(event["request"]+".json")).read_bytes())["payload"]
                    emitted+=len(payload["predictions"])
                receipt[name]={"rounds":len(report["accepted"]),"report":f"{name}/run.json","policy":field_mode or "observer",
                               "predictions":emitted,**config}
                print(f"气体已交付 {emitted} 条预测，第 {len(report['accepted'])} 轮快照 {payload['snapshot']}。")
                continue
            default = "first-proposal" if name=="engine" and driver["mode"]=="rule" else "selected"
            requested="source-combinations" if name=="engine" and args.engine_input_repair else args.policy
            if name=="engine" and args.engine_early_reference:
                requested="early-reference"
            if name=="engine" and args.engine_source_choice:
                requested="source-tool-choice"
            if name=="engine" and args.engine_full_condition:
                requested="full-condition-native"
            if name=="engine" and args.engine_continued_condition:
                requested="continued-condition-native"
            if name=="engine" and args.engine_training_inquiry:
                requested="training-inquiry" if args.engine_training_inquiry=="adaptive" else "training-inquiry-fixed"
            if name=="bearing" and args.bearing_phase:
                requested="phase-fixed"
            policy = local_policy(folder,requested,default=default)
            if (name=="engine" and policy in ("source-combinations","early-reference","source-tool-choice","full-condition-native","continued-condition-native","training-inquiry","training-inquiry-fixed")) or (name=="bearing" and policy=="phase-fixed"):
                from engineering.cli.workflow import engine_reference
                ordinary="first-proposal" if name=="engine" else "selected"
                if args.policy not in (None,ordinary) or driver["mode"]!="rule" or args.select_training_rows:
                    parser.error("该运行的寿命参照需要此前的本地规则及原训练选择")
                harness,batches=assemble_public(name,data_root,folder/"regular",ordinary,driver)
                result=engine_reference.run_prepared(data_root,folder,harness,batches,args.rounds,policy=policy,task=name)
                served=sum(sum(r.role=="serve" and r.available_at==b.ordinal for r in b.samples) for b in batches[:args.rounds])
                receipt[name]={"rounds":result["completed_rounds"],"report":f"{name}/run.json","policy":policy,"predictions":served}
                print(f"{name} 已按 {policy} 交付 {served} 条预测。")
                continue
            if policy=="observer":
                parser.error("该目录使用气体回标方案，请带上原 --gas-feedback 数量继续")
            if policy=="original" and driver["mode"]!="rule":
                parser.error("始终使用原版的目录没有提案操作")
            result = run_task(name, data_root, output/name, policy, args.rounds, driver,
                              select_training_rows=args.select_training_rows)
            receipt[name] = {"rounds":result["completed_rounds"], "report":f"{name}/run.json","policy":policy,
                             "predictions":sum(r["predictions"] for r in result["rounds"])}
            last = result["rounds"][-1]
            example = last["first_prediction"]
            value = LABELS[example["value"]-1] if name == "gas" else f"{example['value']:.2f} {result['target_unit']}"
            print(f"{name} 第 {len(result['rounds'])} 轮使用 {last['selected']}。{last['reason']}。"
                  f"样本 {example['id']} 的预测为 {value}。详见 {output / name / 'run.json'}")
        policies = {entry["policy"] for entry in receipt.values()}
        atomic_json(output/f"public-run-{args.task}.json", {"driver":driver["mode"], "policy":next(iter(policies)) if len(policies)==1 else "mixed",
                    "evidence_kind":"real_model", "tasks":receipt})
