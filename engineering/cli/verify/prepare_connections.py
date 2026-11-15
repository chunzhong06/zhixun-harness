"""预检表格资料或运行A包接入形状与PyTRiO回复形状的离线样例。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]

from harness.contracts import EngineConfig
from harness.runtime.engine import Harness
from harness.testing.fixtures import (FixtureModelTools, FixtureTrainingBackend, OfflineChatTransport,
                              SimulatedHttpService, SimulatedSwanLab)
from harness.models.model_artifacts import ModelArtifacts
from harness.models.training_tools import TrainingModelTools
from harness.runtime.process_tools import ProcessTrainingTools
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.data.package_input import read_package
from harness.integrations.pytrio_planner import PyTrioConfig, PyTrioPlanner
from harness.runtime.storage import LocalStore, atomic_json
from harness.integrations.service_transport import HttpChatTransport
from harness.integrations.tracking import SwanLabRecorder
from engineering.cli.workflow.run_example import load_example


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "rehearse"])
    parser.add_argument("--manifest", type=Path, default=root / "engineering/tests/fixtures/packages/package_layout/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["plan_once", "plan_feedback", "plan_feedback_reuse"], default="plan_once")
    parser.add_argument("--model-config", type=Path, default=root / "engineering/tests/fixtures/services/pytrio_config.json")
    parser.add_argument("--example", type=Path, default=root / "engineering/tests/fixtures/tasks/classification.json",
                        help="与输入清单任务匹配的预写预测样例，仅供离线演练")
    parser.add_argument("--simulate-services", action="store_true",
                        help="用内存HTTP服务和模拟SwanLab验证接入，不读密钥、不联网")
    parser.add_argument("--persist-models", action="store_true",
                        help="同时检查完整样例模型的文件保存与重新加载")
    parser.add_argument("--complete-workflow", action="store_true",
                        help="接入主动检查、组件改法与方法经验；仍使用人工计划和预写预测")
    parser.add_argument("--worker-process", action="store_true", help="在独立进程中运行预写训练替身，记录资源并限制耗时")
    parser.add_argument("--worker-timeout", type=float, default=30., help="单个工作进程的工程时限，单位秒")
    args = parser.parse_args()
    package = read_package(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "inspect":
        atomic_json(args.output / "输入预检.json", package.inspection)
        print(f"资料预检完成：{len(package.batches)}批；未调用计划器或模型。")
        return
    if package.origin != "interface_fixture":
        raise ValueError("实际资料当前只允许inspect预检；联通演练需要接口样例")
    fixture_task, _, fixture = load_example(args.example)
    if fixture_task != package.task:
        raise ValueError("资料与预测样例的任务、指标或单位不一致")
    config = PyTrioConfig(**json.loads(args.model_config.read_text(encoding="utf-8")))
    transport = (HttpChatTransport(config, "fixture-only-key", wire=SimulatedHttpService())
                 if args.simulate_services else OfflineChatTransport())
    planner = PyTrioPlanner(config, transport, args.output / "对话接口记录.json")
    # 分类与连续目标共用接入口，模型仍由明确选择的预写预测表替代。
    tools = (TrainingModelTools(package.task, FixtureTrainingBackend(fixture), ModelArtifacts(args.output / "models"))
             if args.persist_models else FixtureModelTools(fixture))
    if args.worker_process:
        artifacts = ModelArtifacts(args.output / "models")
        runner = LocalJobRunner(args.output / "jobs", artifacts, ExecutionLimits(wall_seconds=args.worker_timeout))
        tools = ProcessTrainingTools(package.task, FixtureTrainingBackend(fixture), artifacts, runner)
    engine = Harness(package.task, EngineConfig(policy=args.policy, component_actions=args.complete_workflow,
        active_inspection=args.complete_workflow, method_memory=args.complete_workflow,
        adaptive_loop=args.complete_workflow), tools, planner, LocalStore(args.output))
    recorder = (SwanLabRecorder(args.output / "tracking", sdk=SimulatedSwanLab(
        args.output / "tracking/simulated-sdk.json")) if args.simulate_services else None)
    expected = engine.run_identity
    receipt = None
    try:
        state = engine.run(package.batches)
    finally:
        # 运行结束或中断都从持久状态导出；补交记录不需要再次调用计划器。
        saved = engine.store.load()
        if recorder and saved and saved["fingerprint"] == expected:
            receipt = recorder.sync(saved, planner.audit)
            print(f"模拟记录已整理：{receipt['cursor']}条；云端上传=0。")
    if receipt and receipt["segments"] and receipt["segments"][-1]["status"] == "failed":
        raise RuntimeError("模拟记录导出未完成；本地记录已保留，重跑可补交")
    atomic_json(args.output / "输入预检.json", package.inspection)
    print(f"离线接入样例完成：{len(state['completed'])}批，方式={args.policy}。")
    print("实际模型调用=0；使用预写预测和模拟服务，未读取密钥。")
    print(f"请求与解析记录：{(args.output / '对话接口记录.json').resolve()}")


if __name__ == "__main__":
    main()
