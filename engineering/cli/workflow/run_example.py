"""运行一个任务的接口样例；不训练模型、不连接外部服务。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# 从仓库内运行时使用同一份源码，工程演练不要求先安装项目包。
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]

from harness.contracts import BatchContext, EngineConfig, TaskSpec
from harness.runtime.engine import Harness
from harness.testing.fixtures import FixtureModelTools, FixturePlanner, FixtureTrainingBackend
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.storage import LocalStore
from harness.models.training_tools import TrainingModelTools


def load_example(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("evidence_kind") != "interface_fixture":
        raise ValueError("该入口只接受明确标记的接口样例")
    task = TaskSpec.parse(value["task"])
    return task, tuple(BatchContext.parse(b) for b in value["batches"]), value["model_fixture"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["plan_once", "plan_feedback", "plan_feedback_reuse"], default="plan_once")
    parser.add_argument("--example", type=Path, default=Path(__file__).resolve().parents[3] / 'engineering/tests/fixtures/tasks/classification.json')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=24, help="预设接口测试额度，非真实算力费用")
    parser.add_argument("--persist-models", action="store_true",
                        help="将预写流水线保存到输出目录models，并由文件重载；仍不拟合模型")
    args = parser.parse_args()
    task, batches, fixture = load_example(args.example)
    config = EngineConfig(policy=args.policy, demo_budget_per_batch=args.budget)
    tools = (TrainingModelTools(task, FixtureTrainingBackend(fixture), ModelArtifacts(args.output / "models"))
             if args.persist_models else FixtureModelTools(fixture))
    state = Harness(task, config, tools, FixturePlanner(), LocalStore(args.output)).run(batches)
    print("接口样例已完成；未训练模型或调用Qwen。")
    for bid, result in state["completed"].items():
        print(f"{bid}：{result['decision']['reason']}；后续版本={result['model_id']}")
    print(f"流程记录：{(args.output / '流程说明.md').resolve()}")


if __name__ == "__main__":
    main()
