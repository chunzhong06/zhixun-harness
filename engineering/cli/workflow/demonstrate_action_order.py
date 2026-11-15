"""保存气体任务形式的预写维护过程；不读取密钥、不联网、不拟合模型。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.contracts import EngineConfig
from harness.methods.search.action_selection import ActionOrderPlanner
from harness.runtime.engine import Harness
from harness.runtime.storage import LocalStore, atomic_json
from harness.testing.fixtures import FixtureModelTools
from harness.testing.gas_scenario import gas_order_scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/engineering/action-order/runs")
    args = parser.parse_args(argv)
    task, batches, fixture = gas_order_scenario()
    config = EngineConfig(policy="plan_feedback", component_actions=True, adaptive_loop=True,
                          max_trials=3, max_plan_calls=4, demo_budget_per_batch=60)
    engine = Harness(task, config, FixtureModelTools(fixture), ActionOrderPlanner(), LocalStore(args.output))
    state = engine.run(batches)
    contexts = [e for e in state["events"] if e["kind"] == "plan_context" and e["context"]["batch_id"] == "gas1"]
    choices = [e for e in state["events"] if e["kind"] == "work_selected" and e["batch_id"] == "gas1"]
    summary = {"evidence_kind": "interface_fixture", "inputs": "人工构造，沿UCI气体类别与128列契约",
               "model_training": False, "real_model_called": False, "model_performance_claim": False,
               "steps": [{"signals": e["context"]["maintenance"]["action_options"]["signals"],
                           "observation_model": e["context"]["maintenance"]["action_options"]["observation_model"],
                           "decision": choices[i]["decision"]} for i, e in enumerate(contexts)],
               "adoption": state["completed"]["gas1"]["decision"],
               "later_feedback": [v for v in state["feedback"].values() if v["source"] == "service_feedback"]}
    atomic_json(args.output / "scenario.json", summary)
    lines = ["# 气体场景的改法选择过程", "", "人工构造输入与预写预测，仅核对决策和反馈过程；没有训练、模型调用或效果实验。", ""]
    for i, step in enumerate(summary["steps"], 1):
        action = step["decision"].get("actions", [{}])[0]
        lines += [f"{i}. {step['decision']['reason']}",
                  f"   资料配方：{action.get('recipe')}，历史比例：{action.get('ratio')}；实际计划和观察见scenario.json。", ""]
    lines += ["共同采用：" + summary["adoption"]["reason"] + "。",
              "实际采用版本：" + summary["adoption"]["selected_model"] + "。", "",
              "随后发回的答案已与原服务模型和预测关联，见scenario.json的later_feedback。"]
    (args.output / "流程说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("气体形式的预写过程已保存；未训练或调用真实模型。")
    return summary


if __name__ == "__main__":
    main()
