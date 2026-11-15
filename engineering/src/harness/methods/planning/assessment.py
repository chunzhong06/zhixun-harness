"""把执行证据与预测比较整理成下一次计划的依据，不代替模型做原因推断。"""
from __future__ import annotations

from copy import deepcopy
import math
from harness.evaluation.metrics import compare_evaluations


def assess_attempt(task, trial, build, evaluation, baseline):
    """分别记录是否执行完整、当前观察如何及哪些判断还缺证据。

    同批探索差值支持安排下一步，最终采用由公共评价决定。没有训练曲线
    时不推断“训练不足”；异常类别来自工具回执，不把运行失败解释成方法无效。
    """
    execution = build.get("execution") or {}
    result = evaluation.get("value") if evaluation.get("status") == "completed" else None
    change = trial.get("action_plan") or build.get("requested_action")
    assessment = {"hypothesis": change.get("reason") if change else None,
        "requested_action": deepcopy(change), "execution_status": build.get("status", "not_required"),
        "effective_action": deepcopy(build.get("effective_action")),
        "proposal_origin": deepcopy(trial.get("proposal_origin")),
        "fit_condition": deepcopy(build.get("fit_condition")),
        "progress": deepcopy(execution.get("progress", [])[-16:]),
        "failure_category": execution.get("failure") or (build.get("error") if build.get("status") == "failed" else None),
        "observation": "comparison_unavailable", "score": None, "utility_delta": None,
        "training_sufficiency": "unmeasured", "objective": task.objective}
    if build.get("fit_condition", {}).get("ready") is False:
        assessment.update(observation="fit_conditions_not_met", execution_status="not_started",
                          next_focus="根据已选资料的类别与特征条件调整资料配方或组件，再安排构建")
    elif build.get("status") == "failed":
        assessment.update(observation="execution_incomplete", next_focus="依据执行阶段和错误核对实现或调整投入，再决定是否重试")
    elif evaluation.get("status") == "failed":
        assessment.update(observation="evaluation_incomplete", next_focus="先核对预测与评价结果是否完整")
    elif result and baseline:
        assessment["comparison"] = compare_evaluations(task, baseline, result)
        assessment["error_profiles"] = {"baseline": deepcopy(baseline.get("error_profile")),
                                        "candidate": deepcopy(result.get("error_profile"))}
        gain = task.utility(result["score"]) - task.utility(baseline["score"])
        assessment["score"] = result["score"]
        assessment["utility_delta"] = gain if math.isfinite(gain) else None
        if assessment["utility_delta"] is not None:
            assessment.update(observation="exploration_improved" if gain > 0 else "no_exploration_improvement",
                next_focus="根据本次比较与剩余额度，选择继续调整、安排共同采用或结束尝试")
    if "next_focus" not in assessment:
        assessment["next_focus"] = "先取得可比结果，再判断该次改法"
    return assessment
