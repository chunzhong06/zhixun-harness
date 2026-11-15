"""维护轮次的工作选择：检查、试改、方法复核和结束共用事实与执行记录。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from harness.contracts import ActionPlan
from harness.methods.diagnosis.investigation import CheckPlan
from harness.methods.planning.planning_facts import validate_action_context, validate_action_evidence


def parse_decision(value):
    """解析一种工作及其参数；原因与证据属于本次决定，不能混入另一种操作。"""
    if not isinstance(value, dict) or not {"operation", "reason", "evidence_refs"} <= set(value):
        raise ValueError("决定需要operation、reason和evidence_refs")
    fields = {"inspect": {"checks", "model_id"}, "experiment": {"actions"},
              "recheck": {"method_id"}, "finish": set()}
    kind = value["operation"]
    if not isinstance(kind, str) or kind not in fields or set(value) != {"operation", "reason", "evidence_refs"} | fields[kind]:
        raise ValueError("工作选择及字段不匹配")
    reason = ActionPlan.parse({"recipe": "recent", "reason": value["reason"], "evidence_refs": value["evidence_refs"]})
    if not reason.reason.strip():
        raise ValueError("决定需要具体理由")
    result = {"operation": kind, "reason": reason.reason, "evidence_refs": list(reason.evidence_refs)}
    if kind == "experiment":
        if not isinstance(value["actions"], list) or not 1 <= len(value["actions"]) <= 3:
            raise ValueError("试改需要一至三项完整动作")
        result["actions"] = [ActionPlan.parse(item).to_record() for item in value["actions"]]
    elif kind == "inspect":
        if not isinstance(value["checks"], list) or not 1 <= len(value["checks"]) <= 4:
            raise ValueError("检查需要一至四项工具计划")
        if not isinstance(value["model_id"], str) or not value["model_id"]:
            raise ValueError("检查需要明确被观察的模型")
        result.update(model_id=value["model_id"], checks=[asdict(CheckPlan.parse(item)) for item in value["checks"]])
    elif kind == "recheck":
        if not isinstance(value["method_id"], str) or not value["method_id"]:
            raise ValueError("复核需要方法身份")
        result["method_id"] = value["method_id"]
    return result


def validate_decision(plan, context):
    """按当时开放的工具、已评价模型及同范围方法核对决定；采用答案不在上下文内。"""
    validate_action_evidence(ActionPlan("recent", reason=plan["reason"], evidence_refs=tuple(plan["evidence_refs"])), context["facts"])
    maintenance = context["maintenance"]
    followups=maintenance.get("inspection_followups",[])
    if followups and plan["operation"]!="inspect" and not {f["evidence_ref"] for f in followups}.intersection(plan["evidence_refs"]):
        raise ValueError("检查后的选择须引用检查事实，并在reason说明采用原分支或改选的依据")
    if plan["operation"] == "experiment":
        for action in plan["actions"]:
            validate_action_context(ActionPlan.parse(action), context)
    elif plan["operation"] == "inspect":
        router = maintenance.get("problem_router")
        if router and not router["allow_inspection"]:
            raise ValueError("本轮问题深查名额已用完或当前没有待深查问题")
        if router and plan["model_id"] not in {i["model_id"] for i in router["selected"]}:
            raise ValueError("问题深查须对应所选问题的观察模型")
        if not maintenance["available_checks"] or plan["model_id"] not in maintenance["decision"]["inspectable_models"]:
            raise ValueError("被检查模型没有当前探索结果或检查未开放")
        if len(plan["checks"]) > (4 if maintenance.get("inspection_order")=="decision_coverage" else maintenance["max_checks"]):
            raise ValueError("超过单轮检查上限")
        for check in plan["checks"]:
            if check.get("problem_id") and not any(p["id"]==check["problem_id"] and p["model_id"]==plan["model_id"]
                                                  for p in (router or {}).get("selected",[])):
                raise ValueError("检查问题与本次观察模型不对应")
            if check["tool"] not in maintenance["available_checks"]:
                raise ValueError("检查工具未开放")
            for branch in ("if_true", "if_false"):
                validate_action_context(ActionPlan.parse(check[branch]), context)
    elif plan["operation"] == "recheck":
        if not any(m["id"] == plan["method_id"] for m in maintenance["methods"]):
            raise ValueError("复核方法不在当前任务和观察范围内")


def recheck_actions(plan, context):
    """复核同资料下沿用当前配方的重训与原方法；两者均重新拟合。"""
    method = next(m for m in context["maintenance"]["methods"] if m["id"] == plan["method_id"])
    candidate = deepcopy(method["action"])
    candidate.update(reason=plan["reason"], evidence_refs=plan["evidence_refs"], method_id=None)
    control = {**candidate, "pipeline": None, "reason": "方法复核的共同资料重训对照"}
    return [ActionPlan.parse(control), ActionPlan.parse(candidate)]
