"""从已保存的一轮维护还原观察、计划和实际决定，不重新运行计划器或模型。"""
from harness.methods.planning.gas_continuation import gas_issue
from harness.evaluation.metrics import confirmation_scope


def gas_service_coverage(state, batch, service_groups):
    """比较实际生效版本的确认来源和服务来源；只读批次标识，不读服务答案或未来决定。"""
    if state["task"]["task_id"] != "official-gas-A":
        return None  # 其他任务的 group 不一定代表采集批。
    model = state["completed"][batch]["model_id"]
    groups = set(service_groups)
    sources = sorted(g for g in groups if isinstance(g, str) and g.startswith("source-")
                     and g.removeprefix("source-").isdigit())
    report = {"model":model, "service_sources":sources, "confirmation_batch":None,
              "confirmed_sources":[], "status":"unknown"}
    if state["models"][model]["parent_id"] is None:
        return {**report, "status":"original"}
    for event in state["events"]:
        if event["kind"] != "batch_committed":
            continue
        bid = event["batch_id"]
        result = state["completed"][bid]
        d = result["decision"]
        if (result["model_id"] == model == d["nominee"] and model != d["current_model"]
                and not d.get("output_fallback") and d.get("comparison", {}).get("status") == "matched"):
            ops = state["operations"]
            scope = confirmation_scope(state["task"]["task_id"],
                ops.get(f"{bid}:explore:{model}", {}).get("value"), ops.get(f"{bid}:adopt:{model}", {}).get("value"))
            if scope and not scope["overlapping_sample_count"]:
                report.update(confirmation_batch=bid, confirmed_sources=scope["confirmation_sources"])
        if bid == batch:
            break
    else:
        return {**report, "confirmation_batch":None, "confirmed_sources":[]}
    if sources and len(sources) == len(groups) and report["confirmed_sources"]:
        report["status"] = "covered" if set(sources) <= set(report["confirmed_sources"]) else "outside"
    return report


def round_evidence(state, batch, *, service_groups=None):
    result = state["completed"][batch]
    decision = result["decision"]
    incumbent = decision["current_model"]
    observation = {}
    actions = []
    for event in state["events"]:
        if event["key"] == f"{batch}:plan_context:0":
            context = event["context"]
            fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
            observation = {"fact_id": fact["id"], "exploration_score": fact["value"]["score"]}
            if "gas_confusions" in context:
                worst = gas_issue(context)
                observation.update(lowest_f1_class=worst["label"] if worst else None,
                                   confusions=context["gas_confusions"],
                                   active_pairs=context.get("gas_active_pairs", []))
            if "device_bias_issue" in fact["value"]:
                observation["device_bias_issue"] = fact["value"]["device_bias_issue"]
        if event["kind"] == "plan_accepted" and event.get("batch_id") == batch:
            actions.extend(event["actions"])

    # 计划可以未执行，探索领先也可能在确认环节失败，分别保留对应回执。
    if decision.get("output_fallback"):
        outcome = "output_fallback"
    elif result["model_id"] != incumbent:
        outcome = "adopted"
    elif decision.get("candidate_score") is not None:
        outcome = "confirmation_rejected"
    elif decision.get("nominee") != incumbent:
        outcome = "confirmation_unavailable"
    elif any(t["status"] == "completed" for t in result["trials"]):
        outcome = "exploration_kept_incumbent"
    elif actions or result["trials"]:
        outcome = "no_completed_challenger"
    else:
        outcome = "no_trial"
    scope = decision.get("confirmation_scope")
    if scope is None and decision.get("comparison", {}).get("status") == "matched":
        # 老回执保留了分组和样本身份，按当时的候选恢复来源，跳过尚未走到确认的轮次。
        operations = state.get("operations", {})
        nominee = decision["nominee"]
        scope = confirmation_scope(state["task"]["task_id"],
            operations.get(f"{batch}:explore:{nominee}", {}).get("value"),
            operations.get(f"{batch}:adopt:{nominee}", {}).get("value"))
    coverage = gas_service_coverage(state, batch, service_groups) if service_groups is not None else None
    return {**({"confirmation_scope": scope} if scope is not None else {}),
            **({"service_coverage":coverage} if coverage is not None else {}),
            "observation": observation, "planned_actions": actions,
            "trials": [{"action": t["action"], "status": t["status"], "model": t.get("model_id"),
                        "exploration_score": t.get("exploration_score")} for t in result["trials"]],
            "outcome": outcome, "reason": decision["reason"], "current": incumbent,
            "selected": result["model_id"], "confirmation": [decision.get("baseline_score"),
                                                             decision.get("candidate_score")]}
