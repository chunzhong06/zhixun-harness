"""将检查的实际范围、证据和后续选择关联；计划器可说明依据后改变原分支。"""
from copy import deepcopy

from harness.contracts import ActionPlan, digest
from harness.models.corrections import inside


def scoped_inputs(engine, check, router, recent, explore, baseline):
    """范围来自已登记问题，工具只取得该范围内合法行；不从提问文字推断条件。"""
    selected=(router or {}).get("selected",[])
    problem=next((p for p in selected if p["id"]==check.problem_id),None) if check.problem_id else next(
        (p for p in selected if p["model_id"]==baseline["model_id"]),None)
    if check.problem_id and problem is None:
        raise ValueError("检查问题不属于本次选择范围")
    scope=problem.get("scope","overall") if problem else "overall"
    predicates=engine.state.get("problems",{}).get(problem["id"],{}).get("predicates",[]) if problem else []
    def selected_row(row):
        if scope.startswith("group:"): return row.group==scope.removeprefix("group:")
        if scope.startswith("class:"): return str(row.label)==scope.removeprefix("class:")
        if scope.startswith("region:"): return bool(predicates) and inside(row.values,predicates)
        return True
    fit=tuple(r for r in recent if selected_row(r))
    valid=tuple(r for r in explore if selected_row(r))
    if problem and problem.get("observation_window")=="current":
        valid=tuple(r for r in valid if r.sample_id not in engine.state["predictions"])
    evidence={"problem_ids":[problem["id"]] if problem else [],"scope":scope,
        "model_id":baseline["model_id"],"fit_versions":[f"{r.sample_id}@{r.revision}" for r in fit],
        "explore_versions":[f"{r.sample_id}@{r.revision}" for r in valid],
        "observation_window":problem.get("observation_window","all") if problem else "all"}
    evidence["input_digest"]=digest(evidence)
    return fit,valid,{**baseline,"predictions":{r.sample_id:baseline["predictions"][r.sample_id] for r in valid}},evidence


def pending_followups(state, batch_id, facts):
    """每个范围/分支组合有独立后续身份；数值缓存复用不抹去新问题的选择记录。"""
    resolved={e["followup_id"] for e in state["events"] if e["kind"]=="inspection_resolution" and e["batch_id"]==batch_id}
    evidence=next((f["id"] for f in facts["evidence"] if f["kind"]=="investigations"),None)
    return [{"id":r["followup_id"],"check_key":r["check_key"],"scope":deepcopy(r["scope"]),
             "signal":r["signal"],"action":deepcopy(r.get("recommended_action")),"evidence_ref":evidence}
            for r in state["inspections"].get(batch_id,[]) if r.get("followup_id") and r["followup_id"] not in resolved]


def record_resolutions(engine,batch,plan,context,context_key):
    """记录原分支兑现或改变理由；结果只关联选择，实际执行仍由正常工具回执确认。"""
    if plan["operation"]=="inspect":
        return
    actions={ActionPlan.parse(a).key for a in plan.get("actions",[])}
    for item in context["maintenance"].get("inspection_followups",[]):
        recommended=ActionPlan.parse(item["action"]).key if item["action"] else None
        outcome="followed" if recommended in actions else "changed_plan" if recommended else "signal_unavailable"
        engine._event(context_key+":followup:"+item["id"],"inspection_resolution",batch_id=batch.batch_id,
            followup_id=item["id"],check_key=item["check_key"],scope=item["scope"],outcome=outcome,
            recommended_action=recommended,selected_actions=sorted(actions),operation=plan["operation"],
            reason=plan["reason"],evidence_refs=plan["evidence_refs"],context_key=context_key)
