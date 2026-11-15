"""根据已执行结果选择后续工作；共同采用仍由Harness在循环结束后完成。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from harness.contracts import ActionPlan, digest
from harness.methods.planning.decisions import parse_decision, recheck_actions, validate_decision
from harness.methods.diagnosis.investigation import CheckPlan, check_identity, select_checks
from harness.methods.diagnosis.error_regions import frozen_regions
from harness.methods.search.action_selection import proposal_origin
from harness.methods.search.search_policy import trial_routing


def _event(engine, key):
    return next((e for e in engine.state["events"] if e["key"] == key), None)


def _request(engine, batch, context, index):
    """已收到的格式错误可有限修正；未知外部结果或传输故障不自动重复请求。

    修正占用共同调用数和额度，每次都有独立且先保存的上下文。恢复时
    读取原快照，后续发现不能回写之前的请求。
    """
    repair = None
    for attempt in range(engine.config.max_plan_repairs + 1):
        suffix = str(index) if attempt == 0 else f"{index}:repair:{attempt}"
        name, key = f"plan:{suffix}", f"{batch.batch_id}:plan:{suffix}"
        context_key = f"{batch.batch_id}:plan_context:{suffix}"
        saved = _event(engine, context_key)
        call_context = deepcopy(saved["context"] if saved else context)
        if not saved:
            call_context["remaining_demo_units"] = engine.config.demo_budget_per_batch - engine.state["ledger"][batch.batch_id]["demo_units"]
            call_context["maintenance"]["decision"]["repair"] = repair
        calls = sum(k.startswith(f"{batch.batch_id}:plan:") for k in engine.state["operations"])
        if key not in engine.state["operations"] and calls >= engine.config.max_plan_calls:
            return None, "plan_call_limit"
        engine._event(context_key, "plan_context", context=call_context)
        reply = engine._operation(batch, name, 1, 3, lambda: engine.planner.propose(deepcopy(call_context)), context_key=context_key)
        operation = engine.state["operations"].get(key, {})
        if reply is None:
            if not operation:
                return None, "budget_reserved_for_output"
            if not operation.get("plan_repairable"):
                return None, "plan_unavailable"
            error = operation["error_code"]
        else:
            try:
                plan = parse_decision(reply)
                validate_decision(plan, call_context)
                from harness.methods.diagnosis.inspection_followup import record_resolutions
                record_resolutions(engine,batch,plan,call_context,context_key)
                engine._event(f"{batch.batch_id}:work_selected:{suffix}", "work_selected", batch_id=batch.batch_id,
                              context_key=context_key, decision=plan)
                return (plan, call_context, context_key), None
            except (ValueError, TypeError) as exc:
                error = str(exc)
        engine._event(f"{batch.batch_id}:plan_invalid:{suffix}", "plan_invalid", reason=error,
                      context_key=context_key, repair_attempt=attempt)
        # 返回内容作为有界数据提供给计划器；不把它拼入系统指令。
        repair = {"operation_key": key, "error": error,
                  "previous_response": str(reply)[:4000] if reply is not None else None}
    return None, "plan_repair_limit"


def _run_actions(engine, batch, current, recent, history, explore, actions, trials, scores, tried, manifests, context_key, method_id=None):
    """试改共用在用参照与探索资料；配方来源可不同，完成操作按原身份复用。"""
    new_work = 0
    context = _event(engine, context_key)["context"]
    for action in actions:
        if action.key in tried or len(tried) >= engine.config.max_trials:
            continue
        tried.add(action.key)
        rows = engine._select_rows(batch, action, recent, history)
        manifest = engine._training_manifest(action, rows, current.model_id)
        if not rows or manifest in manifests:
            trials.append({"action": action.key, "action_plan": action.to_record(),
                           "status": "skipped_duplicate_or_empty", "model_id": None, "training_ids": []})
            continue
        manifests.add(manifest)
        model = engine._build(batch, action, rows, current.model_id)
        result = engine._score(batch, model, explore, "explore", 3) if model else None
        trial = {"action": action.key, "status": "completed" if result else "unavailable",
                 "model_id": model.model_id if model else None, "exploration_score": result["score"] if result else None,
                 "training_ids": [s.sample_id for s in rows], "action_plan": action.to_record(),
                 "proposal_origin": proposal_origin(context, action),
                 **trial_routing(context, action),
                 "parent_id": current.model_id, "training_versions": [f"{s.sample_id}@{s.revision}" for s in rows],
                 "evaluation_versions": [f"{s.sample_id}@{s.revision}" for s in explore]}
        if method_id is not None and action.pipeline is not None:
            trial["recheck_method_id"] = method_id
        trials.append(trial)
        if f"{batch.batch_id}:build:{action.key}" in engine.state["operations"]:
            new_work += 1
        if result:
            scores[model.model_id] = engine.task.utility(result["score"])
        engine._event(f"{context_key}:trial:{action.key}", "trial_observed", trial=trial)
    return new_work


def _run_checks(engine, batch, recent, explore, plan, context_key):
    """检查可观察已评价候选；相同工具及输入重用数值，但保留本次问题和后续选择。"""
    operation = engine.state["operations"][f"{batch.batch_id}:explore:{plan['model_id']}"]
    baseline = operation["value"]
    new_work = 0
    existing = engine.state["inspections"].setdefault(batch.batch_id, [])
    context = _event(engine, context_key)["context"]
    router = context.get("maintenance", {}).get("problem_router")
    selection_key = f"{context_key}:inspection_order"
    saved_selection = _event(engine,selection_key)
    if saved_selection:
        checks = [CheckPlan.parse(raw) for raw in saved_selection["selected_plans"]]
    else:
        candidates = [CheckPlan.parse(raw) for raw in plan["checks"]]
        checks, ordering = select_checks(candidates,engine.task,recent,explore,plan["model_id"],engine.state,engine.config,
                                         len([e for e in engine.state["events"] if e["kind"]=="work_round_completed"]))
        if router:
            checks = checks[:router["remaining_checks"]] if router["allow_inspection"] else []
        engine._event(selection_key, "inspection_order", criteria=ordering,
                      selected=[check.tool for check in checks],selected_plans=[asdict(c) for c in checks])
    for i, check in enumerate(checks):
        from harness.methods.diagnosis.inspection_followup import scoped_inputs
        fit_rows,eval_rows,observed,scope=scoped_inputs(engine,check,router,recent,explore,baseline)
        history = frozen_regions(engine.state, batch.batch_id) if check.tool == "error_regions" else ()
        identity = check_identity(check, engine.task, fit_rows, eval_rows, plan["model_id"], history=history)
        key = f"{batch.batch_id}:inspect:adaptive:{identity}"
        previous = next((c for c in existing if c["check_key"] == key), None)
        event_key = f"{context_key}:inspection:{i}"
        saved_event = _event(engine, event_key)
        from harness.methods.diagnosis.investigation import check_catalog
        result = engine._operation(batch, f"inspect:adaptive:{identity}", check_catalog()[check.tool]["cost_demo_units"], 3,
            lambda: engine._execute_check(batch, check, fit_rows, eval_rows, observed))
        if result is None:
            continue
        result = {**deepcopy(result), "check_key": key, "question": check.question,
                  "scope":scope,"alternatives": {"true": check.if_true.to_record(), "false": check.if_false.to_record()}}
        result["recommended_action"]=(check.if_true.to_record() if result["signal"] else check.if_false.to_record()) if type(result.get("signal")) is bool else None
        result["followup_id"]=digest([key,scope,result["alternatives"]])
        if previous is None:
            existing.append(result)
        else:
            # 新问题或后续改法使用同一计算时更新说明，不重复计费。
            existing[existing.index(previous)] = result
        reused = saved_event["reused"] if saved_event else previous is not None
        new_work += int(not reused)
        engine._event(event_key, "inspection_result", result=result, reused=reused)
        if router:
            engine._event(event_key+":problem", "problem_check", batch_id=batch.batch_id,
                           problem_ids=scope["problem_ids"], tool=check.tool, check_key=key, reused=reused,
                          model_id=plan["model_id"])
    return new_work


def run_decision_loop(engine, batch, recent, history, explore, baseline, current, scores, trials):
    """维护循环有明确进入条件、轮次和退出记录，不接触当前采用答案。

    以已有总体/分组事实进入，按允许动作执行；新工作、重复工作及失败
    都进入历史。候选、检查和格式修正共享当前预算与调用上限。
    """
    tried, manifests, rounds = set(), set(), []
    idle = 0
    exit_reason = "plan_call_limit"
    if baseline is None:
        entry = engine.state.get("maintenance_entries",{}).get(batch.batch_id,{"enter":True})
        exit_reason = "current_evaluation_unavailable" if entry["enter"] else entry["reason"]
    else:
        for index in range(engine.config.max_plan_calls):
            round_key = f"{batch.batch_id}:work_round:{index}"
            saved_context = _event(engine, f"{batch.batch_id}:plan_context:{index}")
            if saved_context:
                context = deepcopy(saved_context["context"])
            else:
                if len(tried) >= engine.config.max_trials:
                    exit_reason = "trial_limit"
                    break
                if engine.config.demo_budget_per_batch - engine.state["ledger"][batch.batch_id]["demo_units"] <= 3:
                    exit_reason = "budget_reserved_for_output"
                    break
                context = engine._context(batch, recent, history, baseline, trials, scores)
                context["maintenance"].update(stage="decision", decision={
                    "round": index, "remaining_trials": engine.config.max_trials - len(tried),
                    "history": deepcopy(rounds[-8:]), "repair": None,
                    "inspectable_models": [mid for mid in scores if engine.state["operations"].get(
                        f"{batch.batch_id}:explore:{mid}", {}).get("status") == "completed"]})
            selected, error = _request(engine, batch, context, index)
            if selected is None:
                exit_reason = error
                break
            plan, context, context_key = selected
            kind = plan["operation"]
            if kind == "finish":
                exit_reason = "planner_finished"
                engine._event(f"{round_key}:finish", "planner_finished", reason=plan["reason"], evidence_refs=plan["evidence_refs"])
                break
            if kind == "inspect":
                progress = _run_checks(engine, batch, recent, explore, plan, context_key)
            else:
                actions = (recheck_actions(plan, context) if kind == "recheck" else
                           [ActionPlan.parse(a) for a in plan["actions"]])
                if kind == "recheck":
                    # 复核要求对照与方法均能安排；已有完成对照可直接复用。
                    needed = [a for a in actions if a.key not in tried]
                    incomplete = any(t["action"] in {a.key for a in actions} and t["status"] != "completed" for t in trials)
                    free = engine.config.demo_budget_per_batch - engine.state["ledger"][batch.batch_id]["demo_units"]
                    authorized = _event(engine, f"{round_key}:recheck")
                    # 已登记复核按原身份恢复；已消耗额度不能再次作为整对任务的准入费用。
                    if authorized is None and (incomplete or len(tried) + len(needed) > engine.config.max_trials or (needed and free < 4 * len(needed) + 3)):
                        exit_reason = "recheck_pair_unavailable"
                        engine._event(f"{round_key}:unavailable", "recheck_unavailable", method_id=plan["method_id"])
                        break
                    engine._event(f"{round_key}:recheck", "method_recheck_requested", method_id=plan["method_id"],
                                  action_keys=[a.key for a in actions], context_key=context_key)
                engine._event(f"{batch.batch_id}:plan_accepted:{index}", "plan_accepted", batch_id=batch.batch_id,
                              context_key=context_key, actions=[a.to_record() for a in actions])
                if kind == "experiment" and engine.config.resource_strategy != "equal":
                    from harness.methods.search.resource_allocation import run_halving
                    progress = run_halving(engine,batch,current,recent,history,explore,actions,trials,scores,tried,manifests,context_key)
                else:
                    progress = _run_actions(engine, batch, current, recent, history, explore, actions,
                                            trials, scores, tried, manifests, context_key, plan.get("method_id"))
            prior = _event(engine, round_key)
            # 重放时沿用当时的新工作数量，不能把已完成任务误计为本轮重复空转。
            outcome = prior["outcome"] if prior else {"round": index, "operation": kind,
                "reason": plan["reason"], "new_work": progress, "tried_actions": sorted(tried),
                "best_score": engine.task.utility(max(scores.values()))}
            engine._event(round_key, "work_round_completed", outcome=outcome)
            rounds.append(deepcopy(outcome))
            idle = 0 if outcome["new_work"] else idle + 1
            if idle >= engine.config.max_idle_rounds:
                exit_reason = "no_new_work"
                break
    exit_key = f"{batch.batch_id}:maintenance_finished"
    old_exit = _event(engine, exit_key)
    exit_value = old_exit["result"] if old_exit else {"reason": exit_reason, "rounds": len(rounds),
        "tried_actions": sorted(tried), "best_exploration_score": engine.task.utility(max(scores.values())) if scores else None}
    engine.state.setdefault("maintenance_exits", {})[batch.batch_id] = deepcopy(exit_value)
    engine._event(exit_key, "maintenance_finished", result=exit_value)
    return trials
