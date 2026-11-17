"""有界互补起点与机会偏置；选择分数和正式模型指标分别保存。

起点覆盖吸收GEPA，机会偏置吸收Loss-Free Balancing。分组由输入事前声明，
只使用本批完整可比记录；过去已确认的构建机会才改变下一次选择。
"""
from math import isfinite, log, sqrt
from harness.core.contracts import digest

DEFAULT_TARGETS = {"history": 2., "processor": 2., "predictor": 2., "correction": 2.,
                   "weight": 1., "loss": 1., "method": 1., "retrain": .5}


def candidate_bases(context):
    """返回总体有效最好及有足够支持的组内领先起点；无覆盖证据时退回单一起点。"""
    m = context["maintenance"]
    facts = {i["kind"]: i["value"] for i in context["facts"]["evidence"]}
    trials = facts["attempts"]["current"]["attempts"]
    valid = []
    for t in trials:
        a = t.get("assessment") or {}
        action = a.get("effective_action") or {}
        gain = a.get("utility_delta")
        if (t.get("status") == "completed" and t.get("final_resource", True)
                and t.get("resource_fraction", 1.) == 1. and action.get("pipeline") and not action.get("correction")
                and a.get("comparison", {}).get("status") == "matched"
                and type(gain) in (int, float) and isfinite(gain)):
            valid.append(t)
    valid.sort(key=lambda t: (-t["assessment"]["utility_delta"], t["model_id"]))
    pool = [valid[0]["model_id"]] if valid and valid[0]["assessment"]["utility_delta"] > 0 else []
    support = m.get("candidate_min_support", 4)
    winners = {}
    for t in valid:
        for g in t["assessment"]["comparison"].get("groups", []):
            gain = g.get("utility_delta")
            # 支持数是描述性筛选条件；独立单位已声明时同时检查单位覆盖。
            if (g["count"] < support or type(gain) not in (int, float) or not isfinite(gain) or gain <= 0
                    or g.get("rows_without_unit", 0) == 0 and g.get("known_unit_count", 0) < 2):
                continue
            candidate = (gain, t["model_id"])
            prior = winners.get(g["group"])
            if prior is None or gain > prior[0] or gain == prior[0] and candidate[1] < prior[1]:
                winners[g["group"]] = candidate
    cap = m.get("candidate_pool_cap", 3)
    # 总体最好优先；之后按新增领先组覆盖选择，数量受限，不新增评价调用。
    while len(pool) < cap:
        remaining = [t for t in valid if t["model_id"] not in pool]
        if not remaining:
            break
        best = min(remaining, key=lambda t: (-sum(v[1] == t["model_id"] for v in winners.values()),
                                            -t["assessment"]["utility_delta"], t["model_id"]))
        if not any(v[1] == best["model_id"] for v in winners.values()):
            break
        pool.append(best["model_id"])
    return pool, {g: {"model_id": v[1], "gain": v[0]} for g, v in sorted(winners.items())}


def apply_routing(options, context):
    """偏置只调整目录顺序；旧机会按稳定构建身份去重，失败尝试也占用机会。"""
    m = context.get("maintenance", {})
    mode = m.get("action_routing", "priority")
    rows = options.get("candidates", [])
    for row in rows:
        row["fit_condition"] = m.get("action_conditions", {}).get(row["key"])
        row["eligible"] = row["fit_condition"] is None or row["fit_condition"]["ready"]
    families = sorted({r["kind"] for r in rows if not r["attempted"] and r["eligible"]})
    targets = {**DEFAULT_TARGETS, **(m.get("routing_targets") or {})}
    counts, bias, seen = {}, {}, set()
    step, limit = m.get("routing_step", .25), m.get("routing_bias_limit", 2.)
    for event in m.get("routing_history", []):
        if event["id"] in seen:
            continue
        seen.add(event["id"])
        chosen = event["family"]
        counts[chosen] = counts.get(chosen, 0) + 1
        eligible = sorted(set(event.get("eligible", [chosen])))
        total = sum(counts.get(f, 0) for f in eligible)
        denominator = sum(targets.get(f, 1.) for f in eligible)
        for f in eligible:
            delta = total * targets.get(f, 1.) / denominator - counts.get(f, 0)
            bias[f] = max(-limit, min(limit, bias.get(f, 0.) + step * ((delta > 1e-12) - (delta < -1e-12))))
    positions = {k: i for i, k in enumerate(options.get("ranked_keys", []))}
    for row in rows:
        row["selection_score"] = -row["priority"] + (bias.get(row["kind"], 0.) if mode == "balanced_opportunity" else 0.)
    if mode == "balanced_opportunity":
        options["ranked_keys"] = [r["key"] for r in sorted(rows, key=lambda r: (-r["selection_score"], positions[r["key"]]))]
    elif mode == "epsilon":
        number = int(digest([context.get("task_id"), context.get("batch_id"), len(context.get("trials", []))])[:12], 16) / 16**12
        if number < m.get("routing_epsilon", .1):
            options["ranked_keys"] = list(options["rotation_keys"])
    rewards={}
    if mode=="reward_ucb":
        options["basis"]="近期同资料配对改善的UCB参照；原指标不变，费用独立记录"
        # AOS的收益归因+UCB选择作为成熟参照；近期窗口应对变化，不声称继承平稳理论界。
        history=[e for e in m.get("routing_history",[]) if e.get("credit_status")=="matched"][-m.get("routing_reward_window",20):]
        scale=1. if context["objective"]["task_type"]=="classification" else m.get("routing_loss_scale")
        if scale is None or not isfinite(scale) or scale<=0:
            raise ValueError("回归收益选择需要事前声明loss_scale，不使用未知跨任务分数尺度")
        for e in history:
            rewards.setdefault(e["family"],[]).append((max(-1.,min(1.,e["utility_gain"]/scale))+1.)/2)
        total=sum(len(values) for values in rewards.values())
        for row in rows:
            values=rewards.get(row["kind"],[])
            row["credit_count"]=len(values)
            row["selection_score"]=(sum(values)/len(values)+m.get("routing_ucb_exploration",1.)*sqrt(2*log(max(2,total))/len(values))) if values else None
        options["ranked_keys"]=[r["key"] for r in sorted(rows,key=lambda r:(r["selection_score"] is not None,
            -(r["selection_score"] or 0.),positions[r["key"]]))]
    options["routing"] = {"mode": mode, "bias": bias, "opportunities": counts, "eligible_families": families,
                          "targets": {f: targets.get(f, 1.) for f in families}, "score_role": "proposal_selection_only",
                          "reward_counts":{family:len(values) for family,values in rewards.items()}}
    return options


def routing_history(state, batch_id, trials):
    """每个动作一次机会，收益只取同资料完整档与来源配方的成对比较，费用包含所有档。"""
    from harness.core.contracts import TaskSpec
    from harness.evaluation.metrics import compare_evaluations
    from harness.evaluation.costs import operation_costs
    task=TaskSpec.parse(state["task"])
    history, seen = [], set()
    batches = [(bid, run["trials"]) for bid, run in state.get("completed", {}).items()] + [(batch_id, trials)]
    for bid, records in batches:
        for t in records:
            key = f"{bid}:{t['action']}"
            op = t.get("build_operation_key") or f"{bid}:build:{t['action']}"
            if key in seen or op not in state["operations"] or not t.get("action_family"):
                continue
            seen.add(key)
            siblings=[s for s in records if s["action"]==t["action"]]
            final=next((s for s in reversed(siblings) if s.get("final_resource",True) and s.get("resource_fraction",1.)==1.),None)
            entry={"id":key,"family":t["action_family"],"eligible":t.get("routing_eligible_families",[]) or [t["action_family"]],
                   "credit_status":"unavailable","utility_gain":None}
            keys=set()
            for attempt in siblings:
                keys.add(attempt.get("build_operation_key") or f"{bid}:build:{attempt['action']}")
                if attempt.get("model_id"): keys.add(f"{bid}:explore:{attempt['model_id']}")
            entry["cost"]=operation_costs([state["operations"][k] for k in keys if k in state["operations"]])
            entry["operation_keys"]=sorted(keys & state["operations"].keys())
            if final and final.get("status")=="completed":
                parent=(final.get("proposal_origin") or {}).get("model_id") or final.get("parent_id")
                before=state["operations"].get(f"{bid}:explore:{parent}",{}).get("value")
                after=state["operations"].get(f"{bid}:explore:{final['model_id']}",{}).get("value")
                if before and after and compare_evaluations(task,before,after)["status"]=="matched":
                    gain=task.utility(after["score"])-task.utility(before["score"])
                    if isfinite(gain): entry.update(credit_status="matched",utility_gain=gain,parent_model=parent,
                        candidate_model=final["model_id"],evaluation_digest=digest([after["sample_versions"],after.get("score_definition")]))
            history.append(entry)
    return history


def trial_routing(context, action):
    """将计划快照中实际匹配的动作族登记到尝试；目录外动作保持未分类。"""
    options = context.get("maintenance", {}).get("action_options")
    if options is None:
        from harness.research.methods.search.action_selection import build_action_options
        options = build_action_options(context)
    row = next((r for r in options.get("candidates", []) if r["key"] == action.key), None)
    return {"action_family": row["kind"] if row else None,
            "routing_eligible_families": options.get("routing", {}).get("eligible_families", []),
            "target_issue_ids": context.get("maintenance", {}).get("problem_router", {}).get("selected_ids", [])}
