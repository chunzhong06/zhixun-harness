"""同资料分档筛选、边界保留与完整档抽查；所有实际构建费用按原操作去重。"""
import math

from harness.core.contracts import digest
from harness.adapters.datasets.data_conditions import independent_units
from harness.research.methods.search.action_selection import proposal_origin
from harness.research.methods.search.search_policy import trial_routing
from harness.evaluation.costs import operation_costs


def resource_rows(rows, fraction, seed, task=None):
    """按独立单位取嵌套子集；分类扩大到覆盖完整拟合池的已观察类别。"""
    groups = sorted(independent_units(rows), key=lambda g: digest([seed, g]))
    selected = set(groups[:max(1, math.ceil(len(groups) * fraction))])
    required = {s.label for s in rows} if task and task.task_type == "classification" else set()
    for group in groups[len(selected):]:
        if required <= {s.label for s in rows if s.unit_id in selected}:
            break
        selected.add(group)
    return tuple(s for s in rows if s.unit_id in selected)


def _rank_reversals(low, full, tolerance):
    """低/完整档动作、父模型和评价修订相同，低档拟合清单被完整档包含。"""
    shared = [k for k in low if k in full and low[k]["parent_id"] == full[k]["parent_id"]
              and low[k]["evaluation_versions"] == full[k]["evaluation_versions"]
              and set(low[k]["training_versions"]) <= set(full[k]["training_versions"])]
    return [(a,b) for i,a in enumerate(shared) for b in shared[i+1:]
            if (low[a]["utility"]-low[b]["utility"])*(full[a]["utility"]-full[b]["utility"]) < 0
            and abs(low[a]["utility"]-low[b]["utility"]) > tolerance
            and abs(full[a]["utility"]-full[b]["utility"]) > tolerance]


def run_halving(engine, batch, current, recent, history, explore, actions, trials, scores, tried, manifests, context_key):
    """固定分档作参照；reliable_halving保留边界、抽查落选项，反转时扩大复查。

    每档完整重建，少量机会不拟合Hyper-Tune代理。抽查也消耗原预算，完整档
    未完成时只保存观察，不进入最终提名；全过程没有额外采用集读取。
    """
    reliable = engine.config.resource_strategy == "reliable_halving"
    context = next(e["context"] for e in engine.state["events"] if e["key"] == context_key)
    candidates = [a for a in actions if a.key not in tried][:max(0, engine.config.max_trials-len(tried))]
    selected = {a.key: engine._select_rows(batch, a, recent, history) for a in candidates}
    distinct = []
    for action in candidates:
        tried.add(action.key)
        manifest = engine._training_manifest(action, selected[action.key], current.model_id)
        if not selected[action.key] or manifest in manifests:
            trials.append({"action": action.key, "status": "skipped_duplicate_or_empty", "model_id": None,
                           "training_ids": [s.sample_id for s in selected[action.key]]})
            continue
        manifests.add(manifest)
        distinct.append(action)
    candidates = distinct
    levels = tuple(engine.config.resource_fractions)
    keep=max(1,math.ceil(len(candidates)/engine.config.resource_reduction))
    audit=min(engine.config.resource_audit_count,max(0,len(candidates)-keep)) if reliable else 0
    possible_eliminations=max(0,len(candidates)-keep-audit)
    planned=len(candidates)
    stage_size=len(candidates)
    for _ in levels[1:]:
        stage_size=min(stage_size,math.ceil(stage_size/engine.config.resource_reduction)+audit)
        planned+=stage_size
    free=engine.config.demo_budget_per_batch-engine.state["ledger"][batch.batch_id]["demo_units"]
    fallback=(reliable and (len(candidates)<3 or not possible_eliminations or planned*4+3>free))
    engine._event(f"{context_key}:allocation_admission","allocation_admission",candidates=len(candidates),
                  ranking_keep=keep,audit_return=audit,possible_eliminations=possible_eliminations,
                  planned_builds=planned,estimated_demo_units=planned*4,remaining_demo_units=free)
    if (fallback or any(any(s.unit_id is None for s in selected[a.key])
            or len({s.unit_id for s in selected[a.key]}) < 2 for a in candidates)):
        levels = (1.,)
        engine._event(f"{context_key}:resource_fallback", "resource_fallback",
                      reason="small_or_unscreenable_cohort_budget_or_independent_units", levels=list(levels))
    operations, full, low_stages, removed, audits = set(), {}, [], {}, set()
    new_work = 0

    def evaluate(action, stage, fraction, final, reason):
        nonlocal new_work
        pool = selected[action.key]
        rows = resource_rows(pool, fraction, engine.config.seed, engine.task) if fraction < 1 else pool
        tag = digest([batch.batch_id, [(s.sample_id,s.revision) for s in rows]])[:16]
        build_key = f"{batch.batch_id}:build:{action.key}@resource-{tag}"
        model = engine._build(batch, action, rows, current.model_id, variant=f"resource-{tag}")
        result = engine._score(batch, model, explore, "explore", 3) if model else None
        trial = {"action": action.key, "action_plan": action.to_record(), "model_id": model.model_id if model else None,
            "proposal_origin": proposal_origin(context, action), **trial_routing(context, action),
            "status": "completed" if result else "unavailable", "exploration_score": result["score"] if result else None,
            "parent_id": current.model_id, "training_ids": [s.sample_id for s in rows],
            "training_versions": [f"{s.sample_id}@{s.revision}" for s in rows],
            "evaluation_versions": [f"{s.sample_id}@{s.revision}" for s in explore],
            "resource_fraction": fraction, "actual_row_fraction": len(rows)/len(pool),
            "allocation_stage": stage, "final_resource": final, "build_operation_key": build_key,
            "execution": "full_rebuild_each_stage", "allocation_reason": reason}
        trials.append(trial)
        engine._event(f"{context_key}:resource:{stage}:{action.key}", "resource_trial", trial=trial)
        for key in (build_key, f"{batch.batch_id}:explore:{model.model_id}" if model else None):
            if key in engine.state["operations"]:
                operations.add(key)
        new_work += int(model is not None)
        if result:
            utility = engine.task.utility(result["score"])
            if final:
                scores[model.model_id] = utility
                full[action.key] = {**trial, "utility": utility}
            return utility, action.key, action, {**trial, "utility": utility}
        return None

    tolerance = engine.config.resource_tie_tolerance
    for stage, fraction in enumerate(levels):
        final = stage == len(levels)-1
        results = [r for action in candidates if (r := evaluate(action,stage,fraction,final,
                   "audit" if action.key in audits else "ranking")) is not None]
        if not results:
            break
        results.sort(key=lambda r: (-r[0],r[1]))
        keep = len(results) if final else max(1,math.ceil(len(results)/engine.config.resource_reduction))
        survivors = list(results[:keep])
        if reliable and not final:
            cutoff = results[keep-1][0]
            survivors = [r for r in results if r[0] >= cutoff-tolerance or r[1] in audits]
            losers = [r for r in results if r not in survivors]
            # 稳定轮换抽查，不使用未来完整档分数挑选落选项。
            losers.sort(key=lambda r: digest([engine.config.seed,batch.batch_id,stage,r[1]]))
            for r in losers[:engine.config.resource_audit_count]:
                audits.add(r[1]); survivors.append(r)
        chosen = {r[1] for r in survivors}
        if not final:
            low_stages.append({r[1]: r[3] for r in results})
            removed.update({r[1]: r[2] for r in results if r[1] not in chosen})
        candidates = [r[2] for r in survivors]
        engine._event(f"{context_key}:allocation:{stage}", "resource_allocation",
            fraction=fraction, survivors=[a.key for a in candidates], audit_keys=sorted(audits), final_resource=final,
            evaluated=[{"action":r[1],"utility":r[0]} for r in results])
    reversals = [pair for low in low_stages for pair in _rank_reversals(low,full,tolerance)]
    if reliable and reversals:
        for key, action in sorted(removed.items()):
            if key not in full:
                evaluate(action, len(levels), 1., True, "ranking_reversal_recheck")
    engine._event(f"{context_key}:resource_summary", "resource_summary",
        strategy=engine.config.resource_strategy, rank_reversals=[list(p) for p in reversals],
        full_resource_actions=sorted(full), unconfirmed_actions=sorted(set(selected)-set(full)),
        operation_keys=sorted(operations), cost=operation_costs([engine.state["operations"][key] for key in sorted(operations)]))
    return new_work
