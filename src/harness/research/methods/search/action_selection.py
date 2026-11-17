"""用合法观察安排有限改法；排序是可比较的规则，不把优先级当作效果估计。

吸收MLE-STAR的当前方案/试改反馈、PipeLens的干预选择及类别覆盖研究。
具体优先规则与完整配方接续为本项目实现，来源和取舍见N087、N089。
"""
from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite

from harness.core.contracts import ActionPlan, digest
from harness.core.recipe import PipelineRecipe
from harness.core.actions import CorrectionSpec
from harness.research.methods.experience.method_memory import effective_recipe
from harness.research.methods.search.search_policy import apply_routing, candidate_bases


ACTION_SELECTION_VERSION = 5
SOURCES = ["N048", "N050", "N083", "N084", "N087", "N089", "N090"]


def _recipe_identity(recipe, ratio, effective, correction=None):
    # JSON的1与1.0写法不同，模型参数含义相同；去重依据实际含义。
    canonical = ({k: float(v) if type(v) in (int, float) else v for k, v in effective.items()}
                 if effective is not None else None)
    return digest([recipe, float(ratio), canonical] + ([correction] if correction else []))


def _build_action_options(context, forced_base=None):
    """生成共用候选、排序依据和未尝试项，不执行训练或读取采用资料。

    当前配置已知时只改指定字段；未知时显式候选使用目录默认配置并标注。
    best_trial沿用本批可比、完整尝试的有效配方，输出完整配置重新构建；
    配方来源与共同对照的在用模型分别记录，不要求后端支持接续训练。
    历史类别统计截断时，未列出的类保持未知。已试动作不重复安排；本批
    最佳完整尝试的错误摘要可更新优先级，跨批原始分数不用于直接排名。
    """
    facts = {i["kind"]: i for i in context["facts"].get("evidence", [])}
    maintenance = context.get("maintenance", {})
    if not maintenance.get("components") or not {"performance", "input_change", "attempts"} <= facts.keys():
        return {"version": ACTION_SELECTION_VERSION, "status": "unavailable", "candidates": []}
    task_type = context["objective"]["task_type"]
    performance = facts["performance"]["value"]
    raw_current = performance.get("current_pipeline")
    current = PipelineRecipe.parse(raw_current) if raw_current else PipelineRecipe()
    service_effective = effective_recipe(raw_current, task_type)
    base_recipe, base_ratio = "recent", 0.
    origin = {"basis": "incumbent", "model_id": performance.get("model_id"),
              "evidence_ref": facts["performance"]["id"]}
    pools = facts["input_change"]["value"]
    attempts = facts["attempts"]["value"]["current"]["attempts"]
    tried = {t["action"] for t in context["trials"]}
    profile = performance.get("error_profile") or {}
    observation_model = performance.get("model_id")
    observation_recipe = None
    profile_ref = facts["performance"]["id"]
    best_gain = 0.
    eligible = []
    for attempt in attempts:
        assessment = attempt.get("assessment") or {}
        gain = assessment.get("utility_delta")
        if (attempt.get("status") == "completed" and attempt.get("final_resource", True) is True
                and attempt.get("resource_fraction", 1.) == 1.
                and type(gain) in (int, float) and isfinite(gain)
                and assessment.get("comparison", {}).get("status") == "matched"):
            eligible.append(attempt)
    # 同分按稳定身份选择，恢复或资料记录顺序变化不会改换起点。
    eligible.sort(key=lambda a: (-a["assessment"]["utility_delta"], str(a.get("model_id", ""))))
    for attempt in eligible:
        assessment = attempt["assessment"]
        gain = assessment["utility_delta"]
        candidate_profile = (assessment.get("error_profiles") or {}).get("candidate")
        if candidate_profile and gain > best_gain:
            profile, best_gain = candidate_profile, gain
            observation_model, profile_ref = attempt.get("model_id"), facts["attempts"]["id"]
            observed_action = assessment.get("effective_action") or assessment.get("requested_action") or {}
            observation_recipe = [observed_action.get("recipe"), observed_action.get("ratio", 0.)]

    if maintenance.get("refinement_basis", "incumbent") in {"best_trial", "complementary"}:
        for attempt in eligible:
            if (forced_base and attempt["model_id"] != forced_base) or (not forced_base and attempt["assessment"]["utility_delta"] <= 0):
                continue
            effective_action = attempt["assessment"].get("effective_action") or {}
            if not effective_action.get("pipeline") or not attempt.get("model_id") or effective_action.get("correction"):
                continue
            action = ActionPlan.parse(effective_action)
            pipe = action.pipeline
            pipe.validate_task_type(task_type)
            if (action.ratio not in context["allowed_recipes"].get(action.recipe, [])
                    or pipe.predictor not in maintenance["components"]["predictor"]
                    or pipe.processor not in maintenance["components"]["processor"]):
                continue
            current = pipe
            base_recipe, base_ratio = action.recipe, action.ratio
            origin = {"basis": "best_trial", "model_id": attempt["model_id"],
                      "action_key": attempt["action"], "utility_delta": attempt["assessment"]["utility_delta"],
                      "evidence_ref": facts["attempts"]["id"]}
            if forced_base:
                profile = (attempt["assessment"].get("error_profiles") or {}).get("candidate") or profile
                observation_model, observation_recipe = attempt["model_id"], [action.recipe, action.ratio]
                profile_ref = facts["attempts"]["id"]
            break

    def class_counts(pool):
        target = pool.get("targets", {})
        return ({r["value"]: r["count"] for r in target.get("counts", [])},
                target.get("omitted_values", 0) == 0)

    recent_counts, recent_complete = class_counts(pools["recent"])
    history_counts, _ = class_counts(pools["history"])
    misses = [c["label"] for c in profile.get("classes", []) if c.get("missed", 0) > 0]
    missing = [c for c in misses if recent_complete and c not in recent_counts]
    recoverable = [c for c in missing if history_counts.get(c, 0) > 0]
    minority = [c for c in misses if recent_counts.get(c, 0) > 0 and recent_complete
                and recent_counts[c] * len(recent_counts) < sum(recent_counts.values())]
    investigations = facts.get("investigations", {}).get("value", {}).get("results", [])
    scale_issue = any(c.get("tool") in {"feature_scale", "tail_profile"} and c.get("signal") is True
                      for c in investigations)
    has_errors = bool(misses) if task_type == "classification" else (
        profile.get("overprediction_count", 0) + profile.get("underprediction_count", 0) > 0)
    evidence_refs = [facts[k]["id"] for k in ("performance", "input_change", "attempts")]
    candidates, identities = [], set()

    def add(kind, pipe=None, recipe=None, ratio=None, method_id=None, correction=None):
        recipe = base_recipe if recipe is None else recipe
        ratio = base_ratio if ratio is None else ratio
        # 方法证据绑定其原配方；普通动作才沿用本轮有效起点。
        if pipe is None and origin["basis"] == "best_trial" and not method_id:
            pipe = current
        if ratio not in context["allowed_recipes"].get(recipe, []):
            return
        if recipe != "recent" and not pools["history"]["count"]:
            return
        if pipe is not None:
            pipe.validate_task_type(task_type)
            if pipe.predictor not in maintenance["components"]["predictor"]:
                return
        effective = effective_recipe(asdict(pipe), task_type) if pipe else service_effective
        identity = _recipe_identity(recipe, ratio, effective, asdict(correction) if correction else None)
        if identity in identities:
            if method_id:
                prior = next(c for c in candidates if c["effective_identity"] == identity)
                prior.setdefault("supporting_method_ids", []).append(method_id)
                if prior["priority"] > 1:
                    prior.update(priority=1, reason="已有共同对照支持相同有效配方适用于当前范围与基线")
                    prior["action"]["reason"] = prior["reason"]
                    prior["action"]["evidence_refs"].append(facts["methods"]["id"])
            return
        identities.add(identity)
        refs = list(evidence_refs)
        if method_id:
            refs.append(facts["methods"]["id"])
        if scale_issue and kind == "processor" and "investigations" in facts:
            refs.append(facts["investigations"]["id"])
        priority, reason = 3, "保留同目录中的其他可执行改法作为探索"
        if recoverable and kind == "history":
            priority, reason = 0, f"漏判类别{recoverable}在近期池缺失、历史池已有；比较补入历史，实际覆盖由选择回执核对"
        elif kind == "weight" and pipe.class_weight == "balanced" and minority:
            priority, reason = 1, f"漏判类别{minority}已有近期资料但数量低于已观察类别均值；保持资料和其他组件比较训练权重"
            if observation_recipe == [recipe, ratio]:
                priority, reason = 0, reason + "；沿用当前最佳尝试的资料配方"
        elif kind == "processor" and scale_issue:
            priority, reason = 1, "已有尺度或尾部检查信号；保持资料及预测器比较处理方式"
        elif kind == "predictor" and has_errors:
            priority, reason = 2, "当前可比观察仍有错误；保持资料与处理方式比较预测器"
        elif method_id:
            priority, reason = 1, "已有共同对照支持该方法适用于当前范围与基线；保留再次评价"
        elif correction:
            priority, reason = (2 if has_errors else 3), "固定旧预测路径，用未参与其拟合的合法资料构造修正分支，之后共同评价"
        elif kind == "retrain":
            reason = "沿用当前配方重新训练，提供近期更新参照"
        plan = ActionPlan(recipe, ratio, reason, tuple(refs), pipe, method_id, correction)
        equivalent_tried = any(t.get("action_plan") and
            _recipe_identity(t["action_plan"]["recipe"], t["action_plan"].get("ratio", 0.),
                effective_recipe(t["action_plan"].get("pipeline"), task_type)
                if t["action_plan"].get("pipeline") else service_effective, t["action_plan"].get("correction")) == identity
            for t in context["trials"])
        candidates.append({"kind": kind, "action": plan.to_record(), "key": plan.key, "effective_identity": identity,
            "proposal_origin": ({"basis": "method", "method_id": method_id,
                                 "evidence_ref": facts["methods"]["id"]} if method_id else
                                {"basis": "frozen_model", "model_id": correction.base_model_id,
                                 "evidence_ref": facts["attempts"]["id"]} if correction else deepcopy(origin)),
            "priority": priority, "reason": reason, "attempted": plan.key in tried or equivalent_tried})

    # 三种顺序共用完整目录与配方，只改变提案先后；历史动作不额外增加容量。
    add("retrain")
    if base_recipe != "recent":
        add("history", recipe="recent", ratio=0.)
    add("processor", replace(current, processor="robust"))
    for predictor in ("forest", "catboost", "linear"):
        if predictor != current.predictor:
            pipe = replace(current, predictor=predictor, regression_loss="default", learning_rate=.05,
                           max_depth=min(current.max_depth, 10) if predictor == "catboost" and current.max_depth else current.max_depth)
            add("predictor", pipe)
    if task_type == "classification":
        add("weight", replace(current, class_weight="balanced" if current.class_weight == "none" else "none"))
    elif current.predictor == "catboost":
        add("loss", replace(current, regression_loss="mae" if current.regression_loss == "default" else "default"))
    for processor in ("standard", "none"):
        add("processor", replace(current, processor=processor))
    if context.get("task", {}).get("condition_features"):
        add("processor", replace(current, processor="condition_standard"))
    for recipe in ("coverage", "uniform"):
        for ratio in context["allowed_recipes"].get(recipe, []):
            add("history", recipe=recipe, ratio=ratio)
    if task_type == "classification" and current.class_weight == "none":
        # 固定、轮换和观察驱动均可尝试相同组合；反馈只影响次序，不扩充某一组的权限。
        for recipe in ("coverage", "uniform"):
            for ratio in context["allowed_recipes"].get(recipe, []):
                add("weight", replace(current, class_weight="balanced"), recipe, ratio)
    for method in maintenance.get("methods", []):
        if method.get("eligible"):
            action = ActionPlan.parse(method["action"])
            add("method", action.pipeline, action.recipe, action.ratio, method["id"])
    for base in maintenance.get("corrections", {}).get("bases", []):
        regions = ([None] if task_type == "regression" else []) + base["region_ids"][:2]
        for region_id in regions:
            correction = CorrectionSpec(base["model_id"], "residual" if task_type == "regression" else "region_replace", region_id)
            # 错误区域可能只有一个类别，森林分支仍可形成类别预测；数值任务拟合残差。
            branch = PipelineRecipe(predictor="forest", trees=32) if task_type == "classification" else PipelineRecipe()
            add("correction", branch, "recent", 0., correction=correction)
    width = max(1, maintenance.get("trial_limit", 3))
    offset = maintenance.get("batch_ordinal", 0) * width % len(candidates) if candidates else 0
    rotation = candidates[offset:] + candidates[:offset]
    positions = {c["key"]: i for i, c in enumerate(rotation)}
    ranked = sorted(candidates, key=lambda c: (c["priority"], positions[c["key"]]))
    return {"version": ACTION_SELECTION_VERSION, "status": "available", "sources": SOURCES,
        "basis": "条件优先级与轮换探索；未学习收益概率或使用跨批原始分数排名",
        "pipeline_basis": ("best_trial" if origin["basis"] == "best_trial" else
                           "current_model" if raw_current else "explicit_candidates_use_catalog_defaults"),
        "refinement_origin": origin,
        "observation_model": observation_model, "observation_recipe": observation_recipe,
        "observation_evidence_ref": profile_ref,
        "signals": {"missed_classes": misses, "missing_recent_classes": missing,
                    "recoverable_classes": recoverable, "underrepresented_missed_classes": minority,
                    "recent_classes_complete": recent_complete, "scale_issue": scale_issue},
        "candidates": candidates, "ranked_keys": [c["key"] for c in ranked],
        "rotation_keys": [c["key"] for c in rotation]}


def build_action_options(context):
    """共用动作生成器；可选互补起点和机会偏置均在共同采用之前生效。"""
    options = _build_action_options(context)
    if options["status"] != "available":
        return options
    if context.get("maintenance", {}).get("refinement_basis") == "complementary":
        pool, winners = candidate_bases(context)
        menus = [_build_action_options(context, mid) for mid in pool] or [options]
        rows, identities = [], {}
        for i in range(max(len(m["candidates"]) for m in menus)):
            for menu in menus:
                if i >= len(menu["candidates"]):
                    continue
                row = deepcopy(menu["candidates"][i])
                if row["effective_identity"] in identities:
                    identities[row["effective_identity"]].setdefault("alternative_origins", []).append(row["proposal_origin"])
                    continue
                identities[row["effective_identity"]] = row
                rows.append(row)
        options = deepcopy(menus[0])
        options.update(candidates=rows, rotation_keys=[r["key"] for r in rows],
                       ranked_keys=[r["key"] for r in sorted(rows, key=lambda r: r["priority"])],
                       candidate_pool={"model_ids": pool, "group_winners": winners})
    return apply_routing(options, context)


def proposal_origin(context, action):
    """从决策时快照定位目录配方来源；目录外动作保持未知，不推测模型意图。"""
    options = context.get("maintenance", {}).get("action_options")
    if options is None:
        options = build_action_options(context)
    row = next((c for c in options.get("candidates", []) if c["key"] == action.key), None)
    return deepcopy(row.get("proposal_origin")) if row else None


def validate_action_options(context):
    """接口接收的建议必须对应同一份合法事实与当前版本，不接受额外执行指令。"""
    supplied = context.get("maintenance", {}).get("action_options")
    if supplied is not None and supplied != build_action_options(context):
        raise ValueError("改法建议与当前观察或版本不一致")


class ActionOrderPlanner:
    """同目录的固定、轮换、观察驱动参照；逐次返回一个动作以读取新反馈。

    evidence方式在至少一次试改后，为最后一个名额保留轮换探索；无证据
    优先项时也按轮换选择。计划器不改变共同采用规则，不调用任何模型。
    """
    def __init__(self, order="evidence"):
        if order not in {"fixed", "rotating", "evidence"}:
            raise ValueError("未知改法顺序")
        self.order = order
        self.identity = f"action-order-v{ACTION_SELECTION_VERSION}:{order}"

    def propose(self, context):
        maintenance = context.get("maintenance", {})
        if maintenance.get("stage") == "inspection":
            return {"checks": [], "stop": True}
        options = build_action_options(context)
        followups=maintenance.get("inspection_followups",[])
        if self.order == "evidence" and not followups:
            from harness.research.methods.diagnosis.problem_tracking import propose_problem_check
            check = propose_problem_check(context, options)
            if check is not None:
                return check
        candidates = {c["key"]: c for c in options["candidates"] if not c["attempted"] and c.get("eligible",True)}
        refs = [i["id"] for i in context["facts"].get("evidence", []) if i["kind"] == "attempts"]
        attempted_count = len({t["action"] for t in context["trials"]
                               if t["action"].split("-")[0] in context["allowed_recipes"]})
        remaining = maintenance.get("decision", {}).get("remaining_trials", maintenance.get("trial_limit", 3) - attempted_count)
        exploration = (self.order == "evidence" and remaining == 1 and attempted_count > 0
                       and maintenance.get("action_routing", "priority") == "priority")
        keys = ([c["key"] for c in options["candidates"]] if self.order == "fixed" else
                options.get("rotation_keys", []) if self.order == "rotating" or exploration else options.get("ranked_keys", []))
        recommended=[ActionPlan.parse(f["action"]).key for f in followups if f["action"]]
        if self.order=="evidence" and recommended:
            keys=list(dict.fromkeys([*(k for k in recommended if k in candidates),*keys]))
        count = max(0, remaining) if context["policy"] == "plan_once" else min(1, max(0, remaining))
        if maintenance.get("resource_strategy") in {"successive_halving", "reliable_halving"}:
            # 反馈模式保留一个后续动作名额，用于吸收本组试验结果。
            count = min(3, max(0, remaining if context["policy"] == "plan_once" or remaining < 3 else remaining-1))
        chosen_rows = [candidates[key] for key in keys if key in candidates][:count]
        if context["policy"] == "plan_once" and self.order == "evidence" and len(chosen_rows) > 1:
            exploit = {c["key"] for c in chosen_rows[:-1]}
            exploratory = next((candidates[k] for k in options["rotation_keys"] if k in candidates and k not in exploit), None)
            if exploratory:
                chosen_rows[-1] = exploratory
        chosen = chosen_rows[0] if chosen_rows else None
        actions = [deepcopy(c["action"]) for c in chosen_rows]
        reason = (f"{self.order}顺序选择；" + ("最后名额保留轮换探索；" if exploration else "") + chosen["reason"]
                  if chosen else "本轮目录已尝试或名额已用完")
        if followups:
            reason=("检查支持不足，依据其他已知事实选择；" if not recommended else
                    "依据检查信号执行对应改法；" if chosen and chosen["key"] in recommended else
                    "检查原分支当前不可用或当前对照顺序选择其他动作；")+reason
            refs=list(dict.fromkeys([*refs,*(f["evidence_ref"] for f in followups)]))
        for i, action in enumerate(actions):
            action["reason"] = reason if i == 0 else f"{self.order}一次计划的后续候选；{chosen_rows[i]['reason']}"
            if followups: action["evidence_refs"]=list(dict.fromkeys([*action["evidence_refs"],*refs]))
        if maintenance.get("stage") == "decision":
            return {"operation": "experiment" if actions else "finish", "reason": reason,
                    "evidence_refs": actions[0]["evidence_refs"] if actions else refs,
                    **({"actions": actions} if actions else {})}
        return {"actions": actions, "stop": not actions}
