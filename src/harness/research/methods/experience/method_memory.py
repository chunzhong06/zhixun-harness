"""保存可执行改法及其对照依据；复用提供候选，最终模型仍由共同评价选择。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

from harness.core.contracts import ActionPlan, digest
from harness.core.recipe import CATALOG_VERSION, CATBOOST_DEFAULT_DEPTH, PipelineRecipe


METHOD_MEMORY_VERSION = 2


def effective_recipe(raw, task_type):
    """按SklearnBackend实际构造参数归一化，供对照与基线适用性共同使用。

    CatBoost的regularization映射l2_leaf_reg，空深度与显式默认深度等价；
    线性回归使用Ridge默认求解器，当前后端没有传入max_iter。
    """
    if raw is None:
        return None
    recipe = asdict(PipelineRecipe.parse(raw))
    if recipe["predictor"] == "catboost" and recipe["max_depth"] is None:
        recipe["max_depth"] = CATBOOST_DEFAULT_DEPTH
    fields = ["processor", "features", "predictor"]
    fields += {"linear": ["regularization"] + (["max_iter"] if task_type == "classification" else []),
               "forest": ["trees", "max_depth"],
               "catboost": ["trees", "max_depth", "learning_rate", "regularization"]}[recipe["predictor"]]
    fields += ["class_weight"] if task_type == "classification" else (
        ["regression_loss"] if recipe["predictor"] == "catboost" else [])
    return {key: recipe[key] for key in fields}


def _trial_cost(state, batch_id, trial):
    """引用原操作的投入，未知量保留为空；比较记录不再次累计账本费用。"""
    build = trial.get("build_operation_key", f"{batch_id}:build:{trial['action']}")
    score = f"{batch_id}:explore:{trial['model_id']}"
    return [{"operation_key": key, "demo_units": op.get("demo_units"),
             "elapsed_seconds": op.get("elapsed_seconds"),
             "execution": deepcopy(op.get("execution"))}
            for key in (build, score) if (op := state.get("operations", {}).get(key)) is not None]


def _comparisons(state, task, batch_id, trial, trials):
    """从已经完成的同资料、同父版本、同资源档结果形成有方向的比较。

    显式组件与继承组件都能作为参照。多个字段一起变化记录组合收益，
    不将两个方案的分差解释成各字段的独立作用或因子交互效应。
    """
    if trial["status"] != "completed":
        return []
    candidate = state.get("models", {}).get(trial["model_id"], {})
    if candidate.get("correction"):
        return []
    after = effective_recipe(candidate.get("pipeline"), task.task_type)
    pairs = []
    for control in trials:
        if state.get("models", {}).get(control.get("model_id"), {}).get("correction"):
            continue
        if (control is trial or control.get("status") != "completed" or
                control.get("model_id") == trial["model_id"] or control.get("final_resource") is False):
            continue
        if not trial.get("parent_id") or control.get("parent_id") != trial["parent_id"]:
            continue
        if any(not trial.get(key) or sorted(control.get(key, [])) != sorted(trial[key])
               for key in ("training_versions", "evaluation_versions")):
            continue
        if control.get("resource_fraction", 1.) != trial.get("resource_fraction", 1.):
            continue
        before = effective_recipe(state.get("models", {}).get(control["model_id"], {}).get("pipeline"), task.task_type)
        if before == after:
            continue
        changes = ({key: {"before": before.get(key), "after": after.get(key)}
                    for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}
                   if before is not None and after is not None else None)
        inherited = control.get("action_plan", {}).get("pipeline") is None
        gain = task.utility(trial["exploration_score"]) - task.utility(control["exploration_score"])
        pairs.append({"id": "comparison-" + digest([batch_id, trial["model_id"], control["model_id"]])[:24],
            "recipe_semantics_version": METHOD_MEMORY_VERSION,
            "comparison": "matched_retraining" if inherited else "matched_components",
            "candidate_model_id": trial["model_id"], "control_model_id": control["model_id"],
            "parent_id": trial["parent_id"], "candidate_recipe": after, "control_recipe": before,
            "changed_fields": changes, "effect": "backend_to_recipe" if changes is None else
                "single_component_change" if len(changes) == 1 else "joint_recipe_change",
            "utility_gain": gain if math.isfinite(gain) else None,
            "training_versions": sorted(trial["training_versions"]),
            "evaluation_versions": sorted(trial["evaluation_versions"]),
            "resource_fraction": trial.get("resource_fraction", 1.),
            "costs": {"control": _trial_cost(state, batch_id, control),
                      "candidate": _trial_cost(state, batch_id, trial)}})
    return sorted(pairs, key=lambda p: (p["comparison"] != "matched_retraining", p["id"]))


def scope_for(task, recent, checks):
    """适用范围只描述已观察条件，不从组名推断设备或故障含义。"""
    return {"task_signature": task.signature, "groups": sorted({s.group for s in recent}),
            "catalog_version": CATALOG_VERSION,
            # 错误分布受当前模型影响，不能把修复后的误差变化当成工况变化。
            "signals": {c["tool"]: {"signal": c["signal"], "threshold": c["threshold"]}
                        for c in checks if c["tool"] == "feature_scale"}}


def describe_inputs(rows):
    """仅从合法拟合输入生成位置/尺度摘要；设备名称不参与距离计算。"""
    if not rows:
        return None
    features = []
    for column in zip(*(s.values for s in rows)):
        scale = max(map(abs,column)) or 1.
        normalized = [value/scale for value in column]
        mean = math.fsum(normalized)/len(column)
        spread = math.sqrt(math.fsum((value-mean)**2 for value in normalized)/len(column))
        features.append({"mean":mean*scale,"std":min(spread,1.)*scale})
    return {"count":len(rows),"features":features}


def input_distance(left, right):
    if not left or not right or len(left["features"]) != len(right["features"]):
        return None
    differences = []
    for a,b in zip(left["features"],right["features"]):
        denominator = max(a["std"],b["std"],abs(a["mean"])*1e-6,abs(b["mean"])*1e-6,1e-12)
        # 先缩放再求差，避免有限大数相减溢出。
        differences.append(abs(a["mean"]/denominator-b["mean"]/denominator)+abs(a["std"]/denominator-b["std"]/denominator))
    return math.fsum(differences)/len(differences) if differences else None


def retrieve_methods(methods, scope, limit=8, *, recent=(), mode="exact", distance_limit=1.,
                     current_model=None, task_type="classification"):
    """返回同任务、同已观察范围的经验；相反对照使该范围暂停推荐。

    单次服务反馈只说明该版本的表现，不改变方法的对照证据。范围不符
    的经验不会自动扩展到新情境，仍保存在完整方法库供后续研究。
    """
    selected = []
    for mid, record in methods.items():
        if record["task_signature"] != scope["task_signature"]:
            continue
        observations = [o for o in record["observations"] if o["scope"] == scope]
        if not observations:
            if mode != "similar_recheck" or record.get("catalog_version") != scope["catalog_version"]:
                continue
            similar = []
            latest_scope = {}
            for observation in record["observations"]:
                latest_scope[digest(observation["scope"])] = observation
            for observation in latest_scope.values():
                if observation["comparison"] not in {"matched_retraining", "matched_components"} or (observation.get("utility_gain") or 0) <= 0:
                    continue
                distance = input_distance(describe_inputs(recent),observation.get("input_summary"))
                if distance is not None and distance <= distance_limit:
                    similar.append((distance,observation))
            if similar:
                distance, origin = min(similar,key=lambda pair:pair[0])
                selected.append({"id":mid,"action":deepcopy(record["action"]),"scope":deepcopy(scope),
                    "status":"similar_scope_needs_recheck","eligible":False,"observations":0,
                    "last_comparison":None,"distance":distance,"source_observation":origin["id"],
                    "source_scope":deepcopy(origin["scope"])})
            continue
        recipe = effective_recipe(current_model.get("pipeline"), task_type) if current_model else None
        paired = [o for o in observations if o["comparison"] in {"matched_retraining", "matched_components"}]
        last = paired[-1] if paired else None
        status = "needs_comparison"
        if last:
            status = "needs_recheck"
            if current_model is None:
                status = "baseline_unknown"
            else:
                matches = [(o, p) for o in paired for p in o.get("comparisons", [])
                           if p.get("recipe_semantics_version") == METHOD_MEMORY_VERSION
                           and p["control_recipe"] == recipe and (recipe is not None or
                               p["parent_id"] == current_model["model_id"])]
                if matches:
                    observation, pair = matches[-1]
                    last = {**deepcopy(observation), "applicable_comparison": deepcopy(pair)}
                    status = "supported_in_observed_scope" if (pair["utility_gain"] or 0) > 0 else "needs_recheck"
                else:
                    status = "baseline_changed_needs_recheck"
                if recipe is not None and recipe == effective_recipe(record["action"].get("pipeline"), task_type):
                    status = "already_current_recipe"
        if observations[-1]["status"] != "completed":
            status = "needs_recheck"
        if current_model and current_model.get("correction"):
            status = "composite_baseline_needs_recheck"
        selected.append({"id": mid, "action": deepcopy(record["action"]), "scope": deepcopy(scope),
                         "status": status, "observations": len(observations),
                         "last_comparison": deepcopy(last),
                         "eligible": status == "supported_in_observed_scope"})
    selected.sort(key=lambda r: (not r["eligible"], r.get("distance",0), r["id"]))
    return selected[:limit]


def record_methods(state, task, batch_id, recent, checks, trials, decision):
    """保存实际完成的比较与适用条件；恢复时按批次/动作身份补齐原事件。"""
    scope = scope_for(task, recent, checks)
    changed = []
    for trial in trials:
        # 复用相同计算或没有入选资料时没有新方法结果，不改变已有支持状态。
        if trial.get("status") == "skipped_duplicate_or_empty":
            continue
        if trial.get("final_resource") is False:
            continue
        raw = trial.get("action_plan")
        if raw is None or raw.get("pipeline") is None:
            continue
        action = ActionPlan.parse(raw)
        # 修正绑定具体基础产物；完整尝试保存在历史，不能当成任意父版本可用的重训配方。
        if action.correction:
            continue
        reusable = action.to_record()
        reusable.update(reason="复用已登记的组件配方；依据本次事实重新评价", evidence_refs=[], method_id=None)
        mid = "method-" + digest([task.signature, CATALOG_VERSION, action.key])[:24]
        record = state["methods"].setdefault(mid, {"id": mid, "task_signature": task.signature,
            "action": reusable, "observations": [], "catalog_version": CATALOG_VERSION})
        observation_id = f"{batch_id}:{action.key}"
        existing = next((o for o in record["observations"] if o["id"] == observation_id), None)
        if existing:
            # 状态可能已保存而逐项事件尚未齐全；让控制器按稳定事件ID补齐。
            changed.append({"method_id": mid, "observation": deepcopy(existing)})
            continue
        pairs = _comparisons(state, task, batch_id, trial, trials)
        pair = pairs[0] if pairs else None
        base = state.get("operations", {}).get(f"{batch_id}:explore:{trial.get('parent_id')}", {}).get("value", {})
        total_gain = (task.utility(trial["exploration_score"]) - task.utility(base["score"])) if (
            trial["status"] == "completed" and base.get("sample_versions") and
            sorted(base["sample_versions"]) == sorted(trial.get("evaluation_versions", []))) else None
        observation = {"id": observation_id, "batch_id": batch_id, "scope": scope,
            "input_summary": describe_inputs(recent),
            "status": trial["status"], "model_id": trial.get("model_id"),
            "parent_id": trial.get("parent_id"), "source_method_id": trial.get("recheck_method_id") or action.method_id,
            "comparison": pair["comparison"] if pair else "candidate_observation",
            "comparisons": pairs, "total_gain_from_parent": total_gain if total_gain is None or math.isfinite(total_gain) else None,
            "control_model_id": pair["control_model_id"] if pair else None,
            "utility_gain": pair["utility_gain"] if pair else None,
            "candidate_pipeline": deepcopy(state.get("models", {}).get(trial.get("model_id"), {}).get("pipeline")),
            "control_pipeline": deepcopy(state.get("models", {}).get(pair["control_model_id"], {}).get("pipeline")) if pair else None,
            "training_versions": trial.get("training_versions", []),
            "evaluation_versions": trial.get("evaluation_versions", []),
            "adopted": bool(trial.get("model_id")) and decision["selected_model"] == trial["model_id"],
            "decision_key": f"{batch_id}:decision", "evidence_kind": state["evidence_kind"]}
        record["observations"].append(observation)
        changed.append({"method_id": mid, "observation": observation})
    return changed
