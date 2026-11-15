"""从当时可见的记录生成计划依据；计算摘要与行动引用分别保留。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from math import isfinite

from harness.evaluation.costs import operation_costs
from harness.contracts import ActionPlan, Sample, digest
from harness.evaluation.metrics import prediction_score, error_profile, scaled_mean
from harness.methods.planning.assessment import assess_attempt
from harness.models.corrections import selection_key

FACTS_VERSION = 9
HISTORY_LIMIT = 8
DETAIL_LIMIT = 16
KINDS = {"performance", "performance_change", "input_change", "attempts", "resources", "service_feedback"}
EXTRA_KINDS = {"investigations", "methods", "problems"}


def _difference(current, reference):
    if current is None or reference is None:
        return None
    value = current - reference
    return value if isfinite(value) else None


def _row_source(rows):
    return {"sample_versions": [f"{s.sample_id}@{s.revision}" for s in rows],
            "data_digest": digest([asdict(s) for s in rows])}


def _distribution(rows, field):
    counts = Counter(getattr(row, field) for row in rows)
    items = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return {"total_values": len(items), "omitted_values": max(0, len(items) - DETAIL_LIMIT),
            "counts": [{"value": key, "count": count} for key, count in items[:DETAIL_LIMIT]]}


def _pool(task, rows):
    return {"count": len(rows), "groups": _distribution(rows, "group"),
            "targets": (_distribution(rows, "label") if task.task_type == "classification"
                        else {"mean": scaled_mean([s.label for s in rows])})}


def build_facts(task, batch, recent, history, baseline, trials, state, config, *, signed_groups=False, check_devices=False):
    """返回计划摘要及本地依据登记，不执行模型或额外评价。

    当前表现取已完成的探索评价，输入变化只看合法fit池；当批采用资料
    不进入摘要。历史评价保留当时保存的分数，不用后来修订的答案重算。
    输出详情有固定上限并报告省略数；完整操作和样本修订来源留在本地。
    """
    items, evidence = [], {}

    def add(kind, value, sources):
        scope = {"task_signature": task.signature, "batch_id": batch.batch_id, "ordinal": batch.ordinal}
        key = "fact-" + digest([scope, kind, value, sources])
        item = {"id": key, "kind": kind, "value": value, "evidence_kind": state["evidence_kind"]}
        items.append(item)
        evidence[key] = {"item": item, "scope": scope, "sources": sources}

    explore = tuple(s for s in batch.samples if s.role == "explore" and s.available_at == batch.ordinal)
    model_id = baseline["model_id"] if baseline else state["current"]
    bundle = state.get("models", {}).get(model_id, {})
    base_key = f"{batch.batch_id}:explore:{model_id}"
    grouped = {}
    for row in explore:
        grouped.setdefault(row.group, []).append(row)
    group_scores = []
    if baseline:
        for group, rows in grouped.items():
            predictions = {s.sample_id: baseline["predictions"][s.sample_id] for s in rows}
            summary = {"group": group, "count": len(rows),
                       "score": prediction_score(task, tuple(rows), predictions)}
            if signed_groups and task.task_type == "regression":
                # 相同的 RMSE 既可能来自寿命高估，也可能来自低估。方向只取已揭晓的探索行。
                profile = error_profile(task, tuple(rows), predictions)
                summary.update({k: profile[k] for k in
                                ("mean_signed_error", "overprediction_count", "underprediction_count")})
                if (rows[0].unit_id is not None and all(s.unit_id == rows[0].unit_id and
                        s.time_start is not None and s.time_end is not None for s in rows)):
                    summary["observation_range"] = [min(s.time_start for s in rows), max(s.time_end for s in rows)]
            group_scores.append(summary)
        group_scores.sort(key=lambda row: (task.utility(row["score"]), row["group"]))
    device_info = {}
    if check_devices and baseline and task.task_type == "regression":
        from harness.methods.planning.rul_continuation import device_biases
        candidates = device_biases(explore, baseline["predictions"])
        issue = next((r for r in candidates if 4 * r["matching"] >= 3 * r["recent_count"]), None)
        device_info = {"device_bias_issue": issue, "eligible_devices": len(candidates)}
    add("performance", {**device_info, "status": "available" if baseline else "unavailable",
                        "model_id": model_id, "metric": task.metric, "target_unit": task.target_unit,
                        "current_pipeline": bundle.get("pipeline"),
                        "current_composition": bundle.get("correction"),
                        "pipeline_source": "model_bundle" if bundle.get("pipeline") else "backend_defined",
                        "retraining_rule": "普通重训沿用基础配方并重建整模型；修正分支配方记录在current_composition，需显式correction动作",
                        "score": baseline["score"] if baseline else None, "count": len(explore),
                        "error_profile": (baseline.get("observation_windows",{}).get("current",{}).get("error_profile")
                                          or baseline.get("error_profile") or error_profile(task,explore,baseline["predictions"])) if baseline else None,
                        "observation_windows": deepcopy(baseline.get("observation_windows",{})) if baseline else {},
                        "score_definition": deepcopy(baseline.get("score_definition")) if baseline else None,
                        "groups_worst_first": group_scores[:DETAIL_LIMIT],
                        "omitted_groups": max(0, len(group_scores) - DETAIL_LIMIT)},
        [{"operation_key": base_key, "model_id": model_id, "rows": _row_source(explore)}])

    # 比较同一个模型在两个批次上的观察；不同样本构成仍保留在依据中。
    previous = None
    for bid in reversed(list(state["completed"])):
        operation_key = f"{bid}:explore:{model_id}"
        op = state["operations"].get(operation_key, {})
        if op.get("status") == "completed":
            previous = (bid, operation_key, op["value"])
            break
    change = {"status": "no_same_model_reference", "model_id": model_id,
              "comparison": "同一模型、不同批次的描述性比较", "utility_delta": None}
    sources = [{"operation_key": base_key}]
    if previous and baseline:
        bid, key, result = previous
        overlap = len(set(result["sample_ids"]) & set(baseline["sample_ids"]))
        change.update(status="overlapping_samples" if overlap else "available",
                      reference_batch=bid, reference_score=result["score"], current_score=baseline["score"],
                      reference_count=len(result["sample_ids"]), current_count=len(explore), overlap_count=overlap)
        if not overlap:
            change["utility_delta"] = _difference(task.utility(baseline["score"]), task.utility(result["score"]))
        sources.append({"operation_key": key})
    add("performance_change", change, sources)

    features = []
    for index, feature in enumerate(task.features[:DETAIL_LIMIT]):
        recent_mean = scaled_mean([s.values[index] for s in recent])
        history_mean = scaled_mean([s.values[index] for s in history])
        features.append({"feature": feature, "recent_mean": recent_mean, "history_mean": history_mean,
                         "mean_delta": _difference(recent_mean, history_mean)})
    add("input_change", {"status": "available" if recent and history else "insufficient_pools",
                         "comparison": "当前可用近期与历史拟合池的描述性统计",
                         "recent": _pool(task, recent), "history": _pool(task, history),
                         "features": features, "omitted_features": max(0, len(task.features) - DETAIL_LIMIT)},
        [{"role": "fit", "pool": "recent", **_row_source(recent)},
         {"role": "fit", "pool": "history", **_row_source(history)}])

    def attempts(bid, batch_trials):
        records = []
        diagnostic = state["feedback"].get(f"diagnostic:{bid}")
        for trial in batch_trials[:DETAIL_LIMIT]:
            mid = trial.get("model_id")
            evaluation_key = f"{bid}:explore:{mid}" if mid else None
            evaluation_op = state["operations"].get(evaluation_key, {})
            evaluation = evaluation_op.get("value") if evaluation_op.get("status") == "completed" else None
            build_key = trial.get("build_operation_key") or (f"{bid}:build:{trial['action']}" if trial["action"] != "reuse" else None)
            keys = [key for key in (build_key, evaluation_key) if key in state["operations"]]
            build = {**state.get("fit_conditions", {}).get(build_key, {}),
                     **state["operations"].get(build_key, {})}
            raw = trial.get("action_plan") or build.get("requested_action")
            selection = state.get("data_selections", {}).get(selection_key(bid, ActionPlan.parse(raw))) if raw else None
            selection_summary = ({key: deepcopy(value) for key, value in selection.items() if key != "selected_versions"}
                                 if selection else None)
            if selection_summary:
                for field in ("observed_classes", "selected_classes", "missing_observed_classes"):
                    values = selection_summary[field]
                    selection_summary["omitted_"+field] = max(0,len(values)-DETAIL_LIMIT)
                    selection_summary[field] = values[:DETAIL_LIMIT]
            delta = (_difference(task.utility(evaluation["score"]), task.utility(diagnostic["score"]))
                     if evaluation and diagnostic else None)
            records.append({"action": trial["action"], "status": trial["status"], "model_id": mid,
                            "final_resource": trial.get("final_resource", True),
                            "resource_fraction": trial.get("resource_fraction", 1.),
                            "training_count": len(trial.get("training_ids", [])) if trial["action"] != "reuse" else None,
                            "score": evaluation["score"] if evaluation else None,
                            "utility_delta_vs_incumbent": delta,
                            "cost": operation_costs([state["operations"][key] for key in keys]),
                            "operation_keys": keys,
                            "data_selection": selection_summary,
                            "assessment": assess_attempt(task, trial, build,
                                                         evaluation_op, diagnostic)})
        return {"attempts": records, "omitted_attempts": max(0, len(batch_trials) - DETAIL_LIMIT)}

    past_ids = list(state["completed"])[-HISTORY_LIMIT:]
    past = [{"batch_id": bid, **attempts(bid, state["completed"][bid]["trials"]),
             "selected_model": state["completed"][bid]["model_id"],
             "decision_reason": state["completed"][bid]["decision"]["reason"]} for bid in past_ids]
    add("attempts", {"current": attempts(batch.batch_id, trials), "past": past,
                     "omitted_batches": max(0, len(state["completed"]) - HISTORY_LIMIT)},
        [{"completed_batches": past_ids, "current_batch": batch.batch_id}])

    allowed_batches = set(state["completed"]) | {batch.batch_id}
    operations = {key: op for key, op in state["operations"].items() if op["batch_id"] in allowed_batches}
    current_ops = [op for op in operations.values() if op["batch_id"] == batch.batch_id]
    add("resources", {"current_batch": operation_costs(current_ops), "all_visible_batches": operation_costs(list(operations.values())),
                      "remaining_demo_units": config.demo_budget_per_batch - state["ledger"][batch.batch_id]["demo_units"],
                      "units_basis": "预设工程点数，非算力费用", "elapsed_basis": "工具调用墙钟耗时，含失败尝试",
                      "gpu_seconds": None, "model_service_tokens": operation_costs(list(operations.values()))["real_service_tokens"]},
        [{"operation_keys": list(operations), "ledger_batch": batch.batch_id}])

    latest = {}
    for key, feedback in state["feedback"].items():
        if feedback["source"] != "service_feedback" or feedback["received_in"] not in allowed_batches:
            continue
        old = latest.get(feedback["sample_id"])
        if old is None or feedback["label_revision"] > old[1]["label_revision"]:
            latest[feedback["sample_id"]] = (key, feedback)
    by_model = {}
    for key, feedback in latest.values():
        by_model.setdefault(feedback["model_id"], []).append((key, feedback))
    model_feedback = []
    for mid, records in sorted(by_model.items())[:HISTORY_LIMIT]:
        rows = tuple(Sample(f["sample_id"], f["label_revision"], (), f["label"], "explore", "", batch.ordinal)
                     for _, f in records)
        predictions = {f["sample_id"]: f["prediction"] for _, f in records}
        issued = sum(p["model_id"] == mid for p in state["predictions"].values())
        model_feedback.append({"model_id": mid, "returned_count": len(records), "issued_count": issued,
                               "pending_count": issued - len(records), "score": prediction_score(task, rows, predictions)})
    add("service_feedback", {"metric": task.metric, "target_unit": task.target_unit,
                             "scope": "已返回答案的原服务预测，按原模型分开；每个样本取最新已知修订",
                             "models": model_feedback, "omitted_models": max(0, len(by_model) - HISTORY_LIMIT)},
        [{"feedback_keys": [key for key, _ in latest.values()]}])
    return {"recent_count": len(recent), "history_count": len(history),
            "recent_classes": sorted({s.label for s in recent}) if task.task_type == "classification" else [],
            "version": FACTS_VERSION, "evidence": items}, evidence


def validate_facts(facts):
    """检查传给对话接口的事实目录；原始来源记录由控制器本地持有。"""
    legacy = {"recent_count", "history_count", "recent_classes"}
    if set(facts) == legacy:
        return  # 保留独立对话格式样例；完整Harness使用带版本的事实目录。
    if set(facts) != legacy | {"version", "evidence"} or facts["version"] != FACTS_VERSION:
        raise ValueError("事实摘要包含未声明字段或版本不兼容")
    items = facts["evidence"]
    if not isinstance(items, list) or not len(KINDS) <= len(items) <= len(KINDS | EXTRA_KINDS):
        raise ValueError("事实目录不完整")
    seen = set()
    for item in items:
        if (not isinstance(item, dict) or set(item) != {"id", "kind", "value", "evidence_kind"}
                or not isinstance(item["id"], str) or not isinstance(item["value"], dict)
                or item["id"] in seen):
            raise ValueError("事实条目格式无效")
        seen.add(item["id"])
    kinds = {item["kind"] for item in items}
    if not KINDS <= kinds <= KINDS | EXTRA_KINDS or len(kinds) != len(items):
        raise ValueError("事实类别缺失或重复")


def validate_action_evidence(action: ActionPlan, facts):
    """引用必须属于本次上下文；引用存在性不替代对改法效果的检验。"""
    if "evidence" not in facts:
        if action.evidence_refs:
            raise ValueError("旧格式上下文没有可引用事实")
        return
    known = {item["id"] for item in facts["evidence"]}
    if not action.reason.strip() or not action.evidence_refs or not set(action.evidence_refs) <= known:
        raise ValueError("计划需要说明理由并引用本次已提供的事实")


def validate_action_context(action, context):
    """检查当前开放的动作及经验身份，控制器与对话解析共同使用。"""
    validate_action_evidence(action, context["facts"])
    if action.ratio not in context["allowed_recipes"].get(action.recipe, []):
        raise ValueError("动作不属于当前资料目录")
    maintenance = context.get("maintenance", {})
    if "continuation_epochs" in context and action.epochs is None and not action.gas_pair:
        raise ValueError("续训工具需要明确遍数，或选择已开放的气体修正")
    if action.epochs is not None and action.epochs not in context.get("continuation_epochs", ()):
        raise ValueError("当前工具不支持此续训轮数")
    catalog = context.get("fit_manifests")
    if (catalog is not None or action.fit_manifest is not None) and (
            catalog is None or (action.fit_manifest not in catalog["names"]
                                 and not (action.fit_manifest is None and catalog.get("implicit_full") is True))):
        raise ValueError("续训动作须选用本轮已登记的训练清单")
    if action.focus_label is not None and action.focus_label not in context.get("focus_labels", ()):
        raise ValueError("当前工具未开放该类别的定向续训")
    if action.gas_pair and action.gas_pair.method not in context.get("gas_pair_methods", ()):
        raise ValueError("当前工具未开放该误判方向修正方法")
    if action.gas_pair and "gas_active_pairs" in context:
        active = context["gas_active_pairs"]
        if len(active) >= 2 or any((p["reported"], p["target"]) == (action.gas_pair.reported, action.gas_pair.target)
                                   for p in active):
            raise ValueError("当前版本已包含该方向，或已达到连续修正的容量")
    if action.pipeline is not None and not maintenance.get("components"):
        raise ValueError("当前运行未开放组件动作")
    if action.pipeline is not None:
        action.pipeline.validate_task_type(context.get("objective", {}).get("task_type"))
        if action.pipeline.processor == "condition_standard" and not context.get("task", {}).get("condition_features"):
            raise ValueError("当前任务未声明可用工况字段")
        if action.pipeline.predictor not in maintenance["components"]["predictor"]:
            raise ValueError("预测器未在当前目录开放")
        if action.pipeline.processor not in maintenance["components"]["processor"]:
            raise ValueError("处理器未在当前任务目录开放")
    if action.method_id is not None:
        method = next((m for m in maintenance.get("methods", []) if m["id"] == action.method_id), None)
        if method is None or not method["eligible"] or ActionPlan.parse(method["action"]).key != action.key:
            raise ValueError("复用方法身份、范围或实际动作不匹配")
    if action.correction:
        catalog = maintenance.get("corrections", {})
        base = next((b for b in catalog.get("bases", []) if b["model_id"] == action.correction.base_model_id), None)
        if not catalog.get("enabled") or base is None:
            raise ValueError("修正基础模型未在当前目录开放")
        expected = "residual" if context["objective"]["task_type"] == "regression" else "region_replace"
        if action.correction.mode != expected or (action.correction.region_id and action.correction.region_id not in base["region_ids"]):
            raise ValueError("修正目标或冻结范围与当前目录不符")
