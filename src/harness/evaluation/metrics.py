"""分类和回归的共同评分以及数值摘要。"""
import math
from collections import Counter

from harness.core.contracts import Sample, TaskSpec


def scaled_mean(values):
    """缩放后求有限值的平均数，避免中间求和溢出；空输入没有平均数。"""
    if not values:
        return None
    scale = max(abs(value) for value in values)
    average = math.fsum((value / scale) / len(values) for value in values) if scale else 0.
    return scale * max(-1., min(1., average))


def macro_f1(pairs):
    """与官方气体参考实现一致：在答案和预测出现过的类别并集上等权平均。"""
    expected, predicted, hits = Counter(), Counter(), Counter()
    for truth, guess in pairs:
        expected[truth] += 1
        predicted[guess] += 1
        if truth == guess:
            hits[truth] += 1
    observed = expected.keys() | predicted.keys()
    if not observed:
        raise ValueError("Macro-F1 需要至少一项预测")
    return math.fsum(2 * hits[c] / (expected[c] + predicted[c]) for c in observed) / len(observed)


def error_profile(task, rows, predictions, limit=16):
    """复用合法评价预测描述错误结构，不增加模型调用或修改主评分。

    类别缺少答案时召回率保留未知；连续目标的高估/低估按预测减真值定义。
    最大误差占比只描述误差集中程度，不能据此把观测标记为噪声。
    """
    prediction_score(task, rows, predictions)
    if task.task_type == "classification":
        support = Counter(s.label for s in rows)
        predicted = Counter(predictions[s.sample_id] for s in rows)
        correct = Counter(s.label for s in rows if predictions[s.sample_id] == s.label)
        classes = [{"label": label, "count": support[label], "predicted_count": predicted[label], "correct": correct[label],
                    "missed": support[label] - correct[label], "false_positive": predicted[label] - correct[label],
                    "recall": correct[label] / support[label] if support[label] else None}
                   for label in task.classes]
        classes.sort(key=lambda c: (c["recall"] is None, c["recall"] or 0., c["label"]))
        return {"task_type": task.task_type, "accuracy": sum(correct.values()) / len(rows), "classes": classes[:limit],
                "omitted_classes": max(0, len(classes)-limit), "count": len(rows)}
    errors = [predictions[s.sample_id] - s.label for s in rows]
    scale = max(abs(e) for e in errors)
    scaled = [e / scale for e in errors] if scale else [0.] * len(errors)
    tail_count = max(1, math.ceil(len(errors) * .1))
    mass = math.fsum(abs(e) for e in scaled)
    return {"task_type": task.task_type, "count": len(rows), "target_unit": task.target_unit,
            "mean_signed_error": scale * max(-1., min(1., math.fsum(e / len(errors) for e in scaled))),
            "overprediction_count": sum(e > 0 for e in errors), "underprediction_count": sum(e < 0 for e in errors),
            "largest_error_count": tail_count,
            "largest_error_absolute_share": math.fsum(sorted((abs(e) for e in scaled), reverse=True)[:tail_count]) / mass if mass else 0.}


def grouped_scores(task, rows, predictions):
    """在原评价行上复用预测计算报告分组；单位覆盖独立记录，不视作置信度。"""
    groups = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    return [{"group": group, "count": len(members),
             "known_unit_count": len({s.unit_id for s in members if s.unit_id is not None}),
             "rows_without_unit": sum(s.unit_id is None for s in members),
             "score": prediction_score(task, tuple(members), {s.sample_id: predictions[s.sample_id] for s in members})}
            for group, members in sorted(groups.items())]


def compare_evaluations(task, baseline, candidate, limit=16):
    """同一批样本修订上的得失摘要；旧记录缺少身份时不补造分组比较。

    用途由调用者控制：探索结果提供给计划器，采用结果在冻结候选后保存。
    负向变化优先显示，完整分组分数仍在各评价回执中；本摘要不增设采用门槛。
    """
    versions = baseline.get("sample_versions")
    if (not versions or sorted(versions) != sorted(candidate.get("sample_versions", []))
            or baseline.get("score_definition") != candidate.get("score_definition")):
        return {"status": "unmatched_or_unversioned", "groups": [], "omitted_groups": 0}
    base = {g["group"]: g for g in baseline.get("groups", [])}
    changes = []
    for group in candidate.get("groups", []):
        before = base.get(group["group"])
        if before is None or before["count"] != group["count"]:
            continue
        delta = task.utility(group["score"]) - task.utility(before["score"])
        changes.append({**group, "baseline_score": before["score"], "utility_delta": delta if math.isfinite(delta) else None})
    changes.sort(key=lambda g: (g["utility_delta"] is None, g["utility_delta"] or 0., g["group"]))
    return {"status": "matched", "baseline_model": baseline.get("model_id"),
            "candidate_model": candidate.get("model_id"), "sample_count": len(versions),
            "group_count": len(changes), "regressed_groups": sum(g["utility_delta"] is not None and g["utility_delta"] < 0 for g in changes),
            "groups": changes[:limit], "omitted_groups": max(0, len(changes)-limit)}


def confirmation_scope(task_id, explored, confirmed):
    """比较同一候选两次评价的来源，仅描述覆盖。历史回执缺少完整身份时返回空。"""
    if not explored or not confirmed or not explored.get("model_id") or explored["model_id"] != confirmed.get("model_id"):
        return None
    if explored.get("purpose") != "explore" or confirmed.get("purpose") != "adopt":
        return None
    sources = []
    identities = []
    for receipt in (explored, confirmed):
        ids = receipt.get("sample_ids", [])
        groups = receipt.get("groups", [])
        if not ids or len(set(ids)) != len(ids) or sum(g["count"] for g in groups) != len(ids):
            return None
        sources.append({g["group"] for g in groups})
        identities.append(set(ids))
    before, after = sources
    # 官方气体 group 是采集批，官方轴承 group 是整根设备；其他任务沿用原声明。
    kind = {"official-gas-A": "collection_batch", "official-bearing-B": "bearing"}.get(task_id, "reported_group")
    return {"source_kind": kind, "exploration_sources": sorted(before), "confirmation_sources": sorted(after),
            "new_confirmation_sources": sorted(after - before),
            "overlapping_sample_count": len(identities[0] & identities[1])}


def prediction_score(task: TaskSpec, rows: tuple[Sample, ...], predictions: dict) -> float:
    """按同一批合法答案计算原始指标，空集及非有限结果不能用于采用。"""
    if not rows or set(predictions) != {r.sample_id for r in rows}:
        raise ValueError("评价需要非空且身份一致的样本和预测")
    if any(not task.accepts_target(r.label) or not task.accepts_target(predictions[r.sample_id]) for r in rows):
        raise ValueError("评价目标或预测与任务不兼容")
    if task.metric == "accuracy":
        return sum(predictions[r.sample_id] == r.label for r in rows) / len(rows)
    if task.metric == "macro_f1":
        return macro_f1((r.label, predictions[r.sample_id]) for r in rows)
    errors = [abs(predictions[r.sample_id] - r.label) for r in rows]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("预测误差超出数值范围")
    if task.metric == "mae":
        return math.fsum(e / len(errors) for e in errors)
    # 先缩放再平方，避免有效大数在计算RMSE中间量时溢出。
    scale = max(errors)
    return scale * math.sqrt(math.fsum((e / scale) ** 2 for e in errors) / len(errors)) if scale else 0.0


def exploration_windows(task, rows, predictions, returned_ids, current_weight):
    """分别观察本批资料和过去服务的返回答案，以事前权重计算共同选取指标。

    返回答案可帮助保留旧能力，其行数不隐式改变窗口权重。None保留原按行
    汇总作对照；只有一个窗口时用该窗口。RMSE在均方误差上加权后开方。
    """
    windows = {}
    for name, members in (("current",tuple(r for r in rows if r.sample_id not in returned_ids)),
                          ("returned",tuple(r for r in rows if r.sample_id in returned_ids))):
        if not members:
            continue
        values={r.sample_id:predictions[r.sample_id] for r in members}
        windows[name]={"score":prediction_score(task,members,values),"count":len(members),
            "sample_versions":[f"{r.sample_id}@{r.revision}" for r in members],
            "groups":grouped_scores(task,members,values),"error_profile":error_profile(task,members,values)}
    pooled=prediction_score(task,rows,predictions)
    weights={name:(current_weight if name=="current" else 1-current_weight) for name in windows} if current_weight is not None else {}
    if len(windows)==1:
        weights={name:1. for name in windows}
    selected=pooled
    if current_weight is not None and len(windows)==2:
        scale=max(w["score"] for w in windows.values())
        if task.metric=="rmse":
            selected=scale*math.sqrt(math.fsum(weights[n]*(w["score"]/scale)**2 for n,w in windows.items())) if scale else 0.
        else:
            selected=math.fsum(weights[n]*w["score"] for n,w in windows.items())
    return {"score":selected,"pooled_score":pooled,"observation_windows":windows,
            "score_definition":{"kind":"window_weighted" if current_weight is not None else "row_pooled",
                                "current_weight":current_weight,"effective_weights":weights}}
