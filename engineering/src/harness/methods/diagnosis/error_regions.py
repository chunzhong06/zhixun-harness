"""有界联合条件搜索与冻结区域复查；复用合法探索预测，不调用模型。

吸收SliceLine的联合谓词与支持数，采用本项目的中位数二分和指标差排序；
不移植其完整剪枝/评分算法。增量SliceLine关于模型变化后误差需更新的条件
落实为每次使用当前预测，并分别标记同模型复现与换模型后的区域表现。见N085。
"""
from copy import deepcopy
from itertools import combinations, product
import math

from harness.contracts import digest
from harness.evaluation.metrics import prediction_score


REGION_VERSION = 2
LIMITS = {"features": 8, "declared_features": 16, "depth": 2, "regions": 4, "history": 16,
          "min_rows": 2, "min_units": 2}


def frozen_regions(state, batch_id):
    """读取已提交批次的冻结条件；复查消耗过的样本/设备不重复当作新证据。"""
    definitions, reviewed = {}, {}
    for bid in state.get("completed", {}):
        if bid == batch_id:
            continue
        for check in state.get("inspections", {}).get(bid, []):
            for region in check.get("regions", []):
                definitions.setdefault(region["id"], deepcopy(region))
            for result in check.get("rechecks", []):
                used = reviewed.setdefault(result["region_id"], {"ids": set(), "units": set()})
                used["ids"].update(result.get("sample_ids", []))
                used["units"].update(result.get("unit_ids", []))
    selected = list(definitions.values())[-LIMITS["history"]:]
    for region in selected:
        used = reviewed.get(region["id"], {"ids": set(), "units": set()})
        region["reviewed_ids"] = sorted(used["ids"])
        region["reviewed_units"] = sorted(used["units"])
    return selected


def _inside(row, predicates):
    return all((row.values[p["index"]] <= p["cut"]) if p["side"] == "le" else
               (row.values[p["index"]] > p["cut"]) for p in predicates)


def _assessment(task, rows, predictions, predicates):
    inside = tuple(s for s in rows if _inside(s, predicates))
    outside = tuple(s for s in rows if not _inside(s, predicates))
    result = {"count": len(inside), "outside_count": len(outside),
              "unit_count": len({s.unit_id for s in inside if s.unit_id is not None}),
              "outside_unit_count": len({s.unit_id for s in outside if s.unit_id is not None}),
              "unknown_units": sum(s.unit_id is None for s in rows),
              "status": "insufficient_rows", "gap": None}
    if min(len(inside), len(outside)) < LIMITS["min_rows"]:
        return result
    a = prediction_score(task, inside, {s.sample_id: predictions[s.sample_id] for s in inside})
    b = prediction_score(task, outside, {s.sample_id: predictions[s.sample_id] for s in outside})
    gap = task.utility(b) - task.utility(a)
    result.update(status="available" if math.isfinite(gap) else "numeric_range",
                  inside_score=a, outside_score=b, gap=gap if math.isfinite(gap) else None)
    return result


def inspect_regions(task, recent, explore, baseline, threshold, *, ordinal=None,
                    history=(), training_rows=()):
    """发现最多二条件区域，冻结后在新批次、未见设备上做描述性复查。

    threshold采用任务原指标的差值单位。分箱只读取fit；区域排序读取explore。
    复查排除发现时的全部fit/explore/模型拟合样本及设备，并排除过去已复查
    单位；缺少独立单位声明时保留未知。采用集与固定参照评价集不在接口内。
    """
    if any(s.role != "fit" for s in recent) or any(s.role != "explore" for s in explore):
        raise ValueError("错误区域检查只接受fit与explore资料")
    predictions = baseline["predictions"]
    if explore:
        prediction_score(task, explore, predictions)
    source_rows = (*recent, *explore, *training_rows)
    excluded_ids = sorted({s.sample_id for s in source_rows})
    excluded_units = sorted({s.unit_id for s in source_rows if s.unit_id is not None})
    complete_units = bool(source_rows) and all(s.unit_id is not None for s in source_rows)
    cuts = []
    # 任务可声明有物理含义的检查范围。未声明时均匀覆盖输入位置，避免总看前几列。
    selected_features = task.diagnostic_features or tuple(
        task.features[i * len(task.features) // min(len(task.features), LIMITS["features"])]
        for i in range(min(len(task.features), LIMITS["features"])))
    for feature in selected_features:
        index = task.features.index(feature)
        ordered = sorted(s.values[index] for s in recent)
        if len(ordered) < 2 or ordered[0] == ordered[-1]:
            continue
        mid = len(ordered) // 2
        cut = ordered[mid] if len(ordered) % 2 else ordered[mid - 1] / 2 + ordered[mid] / 2
        cuts.append((index, feature, cut))
    candidates, seen, searched, evaluable = [], set(), 0, 0
    for depth in range(1, LIMITS["depth"] + 1):
        for selected in combinations(cuts, depth):
            for sides in product(("le", "gt"), repeat=depth):
                predicates = [{"index": i, "feature": f, "cut": cut, "side": side}
                              for (i, f, cut), side in zip(selected, sides)]
                members = tuple(sorted(s.sample_id for s in explore if _inside(s, predicates)))
                searched += 1
                if members in seen:
                    continue
                seen.add(members)
                assessment = _assessment(task, explore, predictions, predicates)
                evaluable += int(assessment["gap"] is not None)
                if assessment["gap"] is not None and assessment["gap"] > threshold:
                    candidates.append((assessment, predicates))
    candidates.sort(key=lambda item: (-item[0]["gap"], -item[0]["count"], len(item[1]), digest(item[1])))
    regions = []
    for assessment, predicates in candidates[:LIMITS["regions"]]:
        region = {"task_signature": task.signature, "version": REGION_VERSION,
                  "model_id": baseline["model_id"], "ordinal": ordinal, "threshold": threshold,
                  "predicates": predicates, "discovery": assessment,
                  "binning": "fit_median_v1", "fit_versions": sorted(f"{s.sample_id}@{s.revision}" for s in recent),
                  "explore_versions": sorted(f"{s.sample_id}@{s.revision}" for s in explore),
                  "excluded_ids": excluded_ids, "excluded_units": excluded_units,
                  "complete_units": complete_units}
        region["id"] = "region-" + digest(region)[:24]
        regions.append(region)
    rechecks = []
    for region in history:
        if region["task_signature"] != task.signature or region["version"] != REGION_VERSION:
            continue
        excluded = set(region["excluded_ids"]) | set(region.get("reviewed_ids", []))
        units = set(region["excluded_units"]) | set(region.get("reviewed_units", []))
        # 当前模型也可能已使用新设备训练；这部分不能成为独立复查资料。
        excluded.update(s.sample_id for s in (*recent, *training_rows))
        units.update(s.unit_id for s in (*recent, *training_rows) if s.unit_id is not None)
        fresh = tuple(s for s in explore if region["ordinal"] is not None and ordinal is not None
                      and ordinal > region["ordinal"] and s.available_at > region["ordinal"]
                      and s.sample_id not in excluded and s.unit_id is not None and s.unit_id not in units)
        assessment = _assessment(task, fresh, predictions, region["predicates"])
        status = "insufficient_fresh_units"
        if not region["complete_units"]:
            status = "discovery_units_unknown"
        elif any(s.unit_id is None for s in (*recent, *training_rows)):
            status = "current_training_units_unknown"
        elif (assessment["gap"] is not None and
              min(assessment["unit_count"], assessment["outside_unit_count"]) >= LIMITS["min_units"]):
            status = "reproduced" if assessment["gap"] > region["threshold"] else "not_reproduced"
            if baseline["model_id"] != region["model_id"]:
                status = "changed_model_observation"
        rechecks.append({"region_id": region["id"], "status": status, "assessment": assessment,
                        "discovery_model": region["model_id"], "current_model": baseline["model_id"],
                        "threshold": region["threshold"], "sample_ids": [s.sample_id for s in fresh],
                        "sample_versions": [f"{s.sample_id}@{s.revision}" for s in fresh],
                        "unit_ids": sorted({s.unit_id for s in fresh})})
    return {"signal": bool(regions) if evaluable else None,
            "regions": regions, "rechecks": rechecks, "observations": [r["discovery"] for r in regions],
            "searched_conditions": searched, "evaluable_conditions": evaluable, "limits": dict(LIMITS),
            "selected_features": list(selected_features),
            "feature_selection": "task_declared" if task.diagnostic_features else "even_coverage",
            "omitted_features": len(task.features) - len(selected_features),
            "objective": task.objective, "ranking": "gap_then_support",
            "interpretation": "探索所得联合条件；冻结后用新单位复查，尚非物理原因或显著性结论"}
