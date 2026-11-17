"""把已回标探索资料上的错误差异映射成一项现有工具可以执行的候选。"""
from harness.core.contracts import ActionPlan, GasPairPatch


def corrected_f1(classes, target=None, reported=None, count=0):
    """理论上的纠错收益用于排列提案。真实修正也会误改好样本，须由独立采用集评价。"""
    scores = []
    for row in classes:
        moved_in = count if row["label"] == target else 0
        moved_out = count if row["label"] == reported else 0
        denominator = row["count"] + row["predicted_count"] + moved_in - moved_out
        if denominator:
            scores.append(2 * (row["correct"] + moved_in) / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def repair_action(context, performance):
    active = {(p["reported"], p["target"]) for p in context["gas_active_pairs"]}
    if len(active) >= 2:
        return None
    classes = performance["value"]["error_profile"]["classes"]
    current = corrected_f1(classes)
    options = []
    for error in context["gas_confusions"]:
        reported, target, count = error["reported"], error["target"], error["count"]
        if count >= 2 and (reported, target) not in active:
            gain = corrected_f1(classes, target, reported, count) - current
            options.append((-gain, reported, target))
    if not options:
        return None
    negative_gain, reported, target = min(options)
    return ActionPlan("recent", gas_pair=GasPairPatch(reported, target, "shape-blend"),
                      reason=f"本轮方向 {reported}→{target} 的可纠正 Macro-F1 增量为 {-negative_gain:.5f}，试用受保护修正。",
                      evidence_refs=(performance["id"],))


def training_action(context, performance):
    catalog = context["fit_manifests"]
    total = catalog.get("explore_n", 0)
    options = []
    for field in catalog["fields"]:
        for split in field["splits"]:
            left = split["explore_le"]
            right = total - left
            if min(left, right) < 3:
                continue
            a, b = split["bias_le"], split["bias_gt"]
            # 两组之间可解释的均方误差，按覆盖率加权，单位是寿命单位的平方。
            separation = (left / total) * (right / total) * (a-b)**2
            if separation > 0:
                sides = [(-abs(a), split["le"]), (-abs(b), split["gt"])]
                options.append((-separation, min(sides)[1]))
    if options:
        negative_score, manifest = min(options)
        reason = f"两侧偏差分离量 {-negative_score:.4g}，在偏差较大的 {manifest} 对应官方训练行续训。"
    else:
        manifest = catalog.get("fallback", "full")
        reason = "当前探索记录没有可分辨的偏差组，使用全池资料。"
    if "allocation" in catalog:
        reason += f"本次按 {catalog['allocation']} 配额取 {catalog['fit_budget']} 条官方训练记录。"
    return ActionPlan("recent", epochs=1, fit_manifest=manifest, reason=reason,
                      evidence_refs=(performance["id"],))


class ErrorSlicePlanner:
    identity = "observed-error-slice-v1"

    def propose(self, context):
        action = None
        if not context["trials"]:
            performance = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
            if "gas_active_pairs" in context:
                action = repair_action(context, performance)
            else:
                action = training_action(context, performance)
        return {"actions":[action.to_record()] if action else [], "stop":action is None}
