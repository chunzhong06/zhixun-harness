"""评价已冻结的问题范围及范围外表现；最终答案只用于评价侧计算。"""
from copy import deepcopy

from harness.learning.models.corrections import inside


def evaluate_problems(state, scope, records):
    """沿原问题条件读取实际服务预测，保留缺标签、缺类及低支持状态。

    全批答案齐备后才给范围分数，避免尚未知类别的样本被排除。条件、
    原始目标与发现时点来自运行记录，不根据最终效果重新切分区域。
    """
    by_id = {r["sample_id"]: r for r in records}
    batches = {}
    for sid, row in scope.items():
        batches.setdefault(row["batch_id"], []).append(sid)
    result = {}
    minimum = state.get("config", {}).get("problem_min_support", 1)
    for key, problem in state.get("problems", {}).items():
        rule = problem["scope"]
        observations = []
        for bid, ids in batches.items():
            if scope[ids[0]]["prediction_ordinal"] < problem["discovered_ordinal"]:
                continue
            complete = all(sid in by_id for sid in ids)
            rows = [by_id[sid] for sid in ids if sid in by_id]
            def belongs(row):
                if rule == "overall": return True
                if rule.startswith("class:"): return str(row["label"]) == rule.removeprefix("class:")
                if rule.startswith("group:"): return scope[row["sample_id"]]["sample"]["group"] == rule.removeprefix("group:")
                if rule.startswith("region:"): return inside(scope[row["sample_id"]]["sample"]["values"], problem["predicates"])
                raise ValueError("无法评价未知的问题范围")
            inner = [r for r in rows if belongs(r)]
            outer = [r for r in rows if not belongs(r)]
            def measure(members):
                return {"support":len(members), "accuracy":sum(r["label"]==r["prediction"] for r in members)/len(members)
                        if complete and members else None}
            target = problem.get("original_target")
            loss = 1-measure(inner)["accuracy"] if complete and inner else None
            status = ("incomplete" if not complete else "unobserved" if not inner else
                "insufficient_support" if len(inner)<minimum else "target_unset" if target is None else
                "target_met" if loss<=target+1e-12 else "open")
            observations.append({"batch_id":bid,"complete":complete,"status":status,
                "expected":len(ids),"returned":len(rows),"inside":measure(inner),
                "outside":measure(outer),"overall":measure(rows),"target_loss":target,
                "served_models":sorted({r["model_id"] for r in rows})})
        result[key] = {"scope":rule,"predicates":deepcopy(problem.get("predicates")),
            "discovered_batch":problem["discovered_batch"],"original_loss":problem.get("original_loss"),
            "original_target":problem.get("original_target"),"observations":observations,
            "attribution":"冻结范围内外的后续观察；不同批次的变化不自动归因为某次维护"}
    return result
