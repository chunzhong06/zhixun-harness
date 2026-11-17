"""气体服务逐条回执与返回答案的配对。原判是此次预测进入气体修正规则前的类别。"""
from collections import Counter

from harness.core.contracts import GasPairPatch


def validate_repairs(inputs, predictions, repairs):
    """首次执行和恢复使用同一检查，确保每条规则的输出衔接下一步，最终等于对外类别。"""
    ids = {row["sample_id"] for row in inputs}
    if len(ids) != len(inputs) or ids != set(predictions) or ids != set(repairs):
        raise ValueError("气体服务回执与预测身份不一致")
    for sid, trace in repairs.items():
        current = trace["before"]
        if (type(current) is not int or current not in range(1, 7) or not isinstance(trace["steps"], list)
                or type(trace["after"]) is not int or type(predictions[sid]) is not int):
            raise ValueError("气体原判类别无效")
        for step in trace["steps"]:
            rule = GasPairPatch(**step)
            if rule.reported != current:
                raise ValueError("气体回执的修正路径不连续")
            current = rule.target
        if current != trace["after"] or current != predictions[sid]:
            raise ValueError("气体回执的最终类别与服务预测不一致")


def _effect(before, after, truth):
    if before == after:
        return "unchanged_correct" if after == truth else "unchanged_wrong"
    if after == truth:
        return "fixed"
    return "broken" if before == truth else "wrong_to_wrong"


def matches_observation(prediction, sample):
    return (prediction.get("sample_revision") is not None and sample.revision > prediction["sample_revision"]
            and list(sample.values) == prediction.get("input_values") and sample.group == prediction.get("group")
            and sample.unit_id == prediction.get("unit_id"))


def assess_return(prediction, sample):
    trace = prediction.get("gas_repair")
    if not trace:
        return {"outcome": "unknown_trace"}
    # 反馈修订可以更正答案。同一身份的输入或来源也发生改变时，不能用它给先前的修正记功。
    if not matches_observation(prediction, sample):
        return {"outcome": "input_mismatch"}
    return {"outcome": _effect(trace["before"], trace["after"], sample.label), "before": trace["before"],
            "after": trace["after"], "steps": [{**s, "outcome": _effect(s["reported"], s["target"], sample.label)}
                                               for s in trace["steps"]]}


def service_repair_summary(state):
    """每次已发服务预测占一条；答案修订只取最新。旧反馈缺批次时只关联唯一匹配的原记录。"""
    issued = {(bid, sid): (batch["model_id"], pred) for bid, batch in state["completed"].items()
              for sid, pred in batch["predictions"].items()}
    by_identity = {}
    for key, (model, pred) in issued.items():
        by_identity.setdefault((key[1], model, pred), []).append(key)
    latest = {}
    unassigned = {}
    for item in state["feedback"].values():
        if item["source"] == "unmatched_gas_feedback":
            sid = item["sample_id"]
            if sid not in unassigned or item["label_revision"] > unassigned[sid]["label_revision"]:
                unassigned[sid] = item
            continue
        if item["source"] != "service_feedback":
            continue
        candidates = by_identity.get((item["sample_id"], item["model_id"], item["prediction"]), ())
        if "prediction_batch" in item:
            key = (item["prediction_batch"], item["sample_id"])
            if key not in candidates:
                continue
        elif len(candidates) == 1:
            key = candidates[0]
        else:
            continue
        if key not in latest or item["label_revision"] > latest[key]["label_revision"]:
            latest[key] = item
    matched_revisions = Counter()
    counts = Counter()
    rules = {}
    for key in issued:
        item = latest.get(key)
        if item is None:
            counts["pending"] += 1
            continue
        sid = item["sample_id"]
        matched_revisions[sid] = max(matched_revisions[sid], item["label_revision"])
        result = item.get("gas_repair_result", {"outcome": "unknown_trace"})
        counts[result["outcome"]] += 1
        for s in result.get("steps", []):
            rule = (s["reported"], s["target"], s["method"])
            rules.setdefault(rule, Counter())[s["outcome"]] += 1
    return {"issued": len(issued), "returned": len(latest), "outcomes": dict(counts),
            "unassigned_return_ids": sorted(sid for sid, item in unassigned.items()
                                            if matched_revisions[sid] < item["label_revision"]),
            "rules": [{"reported": r, "target": t, "method": m, "outcomes": dict(v)}
                      for (r,t,m),v in sorted(rules.items())]}
