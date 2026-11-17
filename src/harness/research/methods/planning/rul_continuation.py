"""用已揭晓的寿命误差提出一次小步续训，让独立确认样本决定是否采用。"""
from harness.core.contracts import ActionPlan
from collections import defaultdict
from statistics import fmean


def overestimated_group(context):
    fact = next(item for item in context["facts"]["evidence"] if item["kind"] == "performance")
    for group in fact["value"]["groups_worst_first"]:
        if group.get("mean_signed_error", 0) > 0 and group.get("overprediction_count", 0) >= 2:
            return fact["id"], group
    return None


class RulContinuationPlanner:
    identity = "rul-group-overestimate-one-epoch-v1"

    def propose(self, context):
        problem = overestimated_group(context)
        if (not problem or 1 not in context.get("continuation_epochs", ())
                or any(t["action"] != "reuse" for t in context["trials"])):
            return {"stop":True, "actions":[]}
        fact_id, group = problem
        unit = context["objective"]["target_unit"]
        unit = {"minutes":"分钟","cycles":"循环"}.get(unit,unit)
        reason = (f"{group['group']} 有 {group['overprediction_count']} 条寿命高估，"
                  f"预测平均高出 {group['mean_signed_error']:.2f} {unit}。"
                  "从当前版本续训一轮，用共同确认样本检查寿命误差。")
        plan = ActionPlan("recent",epochs=1,reason=reason,evidence_refs=(fact_id,))
        return {"stop":False,"actions":[plan.to_record()]}


def device_biases(rows, predictions):
    """仅在已揭晓的探索资料内，对同一设备按时间比较前后半段误差。"""
    devices = defaultdict(list)
    for row in rows:
        if row.unit_id is not None and row.time_end is not None:
            devices[row.unit_id].append(row)
    records = []
    for unit, observations in devices.items():
        ordered = sorted(observations, key=lambda r: (r.time_end, r.sample_id))
        if len(ordered) < 4 or len({r.time_end for r in ordered}) != len(ordered):
            continue
        errors = [predictions[r.sample_id] - r.label for r in ordered]
        split = len(errors) // 2
        first, recent = fmean(errors[:split]), fmean(errors[split:])
        matching = sum(err * recent > 0 for err in errors[split:])
        records.append({"unit": unit, "observations": len(ordered), "first_bias": first, "recent_bias": recent,
                        "recent_count": len(errors) - split, "matching": matching,
                        "start": ordered[split].time_end, "end": ordered[-1].time_end,
                        "direction_changed": first * recent < 0})
    return sorted(records, key=lambda r: (-abs(r["recent_bias"]), r["unit"]))


class RulDevicePlanner:
    """设备上近期误差方向一致时才提议续训，采用与否由 Harness 在其他设备上核对。"""
    identity = "rul-device-recent-bias-v1"

    def propose(self, context):
        fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
        issue = fact["value"].get("device_bias_issue")
        if not issue or 1 not in context.get("continuation_epochs", ()) or context["trials"]:
            return {"stop": True, "actions": []}
        direction = "高估" if issue["recent_bias"] > 0 else "低估"
        reason = (f"{issue['unit']} 最近 {issue['recent_count']} 个已揭晓观测中有 "
                  f"{issue['matching']} 个{direction}。从当前版本做一遍设备均衡续训，在另外的设备上确认。")
        action = ActionPlan("recent", epochs=1, reason=reason, evidence_refs=(fact["id"],))
        return {"stop": False, "actions": [action.to_record()]}
