"""把当前气体类别错误转成一个能由随包训练工具执行的续训假设。"""
from harness.contracts import ActionPlan, GasPairPatch
from harness.data.official_gas import LABELS
from collections import Counter


def gas_planning_context(capabilities, batch, baseline, state):
    """整理当轮已揭晓的错判以及在用版本继承的修正；保留公共计划器的经验列表。"""
    details = {}
    if capabilities.get("focus_labels"):
        details["focus_labels"] = list(capabilities["focus_labels"])
    if not baseline or not capabilities.get("gas_pair_methods"):
        return details
    predictions = baseline["predictions"]
    counts = Counter((s.label, predictions[s.sample_id]) for s in batch.samples
                     if s.role == "explore" and s.sample_id in predictions and s.label != predictions[s.sample_id])
    active = state["current"]
    details.update(gas_pair_methods=list(capabilities["gas_pair_methods"]),
                   gas_confusions=[{"target": t, "reported": r, "count": n}
                                   for (t, r), n in sorted(counts.items(), key=lambda v: (-v[1], v[0]))],
                   gas_has_adopted_candidate=state["models"][active]["parent_id"] is not None)
    if capabilities.get("gas_sequence"):
        # 只沿在用模型的父链回溯，曾尝试但被拒绝的方向不会随下一轮发布。
        recipes = {e["model_id"]: e["effective_recipe"] for e in state["events"] if e["kind"] == "bundle_saved"}
        pairs = []
        while active:
            if pair := recipes[active].get("gas_pair"):
                pairs.append(pair)
            active = state["models"][active]["parent_id"]
        details["gas_active_pairs"] = pairs[::-1]
    return details


def gas_issue(context):
    """只看当时的探索预测与训练覆盖，采用集和后来返回的答案均不参与提议。"""
    facts = {f["kind"]: f for f in context["facts"]["evidence"]}
    errors = facts["performance"]["value"]["error_profile"]["classes"]
    coverage = {x["value"]: x["count"] for x in facts["input_change"]["value"]["recent"]["targets"]["counts"]}
    observed = [r for r in errors if r["missed"] or r["false_positive"]]
    if not observed:
        return None
    # 以类别F1找最受影响的一类。相等时选择漏判较多的类别，最后按固定类别号。
    row = min(observed, key=lambda r: (2*r["correct"] / (r["count"] + r["predicted_count"]),
                                       -r["missed"], r["label"]))
    return {**row, "name": LABELS[row["label"]-1], "training_count": coverage.get(row["label"], 0),
            "evidence_refs": [facts["performance"]["id"], facts["input_change"]["id"]]}


class GasContinuationPlanner:
    identity = "official-gas-local-continuation-v1"

    def propose(self, context):
        issue = gas_issue(context)
        if not issue or not issue["training_count"] or any(t["action"] != "reuse" for t in context["trials"]):
            return {"stop": True, "actions": []}
        # 漏判占主导时尝试多学习两轮；误报占主导时只做一个小步。
        # 这是可被共同验证推翻的候选规则，动作本身不承诺修复任何类别。
        epochs = 3 if issue["missed"] > issue["false_positive"] else 1
        reason = (f"{issue['name']}在探索资料中漏判{issue['missed']}条、误判为该类{issue['false_positive']}条，"
                  f"公开训练集有{issue['training_count']}条该类样本。沿当前权重再学{epochs}轮，检查六类综合表现。")
        action = ActionPlan("recent", reason=reason, epochs=epochs, evidence_refs=tuple(issue["evidence_refs"]))
        return {"stop": False, "actions": [action.to_record()]}


class FocusedGasPlanner(GasContinuationPlanner):
    """当有漏判且训练中有该类样本时，提高它在同等训练量中的占比。"""
    identity = "official-gas-focus-sampling-v1"

    def propose(self, context):
        reply = super().propose(context)
        if not reply["actions"]:
            return reply
        issue = gas_issue(context)
        if issue["missed"] > issue["false_positive"] and issue["label"] in context.get("focus_labels", ()):
            action = reply["actions"][0]
            action["focus_label"] = issue["label"]
            action["reason"] += "保持每遍样本总量，增加该类的抽取次数，并在共同样本上复查。"
        return reply


class GasPairPlanner:
    def __init__(self, method):
        self.method = method
        self.identity = f"gas-confusions-v1-{method}"

    def propose(self, context):
        errors = context.get("gas_confusions", [])
        active = context.get("gas_active_pairs")
        if active is not None:
            if len(active) >= 2:
                return {"actions": [], "stop": True}
            used = {(p["reported"], p["target"]) for p in active}
            errors = [e for e in errors if (e["reported"], e["target"]) not in used]
        elif context.get("gas_has_adopted_candidate"):
            return {"actions": [], "stop": True}
        if context["trials"] or not errors or errors[0]["count"] < 2:
            return {"actions": [], "stop": True}
        err = errors[0]
        fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
        pair = GasPairPatch(err["reported"], err["target"], self.method)
        reason = f"本轮有{err['count']}条{LABELS[pair.target-1]}被判为{LABELS[pair.reported-1]}，优先复查这一方向。"
        action = ActionPlan("recent", reason=reason, evidence_refs=(fact["id"],), gas_pair=pair)
        return {"actions": [action.to_record()], "stop": False}


class FirstWindowPairPlanner:
    """首轮用当时的探索错误建清单，随后只跳过已采用方向；不读取以后各轮的错误排名。"""
    identity = "gas-first-window-fixed-directions-v1"

    def __init__(self, method, first_context=None):
        self.method = method
        self.ranked = None if first_context is None else self._rank(first_context)

    @classmethod
    def restore(cls, method, state, first_batch):
        """接续首轮定序时只恢复首份观察，不用后续已回标资料重排。"""
        first = next((e["context"] for e in (state or {}).get("events", [])
                      if e["key"]==f"{first_batch}:plan_context:0"),None)
        return cls(method,first)

    @staticmethod
    def _rank(context):
        if context["past_decisions"]:
            raise ValueError("固定方向清单须来自首轮、作出首次决定前的观察")
        return [(e["reported"], e["target"]) for e in context["gas_confusions"] if e["count"] >= 2][:2]

    def propose(self, context):
        if self.ranked is None:
            self.ranked = self._rank(context)
        used = {(p["reported"],p["target"]) for p in context["gas_active_pairs"]}
        remaining = [pair for pair in self.ranked if pair not in used]
        if context["trials"] or not remaining:
            return {"actions": [], "stop": True}
        reported, target = remaining[0]
        fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
        action = ActionPlan("recent", reason="沿首轮冻结顺序尝试尚未采用的误判方向",
                            evidence_refs=(fact["id"],), gas_pair=GasPairPatch(reported,target,self.method))
        return {"actions":[action.to_record()],"stop":False}
