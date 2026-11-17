"""按处理分歧安排只读检查：新增证据进入下一次计划，不直接决定模型采用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from harness.core.contracts import ActionPlan, digest
from harness.evaluation.metrics import prediction_score, scaled_mean
from harness.research.methods.diagnosis.tool_registry import CHECK_TOOLS
from harness.research.methods.diagnosis.error_regions import REGION_VERSION, LIMITS, inspect_regions, frozen_regions


def check_catalog():
    return {
        "feature_scale": {"input": "当前合法fit资料", "signal": "非恒定特征的跨度比超过threshold",
                          "threshold": "finite >= 1", "cost_demo_units": 1},
        "tail_profile": {"input": "当前合法fit资料，至少4条", "signal": "单特征全距/四分位距超过threshold",
                         "threshold": "finite >= 1；恒定和四分位距为零分别记录", "threshold_min": 1,
                         "cost_demo_units": 1},
        "error_slices": {"input": "fit中位数及当前探索预测/标签", "signal": "同一特征两侧指标差超过threshold",
                         "threshold": "finite >= 0，采用任务原指标单位；每侧至少2条", "cost_demo_units": 1},
        "error_regions": {"input": "fit分箱、探索预测及已冻结区域", "signal": "联合条件内表现差于区域外超过threshold",
                          "threshold": "finite >= 0，任务原指标差值；探索信号与后续复查分别记录",
                          "version": REGION_VERSION, "limits": dict(LIMITS), "cost_demo_units": 1},
        **CHECK_TOOLS.catalog(),
    }


def check_identity(plan, task, recent, explore, model_id, *, history=()):
    """用计算输入和当前工具版本标识可复用的观察。"""
    tool = check_catalog()[plan.tool]
    return digest({
        "task": task.signature, "tool": plan.tool, "threshold": plan.threshold,
        "tool_version": tool.get("source_sha256", tool.get("version")),
        "region_history": history if plan.tool == "error_regions" else None,
        "fit": [asdict(s) for s in recent],
        "explore": [asdict(s) for s in explore] if plan.tool != "feature_scale" else None,
        "model_id": model_id if plan.tool != "feature_scale" else None,
    })


def select_checks(plans, task, recent, explore, model_id, state, config, round_index=0):
    """按能改变的未尝试动作数/声明成本排序；另按固定轮次保留探索位置。

    这是可检查的覆盖启发式，不将计划器文字信心当作概率。已完成计算仍可
    为新分支提供原结果，成本记为零；相同数据上两种后续动作均已试过时跳过。
    """
    if config.inspection_order == "planner":
        return plans, []
    attempted = {o["requested_action"] and ActionPlan.parse(o["requested_action"]).key
                 for o in state["operations"].values()
                 if o.get("requested_action") and o["batch_id"] == state["pending"]["batch_id"]}
    ranked, notes = [], []
    for index, plan in enumerate(plans):
        branches = {plan.if_true.key,plan.if_false.key}
        novel = len(branches-attempted)
        history = frozen_regions(state, state['pending']['batch_id']) if plan.tool == "error_regions" else ()
        identity = check_identity(plan,task,recent,explore,model_id,history=history)
        key = f"{state['pending']['batch_id']}:inspect:adaptive:{identity}"
        cached = state["operations"].get(key,{}).get("status") == "completed"
        usable = len(recent)>=(4 if plan.tool == "tail_profile" else 2) and (plan.tool not in {"error_slices", "error_regions"} or len(explore)>=4)
        cost = 0 if cached else check_catalog()[plan.tool]["cost_demo_units"]
        priority = novel / max(1,cost)
        notes.append({"tool":plan.tool,"novel_branches":novel,"estimated_demo_units":cost,
                      "usable":usable,"cached":cached,"priority":priority})
        if novel and usable:
            ranked.append((not cached,-priority,index,plan))
    ranked.sort(key=lambda item:item[:3])
    # 定期让一个尚未选中的有效检查获得位置，避免固定排序总压住同一工具。
    if len(ranked)>config.max_checks and (round_index+1)%config.inspection_exploration_every==0:
        choice = ranked[config.max_checks:][round_index % (len(ranked)-config.max_checks)]
        ranked.insert(0,ranked.pop(ranked.index(choice)))
    return [item[3] for item in ranked[:config.max_checks]], notes


@dataclass(frozen=True)
class CheckPlan:
    """两种检查结果须对应不同的可执行改法；threshold为检查信号的判定值。"""
    tool: str
    question: str
    threshold: float
    if_true: ActionPlan
    if_false: ActionPlan
    problem_id: str | None = None

    @classmethod
    def parse(cls, value):
        required = {"tool", "question", "threshold", "if_true", "if_false"}
        if not isinstance(value, dict) or not required <= set(value) <= required | {"problem_id"}:
            raise ValueError("检查计划字段不完整")
        tool, threshold = value["tool"], value["threshold"]
        if tool not in check_catalog() or not isinstance(value["question"], str) or not value["question"].strip():
            raise ValueError("检查工具或问题无效")
        minimum = check_catalog()[tool].get("threshold_min",1 if tool == "feature_scale" else 0)
        if type(threshold) not in {int, float} or not math.isfinite(threshold) or threshold < minimum:
            raise ValueError("检查阈值无效")
        yes, no = ActionPlan.parse(value["if_true"]), ActionPlan.parse(value["if_false"])
        if yes.key == no.key:
            raise ValueError("检查结果没有改变可执行决定")
        pid=value.get("problem_id")
        if pid is not None and (not isinstance(pid,str) or not pid):
            raise ValueError("检查问题身份无效")
        return cls(tool, value["question"], threshold, yes, no,pid)


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else scaled_mean(ordered[mid - 1:mid + 1])


def _tail_profile(task, rows, threshold):
    """比较全距与中部跨度；仅描述分布，不删除样本或认定传感器异常。

    四分位数采用排序位置(n-1)*q的线性插值。先按最大绝对值缩放，
    避免有限极值相减溢出；中部跨度为零的离散分布保留单独状态。
    """
    observations, signals = [], []
    for j, name in enumerate(task.features):
        raw = sorted(row.values[j] for row in rows)
        item = {"feature": name, "count": len(raw), "status": "insufficient_samples"}
        if len(raw) >= 4:
            scale = max(abs(value) for value in raw) or 1.
            values = [value / scale for value in raw]
            def quantile(q):
                position = (len(values) - 1) * q
                index = int(position)
                weight = position - index
                return values[index] * (1 - weight) + values[min(index + 1, len(values) - 1)] * weight
            q1, q3 = quantile(.25), quantile(.75)
            span, iqr = values[-1] - values[0], q3 - q1
            item.update(scale=scale, normalized_q1=q1, normalized_q3=q3,
                        normalized_span=span, normalized_iqr=iqr)
            if raw[0] == raw[-1]:
                item["status"] = "constant"
                signals.append(False)
            elif iqr <= 0:
                item["status"] = "zero_iqr"
            else:
                log_ratio = math.log(span) - math.log(iqr)
                item.update(status="available", range_to_iqr=math.exp(log_ratio) if log_ratio < 709 else None,
                            ratio_overflow=log_ratio >= 709)
                signals.append(log_ratio > math.log(threshold))
        observations.append(item)
    # 某一有效特征足以给出阳性；阴性须覆盖全部特征，未知状态继续保留。
    signal = True if any(signals) else False if len(signals) == len(observations) else None
    return {"signal": signal, "observations": observations,
            "quantiles": "linear: (n-1)*q", "interpretation": "尾部相对中部的跨度检查；用于安排缩放对照"}


def execute_check(plan, task, recent, explore, baseline, *, region_context=None):
    """只读已允许的fit与explore；不训练、不预测、不接触本批采用资料。

    信号支持下一步试改选择，不能据此认定工业故障原因。缺少样本或
    合法两侧分组时返回signal=None，并保留实际覆盖与来源。
    """
    if any(s.role != "fit" for s in recent) or any(s.role != "explore" for s in explore):
        raise ValueError("检查收到未开放用途的资料")
    sources = {"fit_versions": [f"{s.sample_id}@{s.revision}" for s in recent],
               "explore_versions": [f"{s.sample_id}@{s.revision}" for s in explore],
               "model_id": baseline["model_id"],
               "fit_digest": digest([asdict(s) for s in recent])}
    result = {"tool": plan.tool, "question": plan.question, "threshold": plan.threshold,
              "signal": None, "observations": [], "sources": sources,
              "alternatives": {"true": plan.if_true.to_record(), "false": plan.if_false.to_record()},
              "interpretation": "描述性检查；作用通过共同试改评价"}
    if not recent:
        return result
    if plan.tool == "error_regions":
        result.update(inspect_regions(task, recent, explore, baseline, plan.threshold, **(region_context or {})))
        return result
    if plan.tool == "tail_profile":
        result.update(_tail_profile(task, recent, plan.threshold))
        return result
    if plan.tool in CHECK_TOOLS.entries:
        result.update(CHECK_TOOLS.execute(plan.tool,task=task,fit=recent,explore=explore,
                                         baseline=baseline,threshold=plan.threshold))
        result["tool_manifest"] = check_catalog()[plan.tool]
        return result
    if plan.tool == "feature_scale":
        spreads = []
        for j, name in enumerate(task.features):
            raw = [row.values[j] for row in recent]
            scale = max(abs(x) for x in raw) or 1.
            values = [value / scale for value in raw]
            span = max(values) - min(values)
            log_span = math.log(scale) + math.log(span) if span > 0 else None
            if log_span is not None:
                spreads.append(log_span)
            result["observations"].append({"feature": name, "constant": min(raw) == max(raw), "log_span": log_span})
        # 按对数比较跨度，极大/极小的有限输入也不会把比值写成Infinity。
        log_ratio = max(spreads) - min(spreads) if spreads else None
        result["span_ratio"] = math.exp(log_ratio) if log_ratio is not None and log_ratio < 709 else None
        result["ratio_overflow"] = log_ratio is not None and log_ratio >= 709
        if len(recent) >= 2 and spreads:
            result["signal"] = bool(log_ratio > math.log(plan.threshold))
    else:
        differences = []
        for j, name in enumerate(task.features):
            cut = _median([s.values[j] for s in recent])
            left = tuple(s for s in explore if s.values[j] <= cut)
            right = tuple(s for s in explore if s.values[j] > cut)
            item = {"feature": name, "split": cut, "left_count": len(left), "right_count": len(right)}
            if min(len(left), len(right)) >= 2:
                a = prediction_score(task, left, {s.sample_id: baseline["predictions"][s.sample_id] for s in left})
                b = prediction_score(task, right, {s.sample_id: baseline["predictions"][s.sample_id] for s in right})
                gap = abs(a - b)
                if math.isfinite(gap):
                    differences.append(gap)
                    item.update(left_score=a, right_score=b, gap=gap)
            result["observations"].append(item)
        if differences:
            result["signal"] = max(differences) > plan.threshold
        result["objective"] = task.objective
    return result
