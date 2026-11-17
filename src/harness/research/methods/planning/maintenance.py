"""按已释放的服务误差决定维护时机；检测状态和批次决定随运行保存。"""
from __future__ import annotations

import math


def validate_policy(config):
    if config.maintenance_trigger not in {"never", "every_batch", "periodic", "page_hinkley"}:
        raise ValueError("未知维护进入规则")
    for value in (config.maintenance_interval, config.maintenance_probe_interval, config.drift_min_samples):
        if type(value) is not int or value < 1:
            raise ValueError("维护间隔和最少反馈数量须为正整数")
    if not math.isfinite(config.drift_threshold) or config.drift_threshold <= 0 or not math.isfinite(config.drift_delta) or config.drift_delta < 0:
        raise ValueError("变化检测阈值无效")
    if config.loss_scale is not None and (not math.isfinite(config.loss_scale) or config.loss_scale <= 0):
        raise ValueError("回归误差尺度须为正数")


def maintenance_entry(state, task, batch, config):
    """Page-Hinkley上升检测使用当前服务模型尚未消费的误差。

    来源为Page-Hinkley成熟累计偏差公式，参照River固定版本实现，见研究综述。
    报警以本批可见反馈为一段并消费；下一段重新建立均值，参照River报警后
    reset的生命周期。未消费答案的修订重建当前段，已消费答案的修订保留在
    服务评价中，不重新触发旧事件。每批冻结一次决定。周期探查
    覆盖缺少反馈的时段；回归须由调用者声明业务误差尺度，不用当前误差自归一化。
    """
    decisions = state.setdefault("maintenance_entries", {})
    if batch.batch_id in decisions:
        return decisions[batch.batch_id]
    previous = list(decisions.values())
    position = len(previous)
    last_open = max((i for i, item in enumerate(previous) if item["enter"]), default=-1)
    enter, reason, statistic, count = True, "every_batch", None, 0
    if config.maintenance_trigger == "never":
        # 静态服务只保留初始化与预测；首批选型另用every_batch取得探索评价。
        enter, reason = False, "maintenance_disabled"
    elif config.maintenance_trigger == "periodic":
        enter = position % config.maintenance_interval == 0
        reason = "period_due" if enter else "between_periods"
    elif config.maintenance_trigger == "page_hinkley":
        if task.task_type != "classification" and config.loss_scale is None:
            raise ValueError("回归变化检测需要声明loss_scale")
        # 新版本从自己的服务输出建立参照，不混合其他模型产生的误差。
        latest = {}
        for value in state["feedback"].values():
            if value["source"] == "service_feedback" and value["model_id"] == state["current"]:
                sid = value["sample_id"]
                if sid not in latest or value["label_revision"] > latest[sid]["label_revision"]:
                    latest[sid] = value
        batch_order = {key:index for index,key in enumerate(state["completed"])}
        ordered = sorted(latest.values(), key=lambda x: (
            batch_order[state["predictions"][x["sample_id"]]["batch_id"]], x["sample_id"]))
        consumed_entry = next(((bid, item) for bid, item in reversed(list(decisions.items()))
                               if item.get("model_id") == state["current"] and item["reason"] == "error_increase"), None)
        consumed_ids = {sid for sid, _ in consumed_entry[1]["feedback_versions"]} if consumed_entry else set()
        active = [row for row in ordered if row["sample_id"] not in consumed_ids]
        mean, cumulative, minimum = 0., 0., math.inf
        alarm = False
        for active_count, row in enumerate(active, 1):
            loss = float(not row["correct"]) if task.task_type == "classification" else row["absolute_error"] / config.loss_scale
            mean += (loss - mean) / active_count
            cumulative += loss - mean - config.drift_delta
            minimum = min(minimum, cumulative)
            statistic = cumulative - minimum
            alarm = alarm or (active_count >= config.drift_min_samples and statistic > config.drift_threshold)
        count = len(ordered)
        revisions = sorted([x["sample_id"], x["label_revision"]] for x in ordered)
        due = position == 0 or position - last_open >= config.maintenance_probe_interval
        enter = alarm or due
        reason = "error_increase" if alarm else "periodic_probe" if due else "observe_service_feedback"
    result = {"enter": enter, "reason": reason, "policy": config.maintenance_trigger,
              "model_id": state["current"], "feedback_count": count, "statistic": statistic,
              "feedback_versions": revisions if config.maintenance_trigger == "page_hinkley" else []}
    if config.maintenance_trigger == "page_hinkley":
        result.update(detector_feedback_count=len(active),
                      consumed_through_entry=consumed_entry[0] if consumed_entry else None,
                      active_feedback_versions=[[x["sample_id"], x["label_revision"]] for x in active])
    # JSON round-trip稳定，样本修订二元组写为列表。
    result["feedback_versions"] = [list(x) for x in result["feedback_versions"]]
    decisions[batch.batch_id] = result
    return result
