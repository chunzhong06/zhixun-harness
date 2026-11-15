"""跨批墙钟与服务用量准入；操作记录拥有实际费用，恢复不重复扣费。"""
from __future__ import annotations

import math


def validate_limits(config):
    for name in ("run_wall_seconds", "batch_wall_seconds", "output_reserve_seconds"):
        value = getattr(config, name)
        if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
            raise ValueError("运行时间预算须为有限非负秒数")
    for name in ("run_service_tokens", "batch_service_tokens", "service_call_reserve_tokens"):
        value = getattr(config, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("服务预算须为非负整数")
    if (config.run_service_tokens is not None or config.batch_service_tokens is not None) and config.service_call_reserve_tokens < 1:
        raise ValueError("启用服务预算须声明单次调用预留量")


class RuntimeBudget:
    """计时从首次接收数据开始，重启及等待时间仍计入持久墙钟期限。

    Token以服务回执结算。缺少回执时保留该次预留量并暂停追加服务调用；
    配置的单次预留量须与实际服务能力对应，服务侧硬配额另行设置。
    本层控制准入，正在执行任务的限时由工作进程和传输适配器执行。
    """
    def __init__(self, state, config, clock):
        self.state, self.config, self.clock = state, config, clock

    def enter(self, batch_id):
        now = self.clock()
        ledger = self.state.setdefault("runtime", {"started_at": now, "batches": {}})
        ledger["batches"].setdefault(batch_id, {"started_at": now})

    def snapshot(self, batch_id):
        ledger = self.state["runtime"]
        now = self.clock()
        remaining = []
        for scope, origin, maximum in (("run", ledger["started_at"], self.config.run_wall_seconds),
                ("batch", ledger["batches"][batch_id]["started_at"], self.config.batch_wall_seconds)):
            if maximum is not None:
                remaining.append(max(0., maximum - max(0., now - origin)))
        costs = {"run": 0, "batch": 0}
        unknown = False
        for operation in self.state["operations"].values():
            if not (operation.get("service_operation") or operation.get("service_reservation_tokens") or operation.get("service_calls")):
                continue
            calls = operation.get("service_calls", [])
            measured = bool(calls) and all(c.get("usage") is not None for c in calls)
            value = sum(c["usage"]["total_tokens"] for c in calls) if measured else operation.get("service_reservation_tokens",0)
            costs["run"] += value
            if operation["batch_id"] == batch_id:
                costs["batch"] += value
            unknown |= not measured
        return {"remaining_seconds": min(remaining) if remaining else None,
                "service_tokens": costs, "unconfirmed_service_usage": unknown,
                "clock_basis": "persistent_wall_time_including_waits"}

    def admit(self, batch_id, *, essential, service):
        view = self.snapshot(batch_id)
        remaining = view["remaining_seconds"]
        reserve = 0. if essential else self.config.output_reserve_seconds
        if remaining is not None and remaining <= reserve:
            return "wall_budget", view
        if service:
            limited = self.config.run_service_tokens is not None or self.config.batch_service_tokens is not None
            if limited and view["unconfirmed_service_usage"]:
                return "service_usage_unconfirmed", view
            for scope, maximum in (("run", self.config.run_service_tokens), ("batch", self.config.batch_service_tokens)):
                if maximum is not None and view["service_tokens"][scope] + self.config.service_call_reserve_tokens > maximum:
                    return "service_token_budget", view
        return None, view
