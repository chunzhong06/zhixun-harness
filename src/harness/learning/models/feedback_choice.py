"""依据已揭晓的分组误差挑选预装工具。上层负责在反馈可用以后才调用。"""
from math import fsum, isfinite


class HistoricalToolChoice:
    """初值为启动工具；每组等权，同时保留第一次决定以便与后续重选配对。"""
    def __init__(self, order, default):
        if not order or len(set(order)) != len(order) or default not in order:
            raise ValueError("需要不重复的工具和其中一个启动版本")
        self.order = tuple(order)
        self.active = self.first = default
        self.losses = []

    def update(self, losses):
        if not losses or any(set(item) != set(self.order) or not all(isfinite(v) for v in item.values())
                             for item in losses):
            raise ValueError("每组反馈须含全部工具在同一组上的有限误差")
        first_update = not self.losses
        self.losses.extend(dict(item) for item in losses)
        means = {arm: fsum(row[arm] for row in self.losses) / len(self.losses) for arm in self.order}
        self.active = min(self.order, key=means.get)
        if first_update:
            self.first = self.active
        return means
