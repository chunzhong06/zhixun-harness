"""未见工况使用输入父网。新工况取得合法标签后可独立校准，其他工况继续原样交付。"""
from copy import copy

import numpy as np


def group_rows(raw, condition):
    indices, decimals = condition["indices"], condition["decimals"]
    groups = {}
    for i, row in enumerate(raw[:, indices]):
        key = tuple(round(float(v), d) for v, d in zip(row, decimals, strict=True))
        groups.setdefault(key, []).append(i)
    return groups


class EngineParentSupport:
    """原始父版沿用检查点中的输入及目标尺度，四份网络参数在首次工况适配前保存。"""

    def __init__(self, base, arrays, state):
        if base.meta.get("profile") != "C":
            raise ValueError("父版工况读出只支持发动机任务")
        self.parent = copy(base)
        first_layer = arrays["engine_parent_w1"]
        if first_layer.ndim != 2 or first_layer.shape[1] == 0:
            raise ValueError("冻结父网络缺少有效隐藏层")
        # 活跃网络后续可扩宽；留存的父预测按进入首轮时自身的层宽重载。
        hidden = first_layer.shape[1]
        shapes = {"w1":(len(base.meta["paths"]),hidden), "b1":(hidden,), "w2":(hidden,1), "b2":(1,)}
        self.parent.meta = {**base.meta, "hidden":hidden}
        self.parent.params = {}
        for key, shape in shapes.items():
            value = arrays["engine_parent_" + key]
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError("冻结父网络参数形状或数值异常")
            self.parent.params[key] = value
        self.condition = state["condition"]
        spec = state["engine_parent_support"]
        self.quarter_pullback = spec.get("quarter_pullback", False)
        if type(self.quarter_pullback) is not bool:
            raise ValueError("发动机父网回调标记必须为布尔值")
        self.support = {tuple(key) for key in spec["groups"]}
        self.calibration = {tuple(row["key"]): (row["slope"], row["offset"]) for row in spec["calibration"]}
        width = len(self.condition["indices"])
        if (not self.support or self.support & self.calibration.keys()
                or any(len(key) != width or not np.isfinite(key).all() for key in self.support | self.calibration.keys())
                or any(slope < 0 or not np.isfinite((slope, offset)).all() for slope,offset in self.calibration.values())):
            raise ValueError("父版读出的工况或校准参数无效")

    def mix(self, raw, value, derivative):
        for key, indices in group_rows(raw, self.condition).items():
            if key in self.support:
                if self.quarter_pullback:
                    # 两版只在已支持的工况中比较。回调量和训练导数用相同的活动行。
                    parent = self.parent.predict(raw[indices])
                    excess = value[indices]-parent
                    value[indices] -= .25*np.maximum(excess,0.)
                    derivative[indices] *= np.where(excess>0.,.75,1.)
                continue
            pred = self.parent.predict(raw[indices])
            slope, offset = self.calibration.get(key, (1., 0.))
            value[indices] = np.maximum(0., slope * pred + offset)
            derivative[indices] = 0.
        return value, derivative

    def fit_released(self, raw, target):
        """只改变本轮新标签覆盖的后备工况。源预测恒定时，拟合退化为均值校准。"""
        fitted = {}
        for key, indices in group_rows(raw, self.condition).items():
            if key in self.support or len(indices) < 8:
                continue
            pred = self.parent.predict(raw[indices])
            y = target[indices]
            centered = pred - pred.mean()
            variance = centered @ centered
            slope = max(0., float(centered @ (y-y.mean()) / variance)) if variance > 1e-10 else 1.
            fitted[key] = (slope, float(y.mean() - slope * pred.mean()))
        self.calibration.update(fitted)
        return len(fitted)
