"""根据本次飞行得到的工况支持比例，为两种已冻结的传感器偏差估计分配权重。"""
import numpy as np


def select_flight_forest(whole, supported, coverage):
    """可比工况越少，越侧重可比部分；完全没有时两臂已回退到同一输入。"""
    full, comparable, weight = (np.asarray(x, dtype=float) for x in (whole, supported, coverage))
    if (full.shape != comparable.shape or weight.shape != full.shape or not np.isfinite(full).all()
            or not np.isfinite(comparable).all() or not np.all((0 <= weight) & (weight <= 1))):
        raise ValueError("飞行预测与覆盖比例不匹配")
    return weight * full + (1 - weight) * comparable
