"""由当轮公开训练寿命估计上限，不使用验证、查询或设备身份。"""
import numpy as np


def source_limit(y):
    values = np.asarray(y, dtype=np.float64)
    if (values.ndim != 1 or values.size < 10 or not np.isfinite(values).all()
            or values.min() < 0 or values.min() == values.max()):
        return None
    # 有限样本的上侧次序统计；0.9 为开发对照冻结的比例。
    index = min(values.size-1, int(np.ceil(.9*(values.size+1)))-1)
    cap = float(np.partition(values, index)[index])
    return cap if cap > 0 else None
