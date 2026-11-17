"""同一冻结教师实例的有界批量预取；不跨模型或跨输入身份复用。"""
import numpy as np


def prefetch_teacher(cache, raw, predict, *, width=None, nonnegative=False, batch_size=32):
    """按首次出现顺序计算缺失行；重复、重排和已有缓存不改变输出对应关系。"""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('教师预测输入须为有限二维数组')
    missing = {}
    for row in raw:
        key = row.tobytes()
        if key not in cache:
            missing.setdefault(key, row)
    keys = list(missing)
    for start in range(0, len(keys), batch_size):
        part = keys[start:start+batch_size]
        values = np.asarray(predict(np.asarray([missing[key] for key in part], dtype=np.float32)))
        shape = (len(part),) if width is None else (len(part), width)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('教师批量输出的形状或数值无效')
        if nonnegative:
            values = np.maximum(0., values)
        for key, value in zip(part, values, strict=True):
            cache[key] = value.copy() if width is not None else float(value)


def reuse_frozen_runtime(target, source):
    """保存重载后仅复用逐数组相同教师的内存状态，父网络仍使用重载后的权重。"""
    if type(target) is not type(source) or getattr(source, '_teacher', None) is None:
        return False
    left, right = target.auxiliary_arrays, source.auxiliary_arrays
    if left.keys() != right.keys() or any(not np.array_equal(left[key], right[key]) for key in left):
        return False
    target._teacher = source._teacher
    target._prediction_cache = dict(source._prediction_cache)
    return True
