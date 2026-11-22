"""以合法发放的本底电阻辨认新设备支持范围，允许最近的分类上下文缺少某些类别。"""
from hashlib import sha256
import numpy as np


def background(x):
    fields = np.asarray(x, dtype=np.float64).reshape(-1, 16, 8)
    denominator = fields[:, :, 1] - 1.
    resistance = np.divide(fields[:, :, 0], denominator, out=np.full_like(denominator, np.nan),
                           where=np.abs(denominator) > 1e-5)
    return np.log(np.where(resistance > 0, resistance, np.nan))


def supported(query, recent, prior, *, leave_self_out=False):
    """每行分别比较到两个上下文的最近距离；训练时可排除被查询的源行自身。"""
    available = np.isfinite(prior).all(axis=1)
    prior = prior[available]
    available = np.isfinite(recent).all(axis=1)
    selected = np.zeros(len(query), dtype=bool)
    if not len(prior) or not available.any():
        return selected
    for i, row in enumerate(query):
        if not np.isfinite(row).all():
            continue
        distance = np.full(len(recent), np.inf)
        distance[available] = np.median(abs(recent[available] - row), axis=1)
        if leave_self_out:
            distance[i] = np.inf
        selected[i] = distance.min() < np.median(abs(prior-row), axis=1).min()
    return selected


def prepare(x, y, prior, classes, previous=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.shape != (len(y), 128) or y.ndim != 1 or not np.isfinite(x).all():
        return None
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() < 2 or not np.isin(labels, np.arange(len(classes))).all():
        return None
    capacity = len(prior)
    if capacity < 2*len(labels):
        return None
    if len(y) > capacity:
        rng = np.random.default_rng(42)
        keep = np.concatenate([rng.choice(np.flatnonzero(y==label), 2, replace=False) for label in labels])
        remaining = np.setdiff1d(np.arange(len(y)), keep, assume_unique=True)
        chosen = np.sort(np.r_[keep, rng.choice(remaining, capacity-len(keep), replace=False)])
        x, y = x[chosen], y[chosen]
    y = y.astype(np.int64)
    spec = {"format":"gas-recent-device-tabpfn-v1", "rows":len(y), "classes":labels.astype(int).tolist(),
            "source_x_sha256":sha256(x.tobytes()).hexdigest(),"source_y_sha256":sha256(y.tobytes()).hexdigest()}
    if spec == previous:
        return None
    recent = background(x)
    # 同一源资料重发时，所有行都能在父上下文中找到距离为零的对应项。
    # 仅在多数新行与另一条新行更相近时提出新组件，评价和最终选择仍由 SDK 负责。
    if np.count_nonzero(supported(recent, recent, background(prior), leave_self_out=True)) <= len(y)/2:
        return None
    return spec, {"device_context_x":x.copy(), "device_context_y":y}


def validate(spec, arrays, classes):
    x, y = arrays.get("device_context_x"), arrays.get("device_context_y")
    if (not isinstance(spec,dict) or spec.get("format")!="gas-recent-device-tabpfn-v1" or x is None or y is None
            or x.dtype!=np.float64 or y.dtype!=np.int64 or x.shape!=(len(y),128) or y.ndim!=1
            or spec.get("rows")!=len(y) or len(y)>len(arrays["classifier_source_x"]) or not np.isfinite(x).all()
            or spec.get("classes")!=np.unique(y).tolist() or len(spec["classes"])<2
            or not np.isin(y,np.arange(len(classes))).all()
            or sha256(x.tobytes()).hexdigest()!=spec.get("source_x_sha256")
            or sha256(y.tobytes()).hexdigest()!=spec.get("source_y_sha256")):
        raise ValueError("新设备分类上下文与已登记来源不符")
