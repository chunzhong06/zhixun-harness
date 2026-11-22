"""从当前已发放的带标签行估计发动机第三项工况在父模型中的取值。"""
import json

import numpy as np


def setting_columns(paths):
    fields = [json.dumps([["key", "settings"], ["index", i]], separators=(",", ":")) for i in range(3)]
    return [paths.index(field) for field in fields] if all(field in paths for field in fields) else None


def operating_keys(raw, columns):
    return np.column_stack([np.round(raw[:, col], digits) for col, digits in zip(columns, (0, 2, 0), strict=True)])


class SettingAlignment:
    def __init__(self, paths, entries):
        self.columns = setting_columns(paths)
        self.mapping = {}
        if self.columns is None or not isinstance(entries, list):
            raise ValueError("发动机工况字段或映射无效")
        for item in entries:
            key, value = tuple(item["from"]), item["to"]
            if (len(key) != 3 or key in self.mapping or not np.isfinite((*key, value)).all()
                    or type(item["rows"]) is not int or item["rows"] < 8 or value == key[-1]):
                raise ValueError("发动机工况映射状态无效")
            self.mapping[key] = float(value)

    def transform(self, raw):
        x = np.asarray(raw, dtype=np.float64)
        if not self.mapping:
            return x
        keys = operating_keys(x, self.columns)
        result = x.copy()
        for key, target in self.mapping.items():
            match = (keys == key).all(axis=1)
            result[match, self.columns[-1]] = target
        return result


class AlignedEngine:
    """只在模型入口转换外部原始行，内部残差特征及标准化坐标保留自身语义。"""
    def __init__(self, model):
        if model.meta.get("profile") != "C" or model.meta.get("task") != "regression":
            raise ValueError("工况输入映射需要发动机回归父版")
        self.unaligned_engine = model
        self.engine_alignment = SettingAlignment(model.meta["paths"], model.meta["engine_input_alignment"])

    def __getattr__(self, name):
        return getattr(self.unaligned_engine, name)

    def predict(self, raw):
        return self.unaligned_engine.predict(self.engine_alignment.transform(raw))

    def predict_batch(self, raw):
        model = self.unaligned_engine
        return getattr(model, "predict_batch", model.predict)(self.engine_alignment.transform(raw))


def fit(parent, x, y, previous=()):
    """比较同一观测只改第三项后的父预测。缺少样本或收益小于配对误差波动时沿用父版。"""
    columns = setting_columns(parent.meta["paths"])
    if columns is None or len(x) != len(y):
        return None
    keys = operating_keys(x, columns)
    observed, counts = np.unique(keys[:, -1], return_counts=True)
    options = [float(v) for v, n in zip(observed, counts, strict=True) if n >= 8]
    # 仅处理离散工况，避免对近似连续字段逐值搜索。
    if len(options) < 2 or len(options) > 32:
        return None
    entries = {tuple(e["from"]): e for e in previous}
    changed = False
    modes, assignments, frequencies = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(frequencies >= 8):
        mode = modes[group]
        key = tuple(map(float, mode))
        indices = np.flatnonzero(assignments == group)
        # 父版在每个候选编码下都要推理。大批次等距取样，小批次保留所有合法行。
        if len(indices) > 256:
            indices = indices[np.linspace(0, len(indices)-1, 256, dtype=int)]
        rows, target = x[indices], y[indices]
        current = entries[key]["to"] if key in entries else key[-1]
        losses = {}
        for alternate in sorted({*options, current, key[-1]}):
            shifted = rows.copy()
            if alternate != key[-1]:
                shifted[:, columns[-1]] = alternate
            losses[alternate] = np.square(parent.predict(shifted) - target)
        best = min(losses, key=lambda option: float(losses[option].mean()))
        benefit = losses[current] - losses[best]
        if best == current or benefit.mean() <= benefit.std(ddof=1)/np.sqrt(len(benefit)):
            continue
        changed = True
        if best == key[-1]:
            del entries[key]
        else:
            entries[key] = {"from":list(key), "to":best, "rows":len(rows)}
    return [entries[key] for key in sorted(entries)] if changed else None
