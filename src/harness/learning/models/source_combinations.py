"""以源数据的低取值列识别唯一的单字段不一致，无候选或并列时保留输入。"""
from collections import Counter, defaultdict

import numpy as np
from harness.core.contracts import digest


class SourceCombinations:
    def __init__(self, training_values):
        x = np.asarray(training_values, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("训练输入必须是非空有限矩阵")
        self.columns = [i for i, v in enumerate(x.T) if 2 <= len(np.unique(v)) <= 32]
        self.width = x.shape[1]
        self.counts = Counter(tuple(v) for v in x[:, self.columns])
        self._index()

    def _index(self):
        self.values = {col: {v[j] for v in self.counts} for j, col in enumerate(self.columns)}
        self.neighbors = defaultdict(list)
        if len(self.columns) >= 3:
            for entry in self.counts:
                for j in range(len(entry)):
                    self.neighbors[j, entry[:j] + entry[j+1:]].append(entry)
        self.identity = digest([self.columns, sorted([list(k), n] for k, n in self.counts.items())])

    def state(self):
        return {"width": self.width, "columns": self.columns, "counts": sorted([list(k), n] for k, n in self.counts.items())}

    @classmethod
    def restore(cls, record):
        """用训练结束时已知的计数恢复索引；服务端无需训练样本或任何寿命标签。"""
        obj = cls.__new__(cls)
        obj.width, obj.columns = record["width"], record["columns"]
        if (type(obj.width) != int or obj.width < 1 or any(type(i) != int or not 0 <= i < obj.width for i in obj.columns)
                or obj.columns != sorted(set(obj.columns))):
            raise ValueError("源组合的列号不合法")
        obj.counts = Counter()
        for key, n in record["counts"]:
            if len(key) != len(obj.columns) or not np.isfinite(key).all() or type(n) != int or n < 1 or tuple(key) in obj.counts:
                raise ValueError("源组合的取值或计数不合法")
            obj.counts[tuple(key)] = n
        if not obj.counts:
            raise ValueError("源训练组合缺失")
        obj._index()
        return obj

    def transform(self, values, unseen_only=False):
        """索引中的低频项也能产生并列；只在唯一邻居的训练支持足够时采用。"""
        x = np.asarray(values, dtype=float).copy()
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError("新观测与源训练列数不符或含非有限值")
        if not self.neighbors:
            return x
        for row in x:
            key = tuple(row[self.columns])
            if key in self.counts:
                continue
            matches = []
            for j, column in enumerate(self.columns):
                for neighbor in self.neighbors.get((j, key[:j] + key[j+1:]), ()):
                    matches.append((column, neighbor[j], self.counts[neighbor]))
            if len(matches) == 1:
                column, replacement, count = matches[0]
                if count >= 8 and (not unseen_only or row[column] not in self.values[column]):
                    row[column] = replacement
        return x

    def advance_current(self, model, explore, confirm, serve):
        """预测使用本条传感器和冻结的组合表，反馈由普通维护流程单独持有。"""
        return serve_transformed(model, serve, self.transform, "source-combinations")


def serve_transformed(model, serve, transform, policy):
    ids = [r["sample_id"] for r in serve]
    if not ids or len(ids) != len(set(ids)) or any(r["label"] is not None for r in serve):
        raise ValueError("需要唯一、无标签的服务行")
    x = np.asarray([r["values"] for r in serve], dtype=float)
    mapped = transform(x)
    changed = int(np.count_nonzero(np.any(mapped != x, axis=1)))
    base = model.predict(x)
    actual = model.predict(mapped) if changed else base
    return (dict(zip(ids, actual.tolist(), strict=True)), dict(zip(ids, base.tolist(), strict=True)),
            {"chosen": policy, "changed_rows": changed})
