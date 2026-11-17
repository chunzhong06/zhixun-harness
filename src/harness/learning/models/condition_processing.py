"""按已知工况标准化，稀少/新工况使用同一fit资料的全局统计。

借鉴TACDA/EviAdapt作者的工况分组预处理，保留原工况输入。分组精度来自
TaskSpec，不能在探索或采用资料上寻找更好的分组；这是成熟处理能力的接入。
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np


class ConditionStandardScaler(TransformerMixin, BaseEstimator):
    def __init__(self, condition_indices=(), decimals=(), min_group_rows=8):
        self.condition_indices = condition_indices
        self.decimals = decimals
        self.min_group_rows = min_group_rows

    def _array(self, X):
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("工况处理需要有限二维数值")
        return array

    def _keys(self, array):
        return [tuple(round(float(row[i]), d) for i, d in zip(self.condition_indices, self.decimals)) for row in array]

    def fit(self, X, y=None):
        array = self._array(X)
        if (not self.condition_indices or len(self.condition_indices) != len(self.decimals)
                or len(set(self.condition_indices)) != len(self.condition_indices)
                or any(type(i) is not int or not 0 <= i < array.shape[1] for i in self.condition_indices)
                or any(type(d) is not int or not 0 <= d <= 6 for d in self.decimals)
                or type(self.min_group_rows) is not int or self.min_group_rows < 2):
            raise ValueError("工况字段、精度或最小支持数无效")
        self.n_features_in_ = array.shape[1]
        self.signal_indices_ = [i for i in range(self.n_features_in_) if i not in self.condition_indices]
        if not len(array) or not self.signal_indices_:
            raise ValueError("工况处理需要拟合资料及非工况特征")
        self.global_ = StandardScaler().fit(array[:, self.signal_indices_])
        groups = {}
        for i, key in enumerate(self._keys(array)):
            groups.setdefault(key, []).append(i)
        self.group_counts_ = {k: len(v) for k, v in groups.items()}
        self.groups_ = {k: StandardScaler().fit(array[np.ix_(indices, self.signal_indices_)])
                        for k, indices in groups.items() if len(indices) >= self.min_group_rows}
        return self

    def transform(self, X):
        array = self._array(X)
        if not hasattr(self, "global_") or array.shape[1] != self.n_features_in_:
            raise ValueError("工况处理尚未拟合或输入维度改变")
        result = array.copy()
        groups = {}
        for i, key in enumerate(self._keys(array)):
            groups.setdefault(key, []).append(i)
        for key, indices in groups.items():
            columns = np.ix_(indices, self.signal_indices_)
            result[columns] = self.groups_.get(key, self.global_).transform(array[columns])
        return result
