"""将已实测的设备均衡树与局部残差方案提取为可复用组件。

来源是本项目run365实际提名的ParentAwareLocalResidual（执行22），不是论文
预先规定的模型。默认设置保持原计算；分支开关供消融，不使用评价反馈拟合。
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors


class BalancedTreeResidualRegressor(RegressorMixin, BaseEstimator):
    """以最后一列父预测为基准；设备身份只通过合法训练侧groups传入。"""

    def __init__(self, max_iter=180, learning_rate=.04, max_leaf_nodes=15,
                 l2_regularization=2., local_scale=.75, global_weight=1.,
                 anchors=64, neighbors=16, balance_devices=True, parent_distance=True):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.local_scale = local_scale
        self.global_weight = global_weight
        self.anchors = anchors
        self.neighbors = neighbors
        self.balance_devices = balance_devices
        self.parent_distance = parent_distance

    def fit(self, X, y, groups=None):
        self.feature_columns_ = list(X.columns[:-1])
        self.parent_column_ = X.columns[-1]
        features = X[self.feature_columns_].to_numpy(dtype=np.float32)
        parent = X[self.parent_column_].to_numpy(dtype=np.float64)
        residual = np.asarray(y, dtype=np.float64)-parent
        groups = np.zeros(len(X), dtype=int) if groups is None else np.asarray(groups)
        if len(groups) != len(X) or len(y) != len(X):
            raise ValueError('训练设备、标签和输入行数不同')
        self.lo_, self.hi_ = np.quantile(residual, [.01, .99])
        if self.global_weight:
            _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
            weights = 1./counts[inverse]
            weights *= len(weights)/weights.sum()
            self.global_model_ = HistGradientBoostingRegressor(
                max_iter=self.max_iter, learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization,
                early_stopping=False, random_state=17)
            self.global_model_.fit(features, residual,
                                   sample_weight=weights if self.balance_devices else None)
        self.anchors_ = 0
        if self.local_scale:
            if type(self.anchors) is not int or self.anchors < 1 or type(self.neighbors) is not int or self.neighbors < 1:
                raise ValueError('锚点数与邻居数必须为正整数')
            keep = []
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                keep.extend(rows[np.linspace(0, len(rows)-1, min(self.anchors, len(rows))).astype(int)])
            keep = np.asarray(keep, dtype=int)
            self.anchors_ = len(keep)
            self.local_columns_ = [c for c in self.feature_columns_ if c.startswith('native_')] or self.feature_columns_
            local = X[self.local_columns_].to_numpy(dtype=np.float64)
            self.local_median_ = np.median(local, axis=0)
            q25, q75 = np.percentile(local, [25, 75], axis=0)
            self.local_iqr_ = np.maximum(q75-q25, 1e-3)
            self.parent_mean_ = float(parent.mean())
            self.parent_std_ = float(parent.std())+1e-6
            self.local_bank_ = self._local_view(X, parent)[keep]
            self.local_residual_ = residual[keep]
            self.neighbors_ = NearestNeighbors(n_neighbors=min(self.neighbors, len(keep)), n_jobs=1).fit(self.local_bank_)
        return self

    def _local_view(self, X, parent):
        local = X[self.local_columns_].to_numpy(dtype=np.float64)
        scaled = (local-self.local_median_)/self.local_iqr_
        if self.parent_distance:
            scaled = np.column_stack([scaled, ((parent-self.parent_mean_)/self.parent_std_)[:, None]])
        return scaled

    def predict(self, X):
        parent = X[self.parent_column_].to_numpy(dtype=np.float64)
        if not self.global_weight and not self.local_scale:
            return np.maximum(0., parent)
        global_residual = np.zeros(len(X), dtype=np.float64)
        local_residual = np.zeros(len(X), dtype=np.float64)
        if self.global_weight:
            global_residual = self.global_model_.predict(X[self.feature_columns_].to_numpy(dtype=np.float32))
        if self.local_scale:
            distances, indices = self.neighbors_.kneighbors(self._local_view(X, parent))
            for i, (ds, ix) in enumerate(zip(distances, indices, strict=True)):
                positive = ds[ds > 0]
                sigma = max(float(np.median(positive)) if len(positive) else 1., 1e-6)
                weights = np.exp(-.5*(ds/sigma)**2)
                weights /= weights.sum()
                local_residual[i] = float(weights @ self.local_residual_[ix])
        correction = np.clip(self.global_weight*global_residual+self.local_scale*local_residual,
                             self.lo_, self.hi_)
        return np.maximum(0., parent+correction)
