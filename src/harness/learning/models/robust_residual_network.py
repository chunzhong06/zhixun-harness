"""复用run373实际发布的Huber残差网络；默认设置对应执行20的候选3。"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin


def anonymous_source(group):
    # 兼容旧公开投影及没有显式组身份的训练会话。
    return group in ("source", "('source', 'source')")


def coverage_rows(rows, group, limit):
    if anonymous_source(group) or len(rows) <= limit:
        return rows
    return rows[np.linspace(0, len(rows)-1, limit).astype(int)]


def residual_anchors(groups, limit):
    if not len(groups):
        return np.empty(0, dtype=int)
    return np.concatenate([coverage_rows(np.flatnonzero(groups == group), group, limit)
                           for group in np.unique(groups)])


class RobustResidualNetwork(RegressorMixin, BaseEstimator):
    """从合法设备锚点学习父预测残差；预测不读取设备身份或重估训练统计量。"""

    def __init__(self, anchors=128, hidden_width=128, learning_rate=.002,
                 epochs=12, loss='huber', beta=3., include_parent=True):
        self.anchors = anchors
        self.hidden_width = hidden_width
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.beta = beta
        self.include_parent = include_parent

    def fit(self, X, y, groups=None):
        if self.loss not in ('huber', 'squared'):
            raise ValueError('仅支持已登记的Huber或平方损失')
        if (type(self.anchors) is not int or self.anchors < 1 or
                type(self.hidden_width) is not int or self.hidden_width < 2 or
                type(self.epochs) is not int or self.epochs < 1):
            raise ValueError('锚点数、网络宽度和训练轮数必须为有效整数')
        torch.set_num_threads(1)
        self.raw_columns_ = list(X.columns[:-1])
        self.parent_column_ = X.columns[-1]
        base = X[self.parent_column_].to_numpy(dtype=np.float32)
        groups = np.zeros(len(X), dtype=int) if groups is None else np.asarray(groups)
        if len(groups) != len(X) or len(y) != len(X):
            raise ValueError('训练设备、标签和输入行数不同')
        keep = residual_anchors(groups, self.anchors)
        self.anchors_ = len(keep)
        residual = np.asarray(y, dtype=np.float32)-base
        self.lo_, self.hi_ = np.quantile(residual[keep], [.01, .99])
        features = self._features(X, base)
        self.mean_ = features[keep].mean(axis=0)
        self.std_ = features[keep].std(axis=0)+1e-6
        xx = torch.tensor((features[keep]-self.mean_)/self.std_, dtype=torch.float32)
        yy = torch.tensor(residual[keep, None], dtype=torch.float32)
        torch.manual_seed(17)
        self.net_ = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], self.hidden_width), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_width, self.hidden_width//2), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_width//2, 1))
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate, weight_decay=.001)
        for _ in range(self.epochs):
            self.net_.train()
            for indices in torch.randperm(len(xx)).split(1024):
                prediction = self.net_(xx[indices])
                loss = (torch.nn.functional.smooth_l1_loss(prediction, yy[indices], beta=self.beta)
                        if self.loss == 'huber' else ((prediction-yy[indices])**2).mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.optimizer_state_ = optimizer.state_dict()
        self.random_state_ = torch.get_rng_state().clone()
        return self

    def _features(self, X, base):
        raw = X[self.raw_columns_].to_numpy(dtype=np.float32)
        return np.column_stack([raw, base]) if self.include_parent else raw

    def predict(self, X):
        base = X[self.parent_column_].to_numpy(dtype=np.float32)
        features = self._features(X, base)
        self.net_.eval()
        with torch.no_grad():
            residual = self.net_(torch.tensor((features-self.mean_)/self.std_, dtype=torch.float32)).numpy().ravel()
        return np.maximum(0., base+np.clip(residual, self.lo_, self.hi_))
