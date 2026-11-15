"""迁移CoMind本次现场发现的残差模型，显式区分近邻与网络贡献。

来源：frontier-2026/engine/comind-recovered/executions/0014/candidate.py。
该预测器由作者框架现场生成，不是CoMind论文预先指定的固定算法。
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
import torch


class AnchoredResidualRegressor(RegressorMixin, BaseEstimator):
    """均衡抽取设备锚点，以合法训练残差修正最后一列父模型预测。

    native_*存在时近邻使用该视图，否则使用全部输入特征；网络使用全部
    非父预测列。groups由训练调用方提供，模型自身不读取设备或评价文件。
    """

    def __init__(self, anchors=64, local_weight=1., network_weight=.25, shrink_scale=20.):
        self.anchors = anchors
        self.local_weight = local_weight
        self.network_weight = network_weight
        self.shrink_scale = shrink_scale

    def fit(self, X, y, groups=None):
        torch.set_num_threads(1)
        self.raw_columns_ = list(X.columns[:-1])
        self.native_columns_ = [c for c in self.raw_columns_ if c.startswith('native_')] or self.raw_columns_
        self.parent_column_ = X.columns[-1]
        raw = X[self.raw_columns_].to_numpy(dtype=np.float32)
        native = X[self.native_columns_].to_numpy(dtype=np.float64)
        base = X[self.parent_column_].to_numpy(dtype=np.float32)
        groups = np.zeros(len(X), dtype=int) if groups is None else np.asarray(groups)
        if len(groups) != len(X) or len(y) != len(X):
            raise ValueError('训练设备、标签和输入行数不同')
        keep = []
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            keep.extend(rows[np.linspace(0, len(rows)-1, min(self.anchors, len(rows))).astype(int)])
        keep = np.asarray(keep, dtype=int)
        residual = np.asarray(y, dtype=np.float32) - base
        self.lo_, self.hi_ = np.quantile(residual[keep], [.01, .99])
        self.anchors_ = len(keep)
        if self.local_weight:
            self.med_ = np.median(native, axis=0)
            q25, q75 = np.percentile(native, [25, 75], axis=0)
            self.iqr_ = np.maximum(q75-q25, 1e-3)
            self.bmean_ = float(base.mean())
            self.bstd_ = float(base.std()) + 1e-6
            self.bank_ = np.column_stack([(native[keep]-self.med_)/self.iqr_,
                                          ((base[keep]-self.bmean_)/self.bstd_)[:, None]])
            self.bank_res_ = residual[keep]
            self.nn_ = NearestNeighbors(n_neighbors=min(16, len(keep)), n_jobs=1).fit(self.bank_)
        if self.network_weight:
            self.mean_ = raw[keep].mean(axis=0)
            self.std_ = raw[keep].std(axis=0) + 1e-6
            xx = torch.tensor((raw[keep]-self.mean_)/self.std_, dtype=torch.float32)
            yy = torch.tensor(residual[keep, None], dtype=torch.float32)
            torch.manual_seed(17)
            self.net_ = torch.nn.Sequential(torch.nn.Linear(raw.shape[1], 64), torch.nn.ReLU(),
                                             torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=.001, weight_decay=.001)
            for _ in range(12):
                self.net_.train()
                for indices in torch.randperm(len(xx)).split(1024):
                    loss = ((self.net_(xx[indices])-yy[indices])**2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        return self

    def predict(self, X):
        base = X[self.parent_column_].to_numpy(dtype=float)
        local = np.zeros(len(X))
        global_residual = np.zeros(len(X))
        if self.local_weight:
            native = X[self.native_columns_].to_numpy(dtype=np.float64)
            features = np.column_stack([(native-self.med_)/self.iqr_, ((base-self.bmean_)/self.bstd_)[:, None]])
            distances, indices = self.nn_.kneighbors(features)
            for i, (ds, ix) in enumerate(zip(distances, indices, strict=True)):
                positive = ds[ds > 0]
                sigma = max(float(np.median(positive)) if len(positive) else 1., 1e-6)
                weights = np.exp(-.5*(ds/sigma)**2)
                weights /= weights.sum()
                residuals = self.bank_res_[ix]
                mean_residual = float(weights @ residuals)
                effective_n = 1. / float(weights @ weights)
                dispersion = float(np.sqrt(weights @ ((residuals-mean_residual)**2)))
                shrink = (effective_n/(effective_n+8.)) / (1.+dispersion/self.shrink_scale)
                shrink *= min(1., effective_n/4.)
                local[i] = mean_residual * shrink
        if self.network_weight:
            self.net_.eval()
            raw = X[self.raw_columns_].to_numpy(dtype=np.float32)
            with torch.no_grad():
                global_residual = self.net_(torch.tensor((raw-self.mean_)/self.std_, dtype=torch.float32)).numpy().ravel()
        correction = np.clip(self.local_weight*local+self.network_weight*global_residual, self.lo_, self.hi_)
        return np.maximum(0., base+correction)
