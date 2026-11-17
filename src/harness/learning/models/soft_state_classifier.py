"""从SGP40完整CoMind运行的实际候选提取软门控分类器。

默认参数保留原训练顺序和预测规则；可选开关仅用于固定消融。来源为
broad-transfer-full-pair/sgp40/comind/executions/0010的candidate-003，
是该次现场生成的预测模型，并非CoMind论文原文的预测架构。
"""
import random

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class SoftStateExperts(nn.Module):
    def __init__(self, n_features, n_classes=2, n_experts=3, learned_gate=True):
        super().__init__()
        self.learned_gate = learned_gate
        # 均匀门控消融仍按相同顺序初始化门控，保住其余专家的随机起点。
        self.gate = nn.Sequential(nn.Linear(n_features, 16), nn.Tanh(), nn.Linear(16, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(n_features, 32), nn.Tanh(),
                          nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, n_classes))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        if not self.learned_gate:
            weights = torch.full_like(weights, 1.0/len(self.experts))
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed_logits = (weights.unsqueeze(2)*expert_logits).sum(dim=1)
        return mixed_logits, weights, expert_logits


class SoftStateClassifier(ClassifierMixin, BaseEstimator):
    """只在fit提供的源标签上训练；预测按已冻结的列名逐行独立。

    默认包含parent_value，移除模式要求输入中确有该命名列。predict_proba
    只供诊断；实际类别保持原模型的logit argmax，不另调服务阈值。
    """
    def __init__(self, seed=29, n_experts=3, learned_gate=True, include_parent=True,
                 weight_decay=5e-4, entropy_weight=.002, consistency_weight=.01,
                 max_epochs=100, patience=10):
        self.seed = seed
        self.n_experts = n_experts
        self.learned_gate = learned_gate
        self.include_parent = include_parent
        self.weight_decay = weight_decay
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight
        self.max_epochs = max_epochs
        self.patience = patience

    def fit(self, X, y):
        torch.set_num_threads(1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if not self.include_parent and 'parent_value' not in X.columns:
            raise ValueError('移除父预测的消融要求命名列parent_value')
        self.columns_ = [c for c in X.columns if self.include_parent or c != 'parent_value']
        values = X[self.columns_].to_numpy(dtype=np.float32)
        self.fill_ = np.nanmean(values, axis=0)
        self.fill_ = np.where(np.isfinite(self.fill_), self.fill_, 0.0)
        values = np.where(np.isfinite(values), values, self.fill_)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        features = ((values-self.mean_)/self.scale_).astype(np.float32)
        self.classes_, targets = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('分类器至少需要两个源训练类别')
        counts = np.bincount(targets, minlength=len(self.classes_)).astype(np.float32)
        class_weights = torch.tensor(len(targets)/(len(self.classes_)*np.maximum(counts, 1.0)), dtype=torch.float32)
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.long)
        self.net_ = SoftStateExperts(features.shape[1], len(self.classes_), self.n_experts, self.learned_gate)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=.001, weight_decay=self.weight_decay)
        best_loss, best_state, stale_epochs = float('inf'), None, 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            for indices in torch.randperm(len(targets)).split(64):
                optimizer.zero_grad()
                logits, gate_weights, expert_logits = self.net_(x_tensor[indices])
                entropy = -(gate_weights*torch.log(gate_weights+1e-8)).sum(dim=1).mean()
                loss = nn.functional.cross_entropy(logits, y_tensor[indices], weight=class_weights)
                loss = loss-self.entropy_weight*entropy+self.consistency_weight*((expert_logits-logits[:, None, :])**2).mean()
                loss.backward()
                optimizer.step()
            self.net_.eval()
            with torch.no_grad():
                epoch_loss = nn.functional.cross_entropy(self.net_(x_tensor)[0], y_tensor, weight=class_weights).item()
            if epoch_loss < best_loss-1e-5:
                best_loss = epoch_loss
                best_state = {key: value.detach().clone() for key, value in self.net_.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break
        if best_state is None:
            raise ValueError('未产生有限训练状态')
        self.net_.load_state_dict(best_state)
        self.epochs_ = epoch+1
        self.training_loss_ = best_loss
        return self

    def _logits(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        values = X[self.columns_].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.fill_)
        features = np.nan_to_num((values-self.mean_)/self.scale_).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.tensor(features, dtype=torch.float32))[0]

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(dim=1).numpy()]

    def predict_proba(self, X):
        return torch.softmax(self._logits(X), dim=1).numpy()
