"""把同一续训接口用于其他表格任务；原强父预测仍可作为输入及保持选项。"""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
from sklearn.base import BaseEstimator

from harness.learning.models.reference_numpy_mlp import Model
from harness.learning.training.native_continuation import ContinuationSession, run_program, tensor_identity
from harness.learning.training.continuation_recipes import reference_programs


class ContinuationEstimator(BaseEstimator):
    """建立源端初始学习器后可恢复参数、Adam和尺度继续训练。

    该候选在新任务上从合法源数据建立自己的初始模型；它与已部署父模型的
    关系是使用其预测特征，不声称继承外部父模型内部权重。后续continue_fit
    继承的是实际保存重载的本候选初始模型。
    """
    def __init__(self, classification=False, initial_epochs=15, hidden=32, seed=42):
        self.classification = classification
        self.initial_epochs = initial_epochs
        self.hidden = hidden
        self.seed = seed

    def _features(self, frame):
        values = frame[self.columns_].to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, self.fill_)

    def _targets(self, y):
        labels = np.asarray(y)
        if not self.classification: return labels.astype(float)
        mapping = {label: i for i, label in enumerate(self.classes_)}
        if any(label not in mapping for label in labels): raise ValueError('续训出现未登记类别')
        return np.asarray([mapping[label] for label in labels], dtype=int)

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        raw = X.to_numpy(dtype=float)
        self.fill_ = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        self.fill_ = np.where(np.isfinite(self.fill_), self.fill_, 0.)
        features = self._features(X)
        self.classes_ = np.unique(y) if self.classification else np.asarray([])
        target = self._targets(y)
        paths = [json.dumps([['key', str(name)]], separators=(',', ':')) for name in self.columns_]
        self.model_ = Model('classification' if self.classification else 'regression', paths,
                            [str(v) for v in self.classes_], features, target,
                            hidden=self.hidden, seed=self.seed, learning_rate=.003)
        self.model_.meta['profile'] = 'generic-tabular'
        self.model_.fit(features, target, self.initial_epochs)
        self.initial_tensors_ = tensor_identity(self.model_)
        self.continuation_history_ = []
        return self

    def continue_fit(self, X, y, groups=None, recipe='balanced-fine-tuning'):
        """不重新拟合填充值或尺度；程序与训练日志随实际模型一起保存。"""
        programs = reference_programs()
        if recipe not in programs: raise ValueError('没有登记该续训程序')
        before = tensor_identity(self.model_)
        session = ContinuationSession(self.model_, self._features(X), self._targets(y),
                                      groups=groups, max_epochs=12)
        report = run_program(programs[recipe], session)
        if report['loaded_tensors'] != before: raise ValueError('续训未继承实际初始状态')
        self.continuation_history_.append({'recipe': recipe,
            'source_sha256': sha256(programs[recipe].encode()).hexdigest(), 'audit': report})
        return self

    def predict(self, X):
        prediction = self.model_.predict(self._features(X))
        return self.classes_[prediction.argmax(axis=1)] if self.classification else prediction

    def predict_proba(self, X):
        if not self.classification: raise ValueError('回归候选不提供类别概率')
        return self.model_.predict(self._features(X))
