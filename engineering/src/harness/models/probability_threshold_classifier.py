"""对已拟合的二分类器应用固定概率阈值，不重新训练或调整输入。"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityThresholdClassifier(ClassifierMixin, BaseEstimator):
    """基模型需提供类别1/2的predict_proba；阈值由调用方合法反馈确定。

    包装器不持有评分标签。概率恰等于阈值时输出类别1，与两类argmax的
    平手行为一致；预测只依赖当前行与已固定的模型、阈值。
    """
    def __init__(self, base_model, threshold=.5):
        self.base_model = base_model
        self.threshold = threshold

    @property
    def classes_(self):
        return np.asarray(self.base_model.classes_)

    def predict_proba(self, X):
        if not np.array_equal(self.classes_, [1, 2]) or not 0 <= self.threshold <= 1:
            raise ValueError('固定阈值接口要求类别1/2和合法概率阈值')
        values = np.asarray(self.base_model.predict_proba(X))
        if values.shape != (len(X), 2) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('基模型概率不合法')
        return values

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] > self.threshold, 2, 1)
