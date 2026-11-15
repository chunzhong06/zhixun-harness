"""将已交付、已回标和允许训练分开；本模块不读取尚未到达的答案。"""
from hashlib import sha256
import json
import numpy as np


def identity(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class FeedbackTrainingLedger:
    """整批交付后接收事先指定样本的同序标签；任务显式允许时导出训练行。

    此对象约束合作式执行器的接口时序，不构成操作系统隔离。调用者须先将
    预测持久化，再用其摘要登记交付；仅在完整反馈结束后保存恢复点。
    """
    def __init__(self, source_ids, *, allow_feedback_training=False):
        self.source_ids = list(source_ids)
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('源样本ID必须非空且唯一')
        self.allow_feedback_training = bool(allow_feedback_training)
        self.feedback = []
        self.pending = None

    def issue(self, ids, prediction_sha256, *, feedback_ids=None):
        ids = list(ids)
        expected = ids if feedback_ids is None else list(feedback_ids)
        used = set(self.source_ids).union(*(set(b.get('issued_ids', b['ids'])) for b in self.feedback))
        if self.pending is not None or not ids or len(set(ids)) != len(ids) or used.intersection(ids):
            raise ValueError('须先完成上一批反馈；源样本和已交付样本不能重复发放')
        if len(set(expected)) != len(expected) or set(expected) - set(ids):
            raise ValueError('预定回标必须是本批已交付样本的唯一子集')
        if len(prediction_sha256) != 64 or any(c not in '0123456789abcdef' for c in prediction_sha256):
            raise ValueError('须提供实际持久化预测的SHA256')
        self.pending = {'ids': expected, 'issued_ids': ids, 'prediction_sha256': prediction_sha256}

    def observe(self, ids, labels):
        labels = np.asarray(labels, dtype=float)
        if self.pending is None or list(ids) != self.pending['ids']:
            raise ValueError('只能回标刚完成交付的同一批、同序样本')
        if labels.shape != (len(ids),) or not np.isfinite(labels).all():
            raise ValueError('反馈必须是逐行对齐的有限数值标签')
        self.feedback.append({**self.pending, 'labels': labels.tolist()})
        self.pending = None

    def training_view(self, policy):
        """返回源ID、已知反馈标签及权重；近期策略只保留最后两批反馈。"""
        if self.pending is not None:
            raise ValueError('整批发放尚未回标，不能开始下一次训练')
        if policy not in ('source_only', 'cumulative_refit', 'recent_mix_refit'):
            raise ValueError('未知反馈训练策略')
        if policy != 'source_only' and not self.allow_feedback_training:
            raise PermissionError('当前任务只允许反馈评分，不允许反馈进入训练')
        batches = [] if policy == 'source_only' else self.feedback[-2:] if policy == 'recent_mix_refit' else self.feedback
        ids = [i for b in batches for i in b['ids']]
        labels = [y for b in batches for y in b['labels']]
        target_weight = len(self.source_ids) / len(ids) if ids and policy == 'recent_mix_refit' else 1.
        weights = [1.] * len(self.source_ids) + [target_weight] * len(ids)
        return {'source_ids': list(self.source_ids), 'feedback_ids': ids, 'feedback_labels': labels,
                'sample_weights': weights, 'observed_batches': len(self.feedback), 'policy': policy}

    def state(self):
        if self.pending is not None:
            raise ValueError('恢复点必须在整批回标完成之后')
        return {'source_ids': list(self.source_ids), 'allow_feedback_training': self.allow_feedback_training,
                'feedback': json.loads(json.dumps(self.feedback))}

    @classmethod
    def restore(cls, state):
        result = cls(state['source_ids'], allow_feedback_training=state['allow_feedback_training'])
        for batch in state['feedback']:
            result.issue(batch.get('issued_ids', batch['ids']), batch['prediction_sha256'], feedback_ids=batch['ids'])
            result.observe(batch['ids'], batch['labels'])
        return result


def refit_linear_parent(parent, source_x, source_y, target_x_by_id, view):
    """保留父版源端尺度，以相同线性学习器重新拟合；不是原优化器状态续训。"""
    from copy import deepcopy
    from sklearn.base import clone
    import pandas as pd
    ids = view['feedback_ids']
    if len(source_x) != len(view['source_ids']) or len(source_y) != len(source_x):
        raise ValueError('源训练资料与账本不一致')
    model = deepcopy(parent)
    if not ids:
        return model
    rows = pd.DataFrame([target_x_by_id[i] for i in ids], columns=source_x.columns)
    x = pd.concat((source_x, rows), ignore_index=True)
    y = np.concatenate((np.asarray(source_y), view['feedback_labels']))
    name, prototype = model.steps[-1]
    learner = clone(prototype)
    learner.fit(model[:-1].transform(x), y, sample_weight=view['sample_weights'])
    model.steps[-1] = (name, learner)
    return model


class FeedbackParentModel:
    """保存父后端所需输入列，使附带父预测的共同任务输入能直接独立推理。"""
    def __init__(self, model, columns):
        self.model, self.columns = model, list(columns)

    def predict(self, frame):
        return np.maximum(0., self.model.predict(frame[self.columns]))

    @property
    def classes_(self):
        """概率列按父分类器的真实标签排序；不把列索引当类别编号。"""
        return np.asarray(self.model.classes_).copy()

    def predict_proba(self, frame):
        if not hasattr(self.model, 'predict_proba'):
            raise TypeError('该父后端不提供分类概率，请使用predict')
        values = np.asarray(self.model.predict_proba(frame[self.columns]), dtype=float)
        if (values.shape != (len(frame), len(self.classes_)) or not np.isfinite(values).all()
                or (values < 0).any() or not np.allclose(values.sum(axis=1), 1., atol=1e-8, rtol=0)):
            raise ValueError('父分类器概率形状、数值或类别顺序不合法')
        return values


def fit_feedback_parent(parent, features, targets, weights, columns, feedback_rows):
    """两种已实测后端的共同重训入口：保留流水线源尺度，或重拟合单一估计器。

    无反馈时保持父模型。流水线只重拟合末端学习器，单一估计器沿原参数重拟合。
    这是新拟合，不声明继承优化器或原提升树；不改变官方NPZ续训接口。
    """
    from copy import deepcopy
    from sklearn.base import clone
    model = deepcopy(parent)
    if feedback_rows:
        x = features[columns]
        if hasattr(model, 'steps'):
            name, prototype = model.steps[-1]
            learner = clone(prototype)
            learner.fit(model[:-1].transform(x), targets, sample_weight=weights)
            model.steps[-1] = (name, learner)
        else:
            model = clone(parent).fit(x, targets, sample_weight=weights)
    return FeedbackParentModel(model, columns)
