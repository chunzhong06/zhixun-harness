"""在发布之后的真实反馈上组合模型，训练内回放不进入组合权重。"""
from collections import deque

import numpy as np


class PublicationPortfolio:
    """先发预测、后收同刻参考；保存状态可在交付边界恢复。

    候选列依次代表调用方固定的模型身份。最初沿用已采用版，收到八个
    样本外参考后按最近28日均方误差的倒数组合；不把此规则视为统计保证。
    """

    def __init__(self, count, initial, minimum=8, hours=28 * 24):
        if count < 1 or not 0 <= initial < count or minimum < 1 or hours <= 0:
            raise ValueError('发布组合配置无效')
        self.count, self.initial, self.minimum, self.hours = count, initial, minimum, hours
        self.history = deque()
        self.last_clock = None
        self.pending = None

    def predict(self, clock, values):
        values = np.asarray(values, dtype=float)
        if self.pending is not None or (self.last_clock is not None and clock <= self.last_clock):
            raise ValueError('上一条交付尚未接受反馈，或时钟没有前进')
        if values.shape != (self.count,) or not np.isfinite(values).all():
            raise ValueError('模型身份数或预测值无效')
        while self.history and self.history[0][0] <= clock - self.hours:
            self.history.popleft()
        weights = np.zeros(self.count)
        weights[self.initial] = 1.
        if len(self.history) >= self.minimum:
            mse = np.mean([loss for _, loss in self.history], axis=0)
            floor = max(1e-12, 1e-6 * float(np.mean(mse)))
            inverse = 1. / np.maximum(mse, floor)
            weights = inverse / inverse.sum()
        self.pending = (int(clock), values.copy())
        return float(weights @ values), weights

    def observe(self, clock, reference):
        if self.pending is None or clock != self.pending[0]:
            raise ValueError('反馈必须属于刚完成的交付')
        if np.isfinite(reference):
            self.history.append((int(clock), (self.pending[1] - float(reference)) ** 2))
        self.last_clock = int(clock)
        self.pending = None

    def state(self):
        if self.pending is not None:
            raise ValueError('仅在反馈处理完成的交付边界保存状态')
        return {'count': self.count, 'initial': self.initial, 'minimum': self.minimum,
                'hours': self.hours, 'last_clock': self.last_clock,
                'history': [(clock, loss.tolist()) for clock, loss in self.history]}

    @classmethod
    def restore(cls, state):
        model = cls(*(state[k] for k in ('count', 'initial', 'minimum', 'hours')))
        model.last_clock = state['last_clock']
        model.history = deque((clock, np.asarray(loss, dtype=float)) for clock, loss in state['history'])
        return model


def stream(predictions, references, clock, initial, state=None):
    predictions = np.asarray(predictions, dtype=float)
    references, clock = np.asarray(references), np.asarray(clock)
    if predictions.ndim != 2 or len(predictions) != len(clock) or references.shape != clock.shape:
        raise ValueError('发布流输入形状不同')
    model = PublicationPortfolio(predictions.shape[1], initial) if state is None else PublicationPortfolio.restore(state)
    if model.count != predictions.shape[1] or model.initial != initial:
        raise ValueError('恢复后的候选身份或初始版不符')
    issued, weights = [], []
    for now, values, reference in zip(clock, predictions, references, strict=True):
        prediction, weight = model.predict(now, values)
        issued.append(prediction)
        weights.append(weight)
        model.observe(now, reference)
    return np.asarray(issued), np.asarray(weights), model.state()
