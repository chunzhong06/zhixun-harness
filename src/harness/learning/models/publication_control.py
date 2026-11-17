"""发布后保留两套真实预测：依据到期反馈分配权重，或限制超出历史支持的改动。"""
import numpy as np

from harness.learning.models.agile_selector import AgileSelector
from harness.learning.models.publication_portfolio import PublicationPortfolio


MODES = ('current', 'parent', 'inverse_28d', 'agile', 'support_guard', 'agile_guard')


class SourceSupport:
    """用发布时已知输入和预测差校准；分位点是本次开发设置，不是风险保证。"""
    def __init__(self, features, candidate, parent):
        features = np.asarray(features, dtype=float)
        candidate, parent = np.asarray(candidate), np.asarray(parent)
        if features.ndim != 2 or len(features) < 12 or candidate.shape != parent.shape or candidate.shape != (len(features),):
            raise ValueError('支持域校准需要完整的合法历史输入与两套同刻预测')
        if not all(np.isfinite(x).all() for x in (features, candidate, parent)):
            raise ValueError('支持域校准资料非有限')
        self.lower, self.upper = np.quantile(features, [.01, .99], axis=0)
        self.radius = max(1e-8, float(np.quantile(np.abs(candidate-parent), .95)))

    def apply(self, features, parent, candidate, weight):
        features = np.asarray(features, dtype=float)
        if features.shape != self.lower.shape or not np.isfinite(features).all():
            raise ValueError('当前输入没有满足已校准的支持域列契约')
        outside = bool(np.any((features < self.lower) | (features > self.upper)))
        delta = candidate-parent
        correction = weight*delta
        limited = float(np.clip(correction, -self.radius, self.radius)) if outside else correction
        effective = limited/delta if delta != 0 else weight
        return float(parent+limited), float(effective), outside and abs(limited-correction) > 1e-12

    def state(self):
        return {'lower': self.lower.tolist(), 'upper': self.upper.tolist(), 'radius': self.radius}

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        result.lower, result.upper = np.asarray(state['lower']), np.asarray(state['upper'])
        result.radius = state['radius']
        return result


class PublicationControl:
    """每小时先交付、再观察参考；所有模式都保持父版与候选模型原反馈状态。"""
    def __init__(self, mode, support=None, horizon=1024):
        if mode not in MODES or mode.endswith('guard') and support is None:
            raise ValueError('发布控制模式或支持域校准缺失')
        self.mode, self.support, self.horizon = mode, support, horizon
        self.selector = AgileSelector(2, horizon) if mode in ('agile', 'agile_guard') else None
        self.inverse = PublicationPortfolio(2, 1) if mode == 'inverse_28d' else None
        self.last_clock = self.pending = None

    def predict(self, clock, values, features):
        values = np.asarray(values, dtype=float)
        if self.pending is not None or self.last_clock is not None and clock <= self.last_clock:
            raise ValueError('须完成上一小时反馈，且时钟严格递增')
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('需要父版和当前发布版两项有限预测')
        if self.inverse:
            prediction, weights = self.inverse.predict(clock, values)
        else:
            weights = self.selector.prediction()[0] if self.selector else np.array([1., 0.]) if self.mode == 'parent' else np.array([0., 1.])
            prediction = float(weights@values)
        limited = False
        if self.mode.endswith('guard'):
            prediction, weight, limited = self.support.apply(features, *values, weights[1])
            weights = np.array([1-weight, weight])
        self.pending = (int(clock), values.copy())
        return prediction, weights, limited

    def observe(self, clock, reference):
        if self.pending is None or clock != self.pending[0]:
            raise ValueError('参考必须属于刚完成交付的时刻')
        if self.inverse:
            self.inverse.observe(clock, reference)
        if self.selector and np.isfinite(reference):
            self.selector.observe((self.pending[1]-reference)**2)
        self.last_clock, self.pending = int(clock), None

    def state(self):
        if self.pending is not None:
            raise ValueError('仅在当小时反馈结束后保存恢复状态')
        return {'mode': self.mode, 'horizon': self.horizon, 'last_clock': self.last_clock,
                'support': self.support.state() if self.support else None,
                'selector': self.selector.state() if self.selector else None,
                'inverse': self.inverse.state() if self.inverse else None}

    @classmethod
    def restore(cls, state):
        model = cls(state['mode'], SourceSupport.restore(state['support']) if state['support'] else None, state['horizon'])
        model.last_clock = state['last_clock']
        if state['selector']:
            model.selector = AgileSelector.restore(state['selector'])
        if state['inverse']:
            model.inverse = PublicationPortfolio.restore(state['inverse'])
        return model


def stream(clock, predictions, references, features, *, mode, support=None, state=None):
    arrays = [np.asarray(a) for a in (clock, predictions, references, features)]
    clock, predictions, references, features = arrays
    if predictions.shape != (len(clock), 2) or references.shape != clock.shape or len(features) != len(clock):
        raise ValueError('发布流数组长度和模型列不一致')
    controller = PublicationControl(mode, support) if state is None else PublicationControl.restore(state)
    if controller.mode != mode:
        raise ValueError('恢复时不能更换模式')
    issued, weights, limited = [], [], []
    for now, values, reference, inputs in zip(clock, predictions, references, features, strict=True):
        value, weight, limit = controller.predict(now, values, inputs)
        issued.append(value);weights.append(weight);limited.append(limit)
        controller.observe(now, reference)
    return np.asarray(issued), np.asarray(weights), np.asarray(limited), controller.state()
