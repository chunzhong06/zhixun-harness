"""按KDD 2026 Agile OMS的算法1、2实现固定模型池选择；此处为独立方法适配。

来源：https://arxiv.org/html/2605.26919v2 。使用有界时间上界、损失范围重启和
附录B的指数平均乐观预测。调用者只在真实标签到达后更新，不宣称覆盖缺测小时的遗憾保证。
"""
import numpy as np
from scipy.optimize import brentq


def mirror_step(prior, cost, rates, active, floor):
    """求带分量学习率和下界的熵镜像投影；一个公共乘子维持总质量为1。"""
    prior, cost = np.asarray(prior), np.asarray(cost)
    if prior.shape != cost.shape or not active.any() or floor*prior.size >= 1:
        raise ValueError('镜像投影形状、活动学习率或下界无效')
    rate = np.broadcast_to(rates, prior.shape)
    mask = np.broadcast_to(active, prior.shape)
    offset = float(np.min(cost[mask]))
    z = np.log(np.maximum(prior, floor))-rate*(cost-offset)
    log_floor = np.log(floor)

    def weights(multiplier):
        return np.where(mask, np.exp(np.maximum(log_floor, np.minimum(40., z+rate*multiplier))), floor)

    lower = float(np.min((log_floor-z[mask])/rate[mask])-1)
    upper = float(np.max(-z[mask]/rate[mask])+1)
    root = brentq(lambda x: weights(x).sum()-1., lower, upper, xtol=1e-13, rtol=1e-14)
    value = weights(root)
    if not np.isclose(value.sum(), 1., atol=1e-9, rtol=0) or not np.isfinite(value).all():
        raise ValueError('镜像投影没有守住概率质量')
    return value


class AgileSelector:
    """预测不消耗反馈轮次；observe仅接受该次预测之后到达的完整模型损失向量。"""
    def __init__(self, count, horizon=1024):
        if count < 1 or horizon < 2:
            raise ValueError('模型数或轮次上界无效')
        self.count, self.horizon = count, horizon
        self.rates = 2.**np.arange(1, int(np.ceil(np.log2(horizon**2)))+1)/(16*horizon)
        self.floor = 1./(count*len(self.rates)*horizon**3)
        self.scale = self.maximum_error = 1.
        self.mean = np.zeros(count)
        self.updates = self.restarts = 0
        self._reset_weights()

    def _reset_weights(self):
        self.prior = np.broadcast_to(self.rates**2, (self.count, len(self.rates))).copy()
        self.prior /= self.prior.sum()
        self.penalty = np.zeros(len(self.rates))

    def prediction(self):
        active = self.penalty <= 1.
        cost = np.broadcast_to((self.mean/self.scale)[:, None], self.prior.shape)
        expanded = mirror_step(self.prior, cost, self.rates, active, self.floor)
        p = expanded[:, active].sum(axis=1)
        return p/p.sum(), expanded, active

    def observe(self, losses):
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (self.count,) or not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError('每个模型都需要真实、有限且非负的到期损失')
        p, expanded, active = self.prediction()
        # 附录B的公共偏移只在观察损失后使用；对已经作出的预测分布无影响。
        optimism = self.mean+float(p@(losses-self.mean))
        error = losses-optimism
        maximum = max(self.maximum_error, float(np.max(np.abs(error))))
        clipped = optimism+(self.maximum_error/maximum)*error
        residual = (clipped-optimism)/self.scale
        correction = 32*self.rates[None, :]*residual[:, None]**2
        cost = clipped[:, None]/self.scale+correction
        self.prior = mirror_step(self.prior, cost, self.rates, active, self.floor)
        self.penalty += np.sum((32*self.rates[None, :]*np.abs(residual[:, None]) > 1)*expanded*residual[:, None], axis=0)
        self.mean = .5*self.mean+.5*losses
        self.maximum_error = maximum
        self.updates += 1
        if self.updates > self.horizon:
            raise ValueError('超过事前设置的反馈更新上界')
        if maximum > self.scale:
            self.scale = max(2*self.scale, maximum)
            self.restarts += 1
            self._reset_weights()

    def state(self):
        return {key: getattr(self, key).tolist() if isinstance(getattr(self, key), np.ndarray) else getattr(self, key)
                for key in ('count', 'horizon', 'scale', 'maximum_error', 'mean', 'updates', 'restarts', 'prior', 'penalty')}

    @classmethod
    def restore(cls, state):
        model = cls(state['count'], state['horizon'])
        for key in ('scale', 'maximum_error', 'updates', 'restarts'):
            setattr(model, key, state[key])
        for key in ('mean', 'prior', 'penalty'):
            setattr(model, key, np.asarray(state[key], dtype=float))
        return model
