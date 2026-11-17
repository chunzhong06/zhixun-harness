"""按BOA原文图2更新凸组合；每条预测先交付，随后才消费该条到期答案。"""
import numpy as np


class BernsteinAggregation:
    """Wintenberger (2017) fully adaptive BOA，本地平方损失适配。

    原件：references/papers/online-aggregation/boa-1404.1356v5。
    convex用式(2)梯度损失，selection用原文的中心化专家损失。
    状态只接收已经到期的反馈；无答案时调用方不得更新。
    """

    def __init__(self, count, mode="convex", c=20):
        if count < 2 or mode not in ("convex", "selection") or c <= 0:
            raise ValueError("组合至少两个候选，模式和最小量级必须有效")
        self.mode = mode
        self.c = c
        self.prior = np.full(count, 1. / count)
        self.weights = self.prior.copy()
        self.total = np.zeros(count)
        self.rate = np.zeros(count)
        self.squares = np.zeros(count)
        self.max_abs = np.zeros(count)
        self.updates = 0

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.weights.shape or not np.isfinite(values).all():
            raise ValueError("当前候选预测不完整")
        return float(self.weights @ values)

    def update(self, values, answer):
        """使用仍对应本条交付的旧权重更新；调用后权重只用于下一条。"""
        prediction = self.predict(values)
        if not np.isfinite(answer):
            raise ValueError("只可消费已到期的有限答案")
        values = np.asarray(values, dtype=float)
        if self.mode == "convex":
            excess = 2. * (prediction - answer) * (values - prediction)
        else:
            loss = (values - answer) ** 2
            excess = loss - self.weights @ loss
        self.max_abs = np.maximum(self.max_abs, np.abs(excess))
        ranges = np.exp2(1. + np.maximum(-self.c, np.ceil(np.log2(np.maximum(self.max_abs, 2. ** -self.c)))))
        # 图2的修正项用本次范围、上次学习率；触发条件是正超额而非绝对值。
        self.total += .5 * (excess * (1. + self.rate * excess) + ranges * (self.rate * excess > .5))
        self.squares += excess ** 2
        bound = np.full_like(self.squares, np.inf)
        np.divide(-np.log(self.prior), self.squares, out=bound, where=self.squares > 0)
        self.rate = np.minimum(1. / ranges, np.sqrt(bound))
        log_weight = np.log(self.prior) + np.log(self.rate) - self.rate * self.total
        if not np.isfinite(log_weight).all():
            raise FloatingPointError("BOA状态溢出，禁止悄悄更换交付")
        weight = np.exp(log_weight - np.max(log_weight))
        self.weights = weight / weight.sum()
        self.updates += 1


def aggregate(predictions, reference, mode="convex", stop_update_at=None):
    """返回逐条预测及实际使用权重；stop_update_at支持开场后冻结的配对对照。"""
    predictions = np.asarray(predictions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predictions.ndim != 2 or reference.shape != (len(predictions),):
        raise ValueError("预测和到期反馈形状不同")
    model = BernsteinAggregation(predictions.shape[1], mode)
    values = np.empty(len(predictions))
    weights = np.empty_like(predictions)
    for i, row in enumerate(predictions):
        weights[i] = model.weights
        values[i] = model.predict(row)
        if np.isfinite(reference[i]) and (stop_update_at is None or i < stop_update_at):
            model.update(row, reference[i])
    return values, weights
