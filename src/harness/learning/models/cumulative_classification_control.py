"""在完整批次标签到达后选择冻结分类工具，分别比较逐批和完整累计评分。"""
import numpy as np

from harness.learning.models.agile_selector import AgileSelector

MODES = ('fixed', 'batch_mean', 'pooled_history', 'recent_complete', 'agile_batch', 'agile_complete_gain')


def confusion(truth, prediction, classes=2):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 1 or not np.isin(truth, np.arange(1, classes+1)).all() or not np.isin(prediction, np.arange(1, classes+1)).all():
        raise ValueError('标签和预测须为相同长度的合法类别序列')
    return np.bincount((truth.astype(int)-1)*classes+prediction.astype(int)-1,
                       minlength=classes**2).reshape(classes, classes).astype(float)


def macro_f1(counts):
    """固定类别集合；支持最后两维为混淆矩阵的批量计数，零分母按0计。"""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim < 2 or counts.shape[-1] != counts.shape[-2] or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError('混淆矩阵须为有限非负方阵')
    numerator = 2*np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-1)+counts.sum(axis=-2)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return values.mean(axis=-1)


class CumulativeClassificationControl:
    """先决定整批模型并交付，后接收该批完整标签；源前缀交付永久固定。

    不拟合预测模型。Agile模式复用原权重更新，使用最大权重的单模型实际
    交付；这不是论文随机/分布动作的理论保证。完整增量损失只使用已回标
    的真实交付状态，不把某个候选的假想历史冒充实际历史。
    """
    def __init__(self, mode, arms, initial, prefix_confusion, horizon):
        if mode not in MODES or arms < 1 or not 0 <= initial < arms or horizon < 1:
            raise ValueError('选择模式、工具数量或初始工具无效')
        self.mode, self.arms, self.initial, self.horizon = mode, arms, initial, horizon
        self.prefix = np.asarray(prefix_confusion, dtype=float).copy()
        if self.prefix.ndim != 2:
            raise ValueError('实际前缀须是一份累计混淆矩阵')
        macro_f1(self.prefix)
        self.actual = self.prefix.copy()
        self.history = np.zeros((arms, *self.prefix.shape))
        self.batch_loss = np.zeros(arms)
        self.previous = None
        self.selector = AgileSelector(arms, max(2, horizon)) if mode.startswith('agile_') else None
        self.updates, self.pending = 0, None
        self.actual_gain_loss_sum = 0.

    def _best(self, scores):
        candidates = np.flatnonzero(np.asarray(scores) >= np.max(scores)-1e-12)
        return self.initial if self.initial in candidates else int(candidates[0])

    def choose(self, rows):
        if self.pending is not None or rows < 1 or self.updates >= self.horizon:
            raise ValueError('必须先接收上一批反馈，且不能超过冻结的批次数')
        weights = None
        if self.mode == 'fixed' or self.updates == 0:
            chosen = self.initial
        elif self.mode == 'batch_mean':
            chosen = self._best(-self.batch_loss/self.updates)
        elif self.mode == 'pooled_history':
            chosen = self._best(macro_f1(self.prefix+self.history))
        elif self.mode == 'recent_complete':
            # 用刚结束的一批预测下一批的混淆增量；仅缩放已知批次大小，不读取新标签。
            scale = rows/self.previous[0].sum()
            chosen = self._best(macro_f1(self.actual+self.previous*scale))
        else:
            weights = self.selector.prediction()[0]
            chosen = self._best(weights)
        self.pending = (int(chosen), int(rows))
        return chosen, None if weights is None else weights.tolist()

    def observe(self, truth, predictions):
        if self.pending is None:
            raise ValueError('标签到达前须先固定并交付本批预测')
        chosen, rows = self.pending
        truth, predictions = np.asarray(truth), np.asarray(predictions)
        if truth.shape != (rows,) or predictions.shape != (rows, self.arms):
            raise ValueError('只接收刚交付整批的完整候选预测和标签')
        matrices = np.stack([confusion(truth, predictions[:, i], len(self.prefix)) for i in range(self.arms)])
        before = float(macro_f1(self.actual))
        batch_loss = 1-macro_f1(matrices)
        gain_loss = (1-(macro_f1(self.actual+matrices)-before))/2
        if ((gain_loss < -1e-12) | (gain_loss > 1+1e-12)).any():
            raise ValueError('完整增量损失越过[0,1]')
        gain_loss = np.clip(gain_loss, 0., 1.)
        if self.selector:
            self.selector.observe(batch_loss if self.mode == 'agile_batch' else gain_loss)
        self.actual_gain_loss_sum += float(gain_loss[chosen])
        self.actual += matrices[chosen]
        self.history += matrices
        self.batch_loss += batch_loss
        self.previous = matrices
        self.updates += 1
        self.pending = None
        # 对已经实际发布的路径逐批核对望远镜恒等式，避免用候选历史替代实际历史。
        expected = self.updates/2-(float(macro_f1(self.actual))-float(macro_f1(self.prefix)))/2
        if abs(expected-self.actual_gain_loss_sum) > 1e-10:
            raise ValueError('累计更新没有与真实完整交付评分对齐')
        return {'before_macro_f1': before, 'after_macro_f1': float(macro_f1(self.actual)),
                'batch_loss': batch_loss.tolist(), 'gain_loss': gain_loss.tolist()}

    def state(self):
        if self.pending is not None:
            raise ValueError('仅在整批反馈结束后保存状态')
        return {'mode': self.mode, 'arms': self.arms, 'initial': self.initial, 'horizon': self.horizon,
                'prefix': self.prefix.tolist(), 'actual': self.actual.tolist(), 'history': self.history.tolist(),
                'batch_loss': self.batch_loss.tolist(), 'previous': None if self.previous is None else self.previous.tolist(),
                'updates': self.updates, 'actual_gain_loss_sum': self.actual_gain_loss_sum,
                'selector': None if self.selector is None else self.selector.state()}

    @classmethod
    def restore(cls, state):
        result = cls(state['mode'], state['arms'], state['initial'], state['prefix'], state['horizon'])
        for key in ['actual', 'history', 'batch_loss']:
            setattr(result, key, np.asarray(state[key], dtype=float))
        result.previous = None if state['previous'] is None else np.asarray(state['previous'], dtype=float)
        result.updates = state['updates']
        result.actual_gain_loss_sum = state['actual_gain_loss_sum']
        result.selector = None if state['selector'] is None else AgileSelector.restore(state['selector'])
        return result
