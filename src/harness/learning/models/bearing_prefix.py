"""从已出现的频带幅度判断振动变化，按已结束设备的交付误差学习切换时机。"""
import numpy as np


def prefix_condition(spectra):
    """返回当前相对起始段的幅度变化。逐行处理使截短轨迹与整台重放有相同的前缀。"""
    log_bands = np.log1p(np.asarray(spectra, dtype=float))
    if log_bands.ndim != 2 or not np.isfinite(log_bands).all():
        raise ValueError("需要有限的逐时频带幅度")
    condition = []
    for i in range(len(log_bands)):
        base = np.median(log_bands[:min(i+1, 8)],axis=0)
        recent = np.median(log_bands[max(0,i-4):i+1],axis=0)
        change = recent - base
        condition.append([np.median(change), np.quantile(change,.8)])
    return np.asarray(condition)


CONSTANTS = (0., .25, .5, .75, 1.)
THRESHOLDS = (.25, .5, 1., 1.5, 2.)


def fixed_prefix_predictions(condition, clock_prediction, vibration_prediction):
    """所有固定对照也使用相同的实时趋势与两个预测器。"""
    columns = {f"固定振动比{w:g}": (1-w)*clock_prediction+w*vibration_prediction for w in CONSTANTS}
    for j in (0,1):
        for cut in THRESHOLDS:
            columns[f"固定趋势{j}门槛{cut:g}"] = np.where(condition[:,j]>=cut,vibration_prediction,clock_prediction)
    return columns


class PrefixWeight:
    """在过去的完整设备上比较同条件的两种实发预测，学习有收缩的混合系数。"""
    def fit(self, histories):
        numer, denom, bins = [], [], []
        for condition, clock, vibration, truth in histories:
            delta = vibration - clock
            numer.append((truth-clock)*delta)
            denom.append(delta**2)
            bins.append(np.digitize(condition[:,1],THRESHOLDS))
        top, bottom, key = map(np.concatenate,(numer,denom,bins))
        total = bottom.sum()
        global_weight = np.clip(top.sum()/max(total,1.),0,1)
        # 用相当于既有资料二十分之一的全局证据，抑制少量强振动样本造成的极端权重。
        ridge = max(total/20,1.)
        self.weights = np.array([np.clip((top[key==i].sum()+ridge*global_weight)/
                                        (bottom[key==i].sum()+ridge),0,1) for i in range(len(THRESHOLDS)+1)])
        self.global_weight = float(global_weight)
        return self

    def predict(self,condition,clock,vibration):
        w = self.weights[np.digitize(condition[:,1],THRESHOLDS)]
        return clock+w*(vibration-clock)
