"""每次先发浓度，再接受到期的参考；按照上一次可见偏差修正后续小时。"""
from collections import deque

import numpy as np


# 包含原样保持、不同强度与不同持续时间；组合名跨站保持同一语义。
SETTINGS = ((0., 1.), *((gain, hours) for gain in (.25, .5, .75, 1.) for hours in (2., 6., 18.)))


def correct(base, reference, clock, setting):
    """同时校正一组已冻结的基础模型，时钟单位为小时；参考已被调用方按权限遮蔽。"""
    base = np.asarray(base, dtype=float)
    clock = np.asarray(clock)
    reference = np.asarray(reference, dtype=float)
    if (len(base) != len(reference) or reference.shape != clock.shape or base.ndim != 2 or
            np.any(np.diff(clock) <= 0) or not np.isfinite(base).all()):
        raise ValueError("参考、时钟与基础预测的小时身份不完整")
    gain,hours = setting
    if setting not in SETTINGS:
        raise ValueError("未知的参考反馈校正")
    prediction = base.copy()
    # 切点该小时的答案只能作用于严格晚于该时刻的输出；缺失不更新。
    arrivals = np.flatnonzero(np.isfinite(reference))
    for at, end in zip(arrivals, [*arrivals[1:], len(base)-1], strict=True):
        later = slice(at+1, end+1)
        attenuation = gain * np.exp(-(clock[later] - clock[at]) / hours)
        prediction[later] = np.maximum(0, base[later] + attenuation[:,None] * (reference[at] - base[at]))
    return prediction


def name(setting):
    gain,hours = setting
    return f"g{gain:g}-h{hours:g}"


def setting_of(raw):
    value=(float(raw["gain"]),float(raw["hours"]))
    if value not in SETTINGS:
        raise ValueError("参考校正未登记")
    return value


def window_scores(base, windows, y, setting):
    """各窗从本窗首个可见参考前的冷状态出发，先产生该时刻预测再接收其答案。"""
    losses=[]
    for mask in windows:
        ix=np.flatnonzero(mask)
        corrected=correct(base[ix],y[ix],ix,setting)
        losses.append(np.sqrt(np.mean((corrected-y[ix,None])**2,axis=0)))
    return np.asarray(losses)


def reference_losses(tools,model,windows,y,setting):
    rows=np.arange(len(y))
    return window_scores(tools.issue(model,rows)[:,None],windows,y,setting)[:,0]


def choose_from_feedback(predictions, reference, clock, service_start, initial, days=28):
    """候选先发放，再用到期答案更新最近误差；返回逐小时所选身份以便核对交接。"""
    predictions = np.asarray(predictions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    clock = np.asarray(clock)
    if (predictions.ndim != 2 or len(predictions) != len(reference)
            or reference.shape != clock.shape or np.any(np.diff(clock) <= 0)
            or not np.isfinite(predictions).all() or not 0 <= initial < predictions.shape[1]
            or days <= 0):
        raise ValueError("在线候选、参考或初始身份不一致")
    history = deque()
    total_loss = np.zeros(predictions.shape[1])
    choices = np.empty(len(clock), dtype=int)
    selected = initial
    for i, now in enumerate(clock):
        choices[i] = selected
        if np.isfinite(reference[i]):
            loss = (predictions[i] - reference[i]) ** 2
            history.append((now, loss))
            total_loss += loss
            while history and history[0][0] <= now - days * 24:
                total_loss -= history.popleft()[1]
            if now >= service_start:
                selected = int(np.argmin(total_loss))
    return predictions[np.arange(len(clock)), choices], choices
