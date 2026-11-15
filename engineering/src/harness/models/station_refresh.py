"""按照发放时钟补训已提出的配方，旧预测保留原权重身份。"""
from dataclasses import replace

import numpy as np


def prediction_bank(tools, rules, fit, known, clock, service_start, refresh_days, capture=()):
    """refresh_days=-1保持，0只在开场补训，正数按对应天数补训。"""
    boundaries = [0]
    if refresh_days >= 0:
        boundaries.append(int(np.searchsorted(clock, service_start)))
    if refresh_days > 0:
        boundaries.extend(range(boundaries[-1] + refresh_days * 24, len(clock), refresh_days * 24))
    boundaries = sorted(set(boundaries)) + [len(clock)]
    base = np.empty((len(clock), len(rules)))
    reset = np.zeros_like(base, dtype=bool)
    models = {}
    journal = []
    snapshots = []
    parent = tools.train(tools.static, fit, known)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        beginning = int(clock[start])
        training = fit if start == 0 else np.isfinite(known) & (np.arange(len(known)) < beginning)
        trained = {}
        failed = []
        for col, rule in enumerate(rules):
            identity = replace(rule, strength=1.)
            if identity not in trained:
                try:
                    trained[identity] = tools.train(identity, training, known)
                except ValueError:
                    trained[identity] = None
            if trained[identity] is not None:
                models[rule] = (*trained[identity][:-1], rule.strength)
                reset[start, col] = True
            else:
                models.setdefault(rule, parent)
                failed.append(col)
            base[start:end, col] = tools.issue(models[rule], clock[start:end])
        ids = np.flatnonzero(training)
        journal.append({"publish_hour": beginning, "last_reference": int(ids[-1]),
                        "references": len(ids), "unique_fits": sum(v is not None for v in trained.values()),
                        "failed_recipe_ids": failed})
        snapshots.append({"publish_hour": beginning,
                          "models": {rules.index(rule): models[rule] for rule in capture}})
    return base, reset, journal, snapshots


def correct_versions(base, reference, clock, setting, reset):
    """换权重时清除该模型旧偏差，每小时仍先发放再接收参考。"""
    base = np.asarray(base, dtype=float)
    reset = np.asarray(reset, dtype=bool)
    reference = np.asarray(reference, dtype=float)
    clock = np.asarray(clock)
    if (base.ndim != 2 or reset.shape != base.shape or len(base) != len(reference)
            or reference.shape != clock.shape or not np.isfinite(base).all()
            or np.any(np.diff(clock) <= 0)):
        raise ValueError("更新后的候选预测与时钟不一致")
    gain, hours = setting
    if not 0 <= gain <= 1 or hours <= 0:
        raise ValueError("反馈强度或持续时间无效")
    valid = np.zeros(base.shape[1], dtype=bool)
    residual = np.zeros(base.shape[1])
    last = np.zeros(base.shape[1])
    emitted = np.empty_like(base)
    for i, now in enumerate(clock):
        valid[reset[i]] = False
        correction = np.zeros(base.shape[1])
        correction[valid] = gain * np.exp(-(now-last[valid])/hours) * residual[valid]
        emitted[i] = np.maximum(0., base[i] + correction)
        if np.isfinite(reference[i]):
            residual = reference[i] - base[i]
            last[:] = now
            valid[:] = True
    return emitted
