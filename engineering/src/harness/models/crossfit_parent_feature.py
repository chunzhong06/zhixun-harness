"""为二阶段学习器生成没有见过本行标签的父预测，部署仍使用原完整父版。"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def crossfit_parent(parent, features, target, groups, *, classification):
    """设备分组优先；单一源设备使用固定五折，不改变源预处理或父模型超参。

    所有源标签本来都已合法可用，折外拆分仅防止同一行的训练标签通过父
    预测直接进入第二阶段。这里不预测服务数据，也不利用目标反馈定折数。
    """
    x, y, g = np.asarray(features), np.asarray(target), np.asarray(groups)
    if x.ndim != 2 or y.shape != (len(x),) or g.shape != y.shape or not np.isfinite(x).all():
        raise ValueError('源数据、标签与设备组必须逐行对应且特征有限')
    if len(np.unique(g)) > 1:
        splitter = GroupKFold(n_splits=min(5, len(np.unique(g))))
        folds = splitter.split(x, y, g)
        kind = 'group'
    elif classification:
        if np.unique(y, return_counts=True)[1].min() < 5:
            raise ValueError('固定五折要求每类至少五条源样本')
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=17).split(x, y)
        kind = 'stratified'
    else:
        folds = KFold(n_splits=5, shuffle=True, random_state=17).split(x)
        kind = 'random'
    prediction = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    records, models = [], []
    for number, (train, holdout) in enumerate(folds):
        if np.intersect1d(train, holdout).size or np.any(fold_id[holdout] != -1):
            raise ValueError('折外训练/预测行交叉或重复')
        if kind == 'group' and np.intersect1d(g[train], g[holdout]).size:
            raise ValueError('设备同时进入折内和折外')
        model = clone(parent)
        started = time.monotonic()
        model.fit(x[train], y[train])
        prediction[holdout] = model.predict(x[holdout])
        seconds = time.monotonic()-started
        fold_id[holdout] = number
        records.append({'fold': number, 'fit_rows': train.tolist(), 'heldout_rows': holdout.tolist(),
                        'seconds': seconds, 'heldout_groups': np.unique(g[holdout]).tolist()})
        models.append(model)
    if (fold_id < 0).any() or not np.isfinite(prediction).all():
        raise ValueError('源样本未全部获得唯一折外父预测')
    return prediction, fold_id, models, {'kind': kind, 'folds': records,
                                        'seconds': sum(r['seconds'] for r in records)}
