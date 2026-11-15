"""给当前可查询测量选代表点；合法回标用于判断哪些邻域已有证据或仍在出错。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from harness.contracts import digest


def coverage_queries(keys, values, probabilities, observed, history, labels, cap, seed, policy):
    """服务读数只用于估计距离尺度；已回标以外的类别与服务答案不可见。"""
    cap = min(cap, len(keys))
    if not cap:
        return []
    scale = StandardScaler().fit(observed)
    x = np.clip(scale.transform(values), -5, 5)
    distances = cdist(x, x, "sqeuclidean") / x.shape[1]
    positive = distances[distances > 0]
    bandwidth = max(1e-6, float(np.median(positive)) if positive.size else 1.)
    similarity = np.exp(-distances / bandwidth)
    covered = np.zeros(len(keys))
    if policy == "固定均匀覆盖":
        demand = np.ones(len(keys))
    else:
        two = np.sort(probabilities, axis=1)[:, -2:]
        demand = .1 + 1 - (two[:, 1] - two[:, 0])
    if policy == "历史邻域" and len(history):
        previous = np.clip(scale.transform(history), -5, 5)
        ds = cdist(x, previous, "sqeuclidean") / x.shape[1]
        n = min(3, len(previous))
        nearest = np.argsort(ds, axis=1, kind="stable")[:, :n]
        affinity = np.exp(-np.take_along_axis(ds, nearest, axis=1) / bandwidth)
        same = np.asarray(labels)[nearest] == (probabilities.argmax(axis=1) + 1)[:, None]
        # 历史回标若支持当前预测，可部分抵扣覆盖；相反反馈保留该邻域的查询优先级。
        covered = .8 * (affinity * same).mean(axis=1)
    elif policy not in ("历史邻域", "固定难度覆盖", "固定均匀覆盖"):
        raise ValueError("未知回标覆盖规则")
    order = np.array(sorted(range(len(keys)), key=lambda i: digest([seed, keys[i]])))
    chosen = []
    for _ in range(cap):
        gain = np.maximum(similarity - covered[:, None], 0).T @ demand
        gain[chosen] = -np.inf
        best = int(order[gain[order].argmax()])
        chosen.append(best)
        covered = np.maximum(covered, similarity[:, best])
    return chosen
