"""用已返回答案判断哪些测量值得查询；发布预测先交接，查询只读当时的输入。"""
import numpy as np
from scipy.spatial.distance import cdist


def badge(x, p, count, rng):
    """Ash 等的末层梯度表示与平方距离抽样；同一线性分类器向所有方案提供特征。"""
    residual = -p.copy()
    residual[np.arange(len(p)), p.argmax(axis=1)] += 1
    gram = (x @ x.T) * (residual @ residual.T)
    norm = gram.diagonal()
    first = int(norm.argmax())
    selected = [first]
    distance = np.full(len(p), np.inf)
    while len(selected) < count:
        last = selected[-1]
        distance = np.minimum(distance, np.maximum(0, norm + norm[last] - 2 * gram[:, last]))
        distance[selected] = 0
        if distance.sum() > 0:
            selected.append(int(rng.choice(len(p), p=distance / distance.sum())))
        else:
            selected.append(int(rng.choice(np.setdiff1d(np.arange(len(p)), selected))))
    return selected


def graded_coverage(similarity, demand, count):
    covered = np.zeros(len(demand))
    selected = []
    for _ in range(count):
        gain = np.maximum(similarity - covered[:, None], 0).T @ demand
        gain[selected] = -np.inf
        choice = int(gain.argmax())
        selected.append(choice)
        covered = np.maximum(covered, similarity[:, choice])
    return selected


POLICIES = ("随机", "边界", "均匀覆盖", "固定半难度", "难度覆盖", "BADGE", "历史辨别力", "历史局部错误")


def select(policy, keys, x, p, count, seed, past_x, past_prob, past_y):
    count = min(count, len(x))
    if count == 0:
        return [], {}
    rng = np.random.default_rng(seed)
    # 用键排序消除输入文件顺序对平分和抽样的影响。
    order = np.argsort(keys, kind="stable")
    x, p = x[order], p[order]
    two = np.sort(p, axis=1)[:, -2:]
    hardness = 1 - (two[:, 1] - two[:, 0])
    if policy == "随机":
        return order[rng.choice(len(p), size=count, replace=False)].tolist(), {}
    if policy == "边界":
        return order[np.argsort(-hardness, kind="stable")[:count]].tolist(), {}
    if policy == "BADGE":
        return order[badge(x, p, count, rng)].tolist(), {}
    distances = cdist(x, x, "sqeuclidean") / x.shape[1]
    nonzero = distances[distances > 0]
    bandwidth = max(1e-6, float(np.median(nonzero)) if nonzero.size else 1.)
    similarity = np.exp(-distances / bandwidth)
    fixed = {"均匀覆盖": 0., "固定半难度": .5, "难度覆盖": 1.}
    demand = .1 + hardness
    demand /= demand.mean()
    alpha = fixed.get(policy, 0.)
    if policy in ("历史辨别力", "历史局部错误") and len(past_y):
        probs = np.asarray(past_prob)
        correct = probs.argmax(axis=1) + 1 == np.asarray(past_y)
        if policy == "历史辨别力":
            margin = np.sort(probs, axis=1)[:, -2:]
            old_hardness = 1 - (margin[:, 1] - margin[:, 0])
            threshold = np.median(old_hardness)
            difficult = old_hardness >= threshold
            rates = [(1 + (~correct[mask]).sum()) / (2 + mask.sum()) for mask in (difficult, ~difficult)]
            # 两类都常错时，难度无从区分；难例更常错时才应向它们倾斜。
            alpha = float(np.clip((rates[0] - rates[1]) / max(.1, rates[0]), 0, 1))
        else:
            ds = cdist(x, past_x, "sqeuclidean") / x.shape[1]
            which = np.argsort(ds, axis=1, kind="stable")[:, :min(8, len(correct))]
            proximity = np.exp(-np.take_along_axis(ds, which, axis=1) / bandwidth)
            risk = (1 + (proximity * (~correct)[which]).sum(axis=1)) / (3 + proximity.sum(axis=1))
            demand = risk / risk.mean()
            alpha = 1.
    elif policy not in (*fixed, "历史辨别力", "历史局部错误"):
        raise ValueError("未知回标方案")
    chosen = graded_coverage(similarity, (1 - alpha) + alpha * demand, count)
    return order[chosen].tolist(), {"难度或历史风险分量": alpha}
