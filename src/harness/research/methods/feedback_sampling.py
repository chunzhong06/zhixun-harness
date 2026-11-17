"""从本批可回标的无标签候选里选编号。误差历史仅来自先前均匀抽取的探针。"""
import numpy as np

from harness.core.contracts import digest


def select_feedback(keys, probabilities, budget, seed, policy, error_counts):
    p = np.asarray(probabilities, dtype=float)
    counts = np.asarray(error_counts, dtype=int)
    if (p.ndim != 2 or len(p) != len(keys) or p.shape[1] < 2 or counts.shape != (p.shape[1], 2)
            or np.any(counts < 0) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1) or len(set(keys)) != len(keys) or budget < 0):
        raise ValueError("回标池、概率、额度或历史计数无效")
    cap = min(budget, len(keys))
    order = sorted(range(len(keys)), key=lambda i: digest([seed, keys[i]]))
    if policy == "随机":
        return order[:cap], order[:cap]
    two = np.sort(p, axis=1)[:, -2:]
    margin = two[:, 1] - two[:, 0]
    boundary = sorted(order, key=lambda i: margin[i])
    if policy == "边界":
        return boundary[:cap], []
    if policy not in ("类别均衡", "均衡边界", "固定半随机", "历史错误"):
        raise ValueError("未知回标策略")
    labels = p.argmax(axis=1)
    mixed = policy in ("固定半随机", "历史错误")
    probes = order[:(cap + 1) // 2] if mixed else []
    occupied = set(probes)
    ranked = order if policy == "类别均衡" else boundary
    queues = [[i for i in ranked if i not in occupied and labels[i] == c] for c in range(p.shape[1])]
    # 分摊的是探针之外的剩余名额；类内无候选时把名额交给仍有候选的类别。
    weight = ((1 + counts[:, 1]) / (2 + counts.sum(axis=1)) if policy == "历史错误"
              else np.ones(p.shape[1]))
    used = np.zeros(p.shape[1], dtype=int)
    selected = list(probes)
    while len(selected) < cap:
        active = [c for c, q in enumerate(queues) if used[c] < len(q)]
        label = max(active, key=lambda c: weight[c] / (1 + used[c]))
        selected.append(queues[label][used[label]])
        used[label] += 1
    return selected, probes
