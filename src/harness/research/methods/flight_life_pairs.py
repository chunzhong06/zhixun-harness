"""在源端用同台航班寿命差训练单趟读法；部署输入只含当前飞行测量。"""
import numpy as np
from collections import Counter
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler


def paired_indices(keys):
    """选近、中、远三个跨度；比较中每台机器和每个跨度分别等权。"""
    left, right, weights = [], [], []
    for engine in sorted({k[:2] for k in keys}):
        order = sorted((k[2], i) for i, k in enumerate(keys) if k[:2] == engine)
        available = [lag for lag in (4, 12, 32) if lag < len(order)]
        for lag in available:
            for (_, i), (_, j) in zip(order[:-lag], order[lag:], strict=True):
                left.append(i)
                right.append(j)
                weights.append(1 / ((len(order) - lag) * len(available)))
    if not left:
        raise ValueError("源训练没有足够长的同台航班轨迹")
    return np.asarray(left), np.asarray(right), np.asarray(weights)


def matched_indices(keys, operating, method):
    """同一始点和时间间隔带内，按工况距离、随机或最远选一个后续航班。"""
    groups = {}
    for i, (site, unit, cycle) in enumerate(keys):
        groups.setdefault((site, unit), []).append((cycle, i))
    spread = operating.std(axis=0).clip(1e-6)
    condition = operating / spread
    rng = np.random.default_rng(17)
    left, right, weights = [], [], []
    for records in groups.values():
        records.sort()
        cycles, indices = np.array(records).T
        spans = []
        for lo, hi in ((3, 7), (9, 16), (24, 40)):
            pairs = []
            for cycle, i in records:
                candidates = indices[(cycles >= cycle + lo) & (cycles <= cycle + hi)]
                if len(candidates):
                    distance = np.mean((condition[candidates] - condition[i]) ** 2, axis=1)
                    if method == "近":
                        j = candidates[distance.argmin()]
                    elif method == "远":
                        j = candidates[distance.argmax()]
                    elif method == "随机":
                        j = rng.choice(candidates)
                    else:
                        raise ValueError("未知配对规则")
                    pairs.append((i, j))
            if pairs:
                spans.append(pairs)
        for pairs in spans:
            for i, j in pairs:
                left.append(i)
                right.append(j)
                weights.append(1 / (len(spans) * len(pairs)))
    return np.array(left), np.array(right), np.array(weights)


class LifeDifferenceFit:
    """相同映射与绝对寿命损失，只改变同台差分训练项的分量。"""
    def __init__(self, measurements, truth, keys, *, pairs=None):
        self.scale = StandardScaler().fit(measurements)
        self.kernel = RBFSampler(gamma=1 / measurements.shape[1], n_components=256,
                                 random_state=17).fit(self.scale.transform(measurements))
        features = self.encode(measurements)
        groups = [k[:2] for k in keys]
        counts = Counter(groups)
        weights = np.array([1 / counts[k] for k in groups])
        weights /= weights.sum()
        left, right, pair_weight = paired_indices(keys) if pairs is None else pairs
        self.pairs = len(left)
        pair_weight /= pair_weight.sum()
        differences = features[left] - features[right]
        target_difference = truth[left] - truth[right]
        if not np.array_equal(target_difference, [keys[b][2] - keys[a][2] for a, b in zip(left, right)]):
            raise ValueError("源端配对与实际剩余飞行数不符")
        self.center = weights @ features
        self.mean = weights @ truth
        centered = features - self.center
        gram = centered.T @ (weights[:, None] * centered)
        target = centered.T @ (weights * (truth - self.mean))
        pair_gram = differences.T @ (pair_weight[:, None] * differences)
        pair_target = differences.T @ (pair_weight * target_difference)
        penalty = np.eye(len(gram)) * .0008
        self.coefficients = {}
        for name, strength in (("只学绝对寿命", 0.), ("同台差分各半训练", 1.), ("同台差分四倍训练", 4.)):
            self.coefficients[name] = np.linalg.solve(gram + strength * pair_gram + penalty,
                                                      target + strength * pair_target)

    def encode(self, measurements):
        z = np.clip(self.scale.transform(measurements), -5, 5)
        return np.column_stack((z / np.sqrt(z.shape[1]), self.kernel.transform(z)))

    def predict(self, measurements):
        feats = self.encode(measurements) - self.center
        return {k: np.maximum(0, self.mean + feats @ coef) for k, coef in self.coefficients.items()}
