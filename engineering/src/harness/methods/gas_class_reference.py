"""用已返回的各类典型读数修正源模型与现场之间的刻度，现场类别不进入变换。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness.models.field_classification import weights


def normalized(x, mean, scale):
    return (x - mean) / np.maximum(scale, 1e-9)


class ClassReference:
    def __init__(self, source_x, source_y):
        self.scale = StandardScaler().fit(source_x)
        self.source = self.scale.transform(source_x)
        self.y = np.asarray(source_y)
        self.classes = np.unique(self.y)
        self.centers = np.stack([self.source[self.y == c].mean(axis=0) for c in self.classes])

    def transforms(self, all_x, target_x, target_y):
        site = self.scale.transform(all_x)
        center = site.mean(axis=0)
        spread = site.std(axis=0).clip(.02)
        sims = np.zeros_like(center)
        gain = np.ones_like(center)
        offset = np.zeros_like(center)
        if len(target_x):
            tx = normalized(self.scale.transform(target_x), center, spread)
            counts = np.array([(target_y == c).sum() for c in self.classes])
            confirmed = counts >= 2
            if confirmed.sum() >= 2:
                old = self.centers[confirmed]
                new = np.stack([tx[target_y == c].mean(axis=0) for c in self.classes[confirmed]])
                old_mean, new_mean = old.mean(axis=0), new.mean(axis=0)
                da, db = old - old_mean, new - new_mean
                signal = np.mean(db ** 2, axis=0)
                cov = np.mean(da * db, axis=0)
                # 两个域的类别均值各占相同分量，避免查询频繁的类别主导校准。
                gain = np.clip((cov + .2) / (signal + .2), .2, 4.)
                offset = old_mean - gain * new_mean
                noise = np.mean(np.stack([tx[target_y == c].var(axis=0) / (target_y == c).sum()
                                          for c in self.classes[confirmed]]), axis=0)
                sims = np.clip((signal - noise) / (signal + noise + .05), 0, 1)
        return center, spread, gain, offset, sims

    def predict(self, all_x, history_x, history_y, values, mode):
        cx, sd, gain, off, confidence = self.transforms(all_x, history_x, history_y)
        strengths = {"原刻度": 0., "独立刻度": 0., "固定四分一": .25, "固定各半": .5,
                     "固定四分三": .75, "固定校准": 1., "现场证据": confidence}
        def transform(x):
            if not len(x):
                return np.empty((0, self.source.shape[1]))
            z = self.scale.transform(x)
            if mode == "原刻度":
                return z
            z = normalized(z, cx, sd)
            strength = strengths.get(mode, 0.)
            return z + strength * (z * (gain - 1) + off)
        target = transform(history_x)
        has_all = set(history_y) == set(self.classes)
        if mode == "只用现场" and has_all:
            x, y = target, history_y
            weight = weights(y, len(y))
        else:
            x = np.vstack((self.source, target))
            y = np.r_[self.y, history_y]
            weight = weights(self.y, len(self.y))
            if len(history_y):
                weight = np.r_[weight, weights(history_y, len(self.y))]
        clf = LogisticRegression(C=1, random_state=17, max_iter=1000).fit(x, y, sample_weight=weight)
        return clf.predict_proba(transform(values))
