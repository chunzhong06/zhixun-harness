"""用源端分歧实例估计推翻基础模型的净收益，状态可序列化供发放时重用。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

ANCHOR = "固定去噪线性"
PRIMARY = "相近源测量的分歧纠错"


class RescuePolicy:
    def __init__(self, source_x, margins, labels, names):
        self.x = source_x
        self.margins = margins
        self.names = names
        self.anchor = names.index(ANCHOR)
        self.labels = labels
        self.stack = LogisticRegression(C=1, class_weight="balanced", max_iter=1000).fit(margins, labels)
        self.boost = np.where(labels, .5 / labels.mean(), .5 / (1 - labels.mean()))

    def predict(self, x, margins):
        order = np.argsort(cdist(x, self.x, metric="sqeuclidean"), axis=1)
        answers = margins >= 0
        main = answers[:, self.anchor]
        source_answers = self.margins >= 0
        main_src = source_answers[:, self.anchor]
        def choose(neighbors):
            out = main.copy()
            changes = []
            for i, mask in enumerate(neighbors):
                rivals = np.flatnonzero(answers[i] != main[i])
                # 只从源端同方向的分歧估计改对/改错。分歧太少时优先维持主模型。
                gains = []
                for j in rivals:
                    conflict = (source_answers[mask, j] != main_src[mask]) & (main_src[mask] == main[i])
                    at = mask[conflict]
                    value = np.sum(self.boost[at] * np.where(self.labels[at] == source_answers[at, j], 1., -1.))
                    gains.append(value / (len(at) + 4) if len(at) >= 2 else 0.)
                if gains and max(gains) > .12:
                    winner = rivals[int(np.argmax(gains))]
                    out[i] = answers[i, winner]
                    changes.append([int(i), self.names[winner]])
            return 1 + out, changes
        local, changes = choose(order[:, :16])
        global_choice, _ = choose(order)
        return ({PRIMARY: local, "全源分歧纠错": global_choice,
                 "固定源端堆叠": 1 + self.stack.predict(margins)}, changes)
