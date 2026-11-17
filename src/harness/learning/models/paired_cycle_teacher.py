"""用两台芯片测量同一组标气时的读数和实际浓度，校准目标芯片。"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from harness.learning.models.cycle_install import balanced_f1, mean_inputs, ridge

TARGET = "直接预测目标浓度"
PAIRED = "按同配方源机预测修正芯片偏差"
INVERSE = "按目标标气真值修正源预测偏差"
PRIMARY = "按当前测量的校准覆盖程度合并源与本机训练"


class PairedCycleTeacher:
    def fit(self, source, y, target, paired, strength):
        """paired 给出每种目标标气在源标气表中的位置。"""
        source_inputs = mean_inputs(source)
        target_inputs = mean_inputs(target)
        self.source = ridge(source_inputs, y)
        self.scale = StandardScaler().fit(target_inputs)
        reference = self.scale.transform(target_inputs)
        source_prediction = self.source.predict(target_inputs)
        self.truth = Ridge(alpha=strength).fit(reference, y[paired])
        self.correction = Ridge(alpha=strength).fit(reference, y[paired] - source_prediction)
        paired_prediction = self.source.predict(source_inputs[paired])
        self.internal = Ridge(alpha=strength).fit(reference, paired_prediction - source_prediction)
        self.reference = reference
        self.gram = reference @ reference.T + strength * np.eye(len(reference))
        return self

    def predict(self, groups):
        inputs = mean_inputs(groups)
        z = self.scale.transform(inputs)
        source = self.source.predict(inputs)
        target = self.truth.predict(z)
        corrected = source + self.correction.predict(z)
        paired = source + self.internal.predict(z)
        # 用装机样本能够解释的信号比例决定现场权重，预测阶段无需答案。
        cross = z @ self.reference.T
        explained = np.sum(np.linalg.solve(self.gram, cross.T).T * cross, axis=1)
        fraction = np.clip(explained / (np.sum(z * z, axis=1) + 1e-9), 0, 1)
        local = (target + corrected) / 2
        return {
            TARGET: target,
            PAIRED: paired,
            INVERSE: corrected,
            PRIMARY: fraction * local + (1 - fraction) * paired,
            "三种目标监督固定等权": (target + corrected + paired) / 3,
            "本机与源校正固定各半": local,
        }


def install_profiles(source, y, target, paired):
    """装机试验只用共同配方，以整种标气留出，冻结固定和现场两种权重。"""
    threshold = float(np.median(y))
    folds = np.arange(len(paired)) % 5
    scores = {}
    for strength in (1., 15., 100.):
        predicted = np.empty(len(paired))
        for fold in np.unique(folds):
            test = folds == fold
            keep = np.ones(len(source), dtype=bool)
            keep[paired[test]] = False
            remaining = np.cumsum(keep) - 1
            model = PairedCycleTeacher().fit(
                [group for group, keep_group in zip(source, keep, strict=True) if keep_group],
                y[keep], [target[i] for i in np.flatnonzero(~test)],
                remaining[paired[~test]], strength,
            )
            predicted[test] = model.predict([target[i] for i in np.flatnonzero(test)])[PRIMARY]
        error = np.mean((y[paired] - predicted) ** 2)
        scores[strength] = (balanced_f1(y[paired], predicted, threshold), -float(error))
    best = max(scores, key=scores.get)
    models = {s: PairedCycleTeacher().fit(source, y, target, paired, s) for s in {1., best}}
    return {"threshold": threshold, "selected": best, "models": models}
