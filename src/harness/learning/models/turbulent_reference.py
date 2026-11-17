"""在源端标气上学习两个背景的读法，推理只接收本次暴露的传感器特征。"""
from io import BytesIO

import numpy as np
from scipy.special import expit, softmax
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def exposure_response(raw):
    samples = np.loadtxt(BytesIO(raw), delimiter=",")
    t = samples[:, 0]
    adc = samples[:, 3:]
    if np.any((adc <= 0) | (adc >= 3110)):
        raise ValueError("传感器超出作者给出的电阻转换范围")
    # 取对数后，各传感器相对本次洁净空气的电导变化可直接相减。
    conductance = np.log(adc / (3110 - adc))
    base = np.median(conductance[t < 60], axis=0)
    windows = [(60, 105), (105, 150), (150, 195), (195, 240), (240, 285)]
    return np.concatenate([np.mean(conductance[(t >= a) & (t < b)], axis=0) - base for a, b in windows])


class BackgroundReference:
    """为甲烷及一氧化碳分别拟合识别器；源端单气标样决定两种背景各像当前读数多少。"""

    def fit(self, features, labels, groups):
        self.scale = StandardScaler(with_mean=False).fit(features)
        x = self.scale.transform(features)
        ethylene, background, dose = np.asarray(groups).T
        self.models = []
        self.reference = []
        # 仅靠乙烯的原件可给两种背景分支共同训练。其余原件按标定气体区分。
        pure_e = x[(ethylene != "n") & (dose == "n")]
        for gas in ("me", "co"):
            eligible = (background == gas) | (dose == "n")
            self.models.append(SVC(C=4, class_weight="balanced").fit(x[eligible], labels[eligible]))
            pure_b = x[(background == gas) & (ethylene == "n")]
            # 用两种单气测量及其简单叠加预估混合响应。偏离越大，背景归属越不确定。
            mixed = (pure_e[:, None, :] + pure_b[None, :, :]).reshape(-1, x.shape[1])
            self.reference.append(np.vstack((pure_e, pure_b, mixed)))
        self.pooled = SVC(C=4, class_weight="balanced").fit(x, labels)
        self.linear = LogisticRegression(C=1, class_weight="balanced", max_iter=1000).fit(x, labels)
        dist = cdist(x, x, metric="sqeuclidean")
        self.distance_scale = max(float(np.median(dist[np.triu_indices(len(x), 1)])) / 8, 1e-8)
        return self

    def outputs(self, features):
        x = self.scale.transform(features)
        prob = np.column_stack([expit(model.decision_function(x)) for model in self.models])
        distance = np.column_stack([cdist(x, ref, metric="sqeuclidean").min(axis=1) for ref in self.reference])
        weights = softmax(-distance / self.distance_scale, axis=1)
        return {
            "全部标样核": expit(self.pooled.decision_function(x)),
            "全部标样线性": expit(self.linear.decision_function(x)),
            "固定甲烷读法": prob[:, 0], "固定一氧化碳读法": prob[:, 1],
            "固定各半": prob.mean(axis=1),
            "标气距离选择背景": prob[np.arange(len(x)), distance.argmin(axis=1)],
            "标气距离组合背景": (prob * weights).sum(axis=1),
        }
