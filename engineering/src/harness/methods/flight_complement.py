"""按当次航班的预测分歧，学习序列模型与森林的分量。训练端不向规则传递机队身份。"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def fixed_mix(predictions, truth, sites):
    groups = [sites == site for site in np.unique(sites)]
    count = predictions.shape[1]
    def objective(weights):
        error = predictions @ weights - truth
        return np.mean([np.sqrt(np.mean(error[group] ** 2)) for group in groups])
    result = minimize(objective, np.full(count, 1 / count), method="SLSQP",
                      bounds=[(0, 1)] * count, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                      options={"ftol": 1e-9, "maxiter": 200})
    if not result.success:
        raise ValueError(result.message)
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


class FlightComplement:
    """同一组历史航班同时拟合固定对手和受限的现场修正。"""
    def __init__(self, predictions, truth, sites):
        self.forest = fixed_mix(predictions[:, :-1], truth, sites)
        self.fixed = fixed_mix(predictions, truth, sites)
        base, gap, features = self.split(predictions)
        self.center = features.mean(axis=0)
        self.scale = features.std(axis=0).clip(.1)
        design = self.design(features)
        # 每个机队等权；目标直接对应最后输出的误差，轻微分歧的样本不会产生极端权重。
        weight = np.zeros(len(truth))
        for site in np.unique(sites):
            mask = sites == site
            weight[mask] = len(truth) / (len(np.unique(sites)) * mask.sum())
        self.alpha = self.fixed[-1]
        self.gate = Ridge(alpha=len(truth) * 1.5, fit_intercept=False).fit(
            gap[:, None] * design, truth - base - self.alpha * gap, sample_weight=weight)
        self.affine = Ridge(alpha=len(truth), fit_intercept=True).fit(
            predictions, truth, sample_weight=weight)
        self.mean_weight = float(np.average(self.weights(predictions), weights=weight))

    def split(self, predictions):
        trees, seq = predictions[:, :-1], predictions[:, -1]
        base = trees @ self.forest
        gap = seq - base
        # 包含序列与各段读法的差异，区分全部森林向同一侧偏与森林内部互有分歧。
        return base, gap, np.column_stack((base, gap, abs(gap), trees.std(axis=1),
                                         trees[:, 0] - trees[:, 2], seq - trees.min(axis=1),
                                         seq - trees.max(axis=1)))

    def design(self, features):
        return np.column_stack((np.ones(len(features)), np.clip((features - self.center) / self.scale, -3, 3)))

    def weights(self, predictions):
        _, _, features = self.split(predictions)
        return np.clip(self.alpha + self.gate.predict(self.design(features)), 0, 1)

    def forecast_all(self, predictions):
        base, gap, _ = self.split(predictions)
        fixed = predictions @ self.fixed
        adaptive = base + self.weights(predictions) * gap
        return {"航班互补调权": adaptive, "现场与六工具固定各半": (adaptive + fixed) / 2,
                "冻结现场平均权重": base + self.mean_weight * gap, "六工具固定最优": fixed,
                "固定线性校准": self.affine.predict(predictions), "森林固定最优": base,
                "论文与森林固定各半": base + .5 * gap, "固定用论文": predictions[:, -1]}
