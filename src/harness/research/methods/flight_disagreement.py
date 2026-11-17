"""用跨机队、未参与对应训练的预测学习怎样分配各工具的权重。"""
import numpy as np
from scipy.optimize import minimize


def disagreement(predictions):
    return np.max(predictions, axis=1) - np.min(predictions, axis=1)


def fit_weights(predictions, labels, weight):
    count = predictions.shape[1]
    prior = np.full(count, 1 / count)
    sample_weight = np.asarray(weight, dtype=float)
    sample_weight = sample_weight / sample_weight.sum()
    def loss(p):
        return float(sample_weight @ ((predictions @ p - labels)**2))
    result = minimize(loss, prior, method="SLSQP", bounds=[(0, 1)] * count,
                      constraints=[{"type":"eq","fun":lambda p: p.sum()-1}],
                      options={"ftol":1e-8,"maxiter":200})
    if not result.success:
        raise ValueError("有限权重拟合失败：" + result.message)
    w = np.maximum(0, result.x)
    return w / w.sum()


def fit_disagreement(predictions, labels, fleets, units):
    """每个机队等权，每台机器在自己机队内等权。有限混合保留单工具两端。"""
    weight = np.zeros(len(labels))
    for site in np.unique(fleets):
        on_site = fleets == site
        machines = np.unique(units[on_site])
        for unit in machines:
            group = on_site & (units == unit)
            weight[group] = 1 / (len(machines) * group.sum())
    global_weights = fit_weights(predictions, labels, weight)
    gaps = disagreement(predictions)
    cuts = np.quantile(gaps, [1/3, 2/3])
    groups = np.searchsorted(cuts, gaps, side="right")
    local = []
    for g in range(3):
        keep = groups == g
        # 稀少分歧组以一台设备的有效样本量向公共组合收缩。
        shrunk = global_weights
        if keep.any():
            specific = fit_weights(predictions[keep], labels[keep], weight[keep])
            count = len(set(zip(fleets[keep], units[keep])))
            shrunk = (count * specific + global_weights) / (1 + count)
        local.append(shrunk)
    return {"cuts":cuts.tolist(),"global":global_weights.tolist(),"local":np.array(local).tolist()}


def combine(predictions, parameters):
    groups = np.searchsorted(parameters["cuts"], disagreement(predictions), side="right")
    return np.sum(predictions * np.asarray(parameters["local"])[groups], axis=1)
