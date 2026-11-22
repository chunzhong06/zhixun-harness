"""以单航段训练树的分歧估计三段相对可信度，限制分量围绕固定源端解变化。"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler


def flight_evidence(learner, views):
    n, count, width = views.shape
    flat = views.reshape(-1, width)
    trees = np.stack([tree.predict(flat).reshape(n, count) for tree in learner.tree.estimators_])
    readings = trees.mean(axis=0)
    spreads = np.log1p(trees.std(axis=0))
    # 波动比较相对同一航班的另外两段，可以跨模型输出的绝对尺度使用。
    return readings, np.stack((spreads, spreads - spreads.mean(1)[:, None],
                               abs(readings - np.median(readings, axis=1)[:, None]),
                               readings - readings.mean(1)[:, None],
                               np.broadcast_to(readings.mean(1)[:, None], spreads.shape)), axis=2)


class SegmentPrecision:
    def fit(self, predictions, evidence, truth, sites):
        self.scale = StandardScaler().fit(evidence.reshape(-1, evidence.shape[-1]))
        e = self.transform(evidence)
        groups = [np.flatnonzero(sites == v) for v in np.unique(sites)]
        def fit_one(use_context):
            def objective(params):
                logits = params[:3] + (e @ params[3:] if use_context else 0)
                weights = np.broadcast_to(softmax(logits, axis=-1), predictions.shape)
                pred = np.sum(weights * predictions, axis=1)
                error = pred - truth
                # 对六组等权RMSE的同形式目标求解析梯度，直接优化段间组合造成的最终寿命误差。
                grad = np.zeros(len(params))
                loss = 0.
                for ix in groups:
                    rmse = np.sqrt(np.mean(error[ix] ** 2) + 1e-8)
                    loss += rmse / len(groups)
                    dz = (error[ix] / (rmse * len(ix) * len(groups)))[:, None] * weights[ix] * (predictions[ix] - pred[ix, None])
                    grad[:3] += dz.sum(0)
                    if use_context:
                        grad[3:] += np.einsum("ns,nsf->f", dz, e[ix])
                regularization = np.r_[np.full(3, .025), np.full(len(params) - 3, .10)]
                return loss + .5 * sum(regularization * params ** 2), grad + regularization * params
            result = minimize(objective, np.zeros(3 + (e.shape[-1] if use_context else 0)), jac=True, method="L-BFGS-B")
            if not result.success:
                raise ValueError(result.message)
            return result.x
        self.context = fit_one(True)
        self.fixed = softmax(fit_one(False))
        return self

    def transform(self, e):
        return np.clip(self.scale.transform(e.reshape(-1, e.shape[-1])).reshape(e.shape), -5, 5)

    def weights(self, evidence):
        return softmax(self.context[:3] + self.transform(evidence) @ self.context[3:], axis=1)
