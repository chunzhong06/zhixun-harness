"""用同种背景、同档浓度下的源端释放学习响应差，并为每次测量比较背景解释。"""
import numpy as np
from scipy.special import expit, logsumexp, softmax
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PairedGasContrast:
    def fit(self, features, groups):
        self.scale = StandardScaler().fit(features)
        x = self.scale.transform(features)
        groups = np.asarray(groups)
        conditions, codes = np.unique(groups, axis=0, return_inverse=True)
        means = np.stack([x[codes == i].mean(0) for i in range(len(conditions))])
        # 两次源端独立释放的差异给出噪声尺度，避免把物质间的差别当作噪声压掉。
        residuals = x - means[codes] if len(means) < len(x) else x - x.mean(axis=0)
        noise = OAS(assume_centered=True).fit(residuals)
        self.whitener = np.linalg.cholesky(noise.precision_)
        self.templates = means @ self.whitener
        self.present = conditions[:, 0] != "n"
        self.backgrounds = []
        for gas in ("co", "me"):
            for level in ("l", "m", "h"):
                match = (conditions[:, 1] == gas) & (conditions[:, 2] == level)
                self.backgrounds.append(match)
        self.clean = conditions[:, 2] == "n"
        self.classifier = LogisticRegression(C=1, class_weight="balanced", max_iter=1200)
        self.classifier.fit(x @ self.whitener, groups[:, 0] != "n")
        return self

    def outputs(self, features):
        x = self.scale.transform(features) @ self.whitener
        distances = cdist(x, self.templates, metric="sqeuclidean")
        # 每一背景先比较有无乙烯的距离。用激活最强的乙烯浓度显式扣除相同背景响应。
        negative = np.stack([distances[:, m & ~self.present].min(1) for m in self.backgrounds], axis=1)
        positive = np.stack([distances[:, m & self.present].min(1) for m in self.backgrounds], axis=1)
        clean_positive = distances[:, self.clean].min(1)
        distance_scale = self.templates.shape[1]
        delta = (negative - positive) / distance_scale
        by_condition = expit(delta)
        fit = np.minimum(negative, positive)
        weights = softmax(-fit / distance_scale, axis=1)
        best = np.argmin(fit, axis=1)
        profiles = -distances / distance_scale
        global_odds = (logsumexp(profiles[:, self.present], axis=1) - np.log(self.present.sum())
                       - logsumexp(profiles[:, ~self.present], axis=1) + np.log((~self.present).sum()))
        nearest = (negative.min(1) - np.minimum(positive.min(1), clean_positive)) / distance_scale
        results = {
            "固定去噪线性": self.classifier.predict_proba(x)[:, 1],
            "固定最近标样": expit(nearest),
            "固定全条件似然": expit(global_odds),
            "固定六背景等权": by_condition.mean(axis=1),
            "按测量选配对背景": by_condition[np.arange(len(x)), best],
            "按测量加权配对背景": np.sum(weights * by_condition, axis=1),
        }
        for i, (gas, level) in enumerate((g, l) for g in ("一氧化碳", "甲烷") for l in ("低", "中", "高")):
            results["固定配对" + gas + level] = by_condition[:, i]
        return results
