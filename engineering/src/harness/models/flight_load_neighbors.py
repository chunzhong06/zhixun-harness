"""从一趟飞行的传感器与油门关系估寿命，设备和机队身份仅用于源训练库的分组。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler


def load_profile(preparation, rows):
    scale, healthy, _, _ = preparation
    residual = rows[:, :14] - healthy.predict(scale.transform(rows[:, 14:]))
    # 按实测油门排序等分，即使燃机长时间维持一个设置也不会有空组。
    order = np.argsort(rows[:, 16], kind="stable")
    parts = np.array_split(order, min(3, len(order)))
    segments = []
    for i in range(3):
        group = residual[parts[min(i, len(parts)-1)]]
        lo, mid, hi = np.percentile(group, [25, 50, 75], axis=0)
        segments.extend((mid, hi-lo))
    return np.concatenate(segments)


RULES = ("最近设备", "机队距离加权", "最近机队", "最近设备与森林各半", "机队加权与森林各半")


class FlightLoadNeighbors:
    def __init__(self, preparation, x, y, fleets, units):
        self.preparation = preparation
        self.scale = RobustScaler().fit(x)
        self.library = self.scale.transform(x)
        self.y = np.asarray(y, dtype=float)
        self.fleets = sorted(set(fleets))
        self.blocks = {site: [np.flatnonzero((fleets == site) & (units == u))
                              for u in np.unique(units[fleets == site])] for site in self.fleets}
        config = dict(n_estimators=100, max_depth=6, random_state=17, n_jobs=1)
        self.forest = RandomForestRegressor(**config).fit(x,y)
        self.extra = ExtraTreesRegressor(**config).fit(x,y)
        self.rule = None

    @staticmethod
    def _estimate(distances, labels):
        count = min(5, distances.shape[1])
        ix = np.argsort(distances, axis=1, kind="stable")[:, :count]
        near = np.take_along_axis(distances, ix, axis=1)
        targets = np.take_along_axis(labels, ix, axis=1)
        weights = 1 / np.maximum(near, 1e-8)
        return (weights*targets).sum(1) / weights.sum(1), near.mean(1)

    def predictions(self, x):
        x = np.atleast_2d(x)
        distance = cdist(self.scale.transform(x), self.library, metric="cityblock") / self.library.shape[1]
        estimates, costs, all_dist, all_y = [], [], [], []
        result = {"负荷森林":self.forest.predict(x), "负荷随机树":self.extra.predict(x)}
        for site, devices in self.blocks.items():
            choices = np.column_stack([indices[distance[:,indices].argmin(1)] for indices in devices])
            d = np.take_along_axis(distance, choices, axis=1)
            y = self.y[choices]
            values, cost = self._estimate(d, y)
            result["固定参考"+site] = values
            estimates.append(values)
            costs.append(cost)
            all_dist.append(d)
            all_y.append(y)
        est, cost = np.column_stack(estimates), np.column_stack(costs)
        weights = 1 / np.maximum(cost, 1e-8)**2
        result["各机队等权"] = est.mean(1)
        result["最近设备"], _ = self._estimate(np.column_stack(all_dist),np.column_stack(all_y))
        result["最近机队"] = est[np.arange(len(x)),cost.argmin(1)]
        result["机队距离加权"] = (est*weights).sum(1)/weights.sum(1)
        result["最近设备与森林各半"] = (result["最近设备"]+result["负荷森林"])/2
        result["机队加权与森林各半"] = (result["机队距离加权"]+result["负荷森林"])/2
        return result

    def forecast(self, rows):
        if self.rule not in RULES:
            raise ValueError("缺少开发结束时封存的运行规则")
        x = load_profile(self.preparation, rows)
        return float(self.predictions(x)[self.rule][0])
