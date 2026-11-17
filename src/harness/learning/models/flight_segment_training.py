"""在同一航班的三个连续片段上学习同一个剩余寿命，发放时只用当前航班。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def segments(prep, rows):
    scale, healthy, _, _ = prep
    if len(rows) < 3:
        raise ValueError("三个航段各需至少一个读数")
    operating = rows[:, 14:]
    residual = rows[:, :14] - healthy.predict(scale.transform(operating))
    chunks = np.array_split(np.arange(len(rows)), 3)
    vectors = []
    for ix in chunks:
        x = residual[ix]
        q = np.percentile(x, [25, 50, 75], axis=0)
        vectors.append(np.r_[q[1], q[2] - q[0], np.mean(abs(x), axis=0), operating[ix].mean(0)])
    return np.stack(vectors)


class SegmentLearner:
    def fit(self, views, targets):
        flat = views.reshape(-1, views.shape[-1])
        labels = np.repeat(targets, 3)
        self.tree = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=5, random_state=17, n_jobs=1)
        self.tree.fit(flat, labels)
        self.linear = make_pipeline(StandardScaler(), Ridge(alpha=len(targets) * .025))
        self.linear.fit(flat, labels)
        self.whole = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=5, random_state=17, n_jobs=1)
        self.whole.fit(views.mean(1), targets)
        return self

    def outputs(self, views):
        n, count, width = views.shape
        flat = views.reshape(-1, width)
        segments = self.tree.predict(flat).reshape(n, count)
        linear = self.linear.predict(flat).reshape(n, count)
        return np.column_stack((segments, linear.mean(1), self.whole.predict(views.mean(1))))
