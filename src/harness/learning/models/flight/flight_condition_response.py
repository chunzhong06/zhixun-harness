"""在每趟飞行实际覆盖的工况附近读取传感器响应，统一到源端选定的固定参照。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor


def reference_points(prep):
    points = prep[2].cluster_centers_
    chosen = [int(np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1)))]
    for _ in range(5):
        chosen.append(int(np.argmax(cdist(points, points[chosen]).min(1))))
    return points[chosen]


def read_conditions(rows, prep, points):
    scaler, healthy, _, _ = prep
    settings = scaler.transform(rows[:, 14:])
    residual = rows[:, :14] - healthy.predict(settings)
    x = np.column_stack((np.ones(len(rows)), settings))
    system = x.T @ x + len(rows) * np.diag([.00001, .08, .08, .08, .08])
    beta = np.linalg.solve(system, x.T @ residual)
    distances = cdist(points, settings)
    ix = np.argsort(distances, axis=1)[:, :max(1, int(np.ceil(len(rows) / 8)))]
    profiles = []
    for point, neighborhood in zip(points, ix, strict=True):
        nearby = residual[neighborhood]
        q = np.percentile(nearby, [25, 50, 75], axis=0)
        # 用整趟测得的响应斜率，把相邻实测点校到共同工况；离得远时会外推，距测量的距离另作证据。
        corrected = nearby + (point - settings[neighborhood]) @ beta[1:]
        profiles.append(np.r_[np.median(corrected,axis=0), q[1], q[2]-q[0], point])
    return np.stack(profiles), np.take_along_axis(distances, ix, axis=1).mean(1)


def measured_conditions(rows, prep, points):
    """仅发布六个邻域的实测中位数和四分位差，与消融中的32个输入一致。"""
    scaler, healthy, _, _ = prep
    settings = scaler.transform(rows[:,14:])
    residual = rows[:,:14] - healthy.predict(settings)
    nearest = np.argsort(cdist(points, settings), axis=1)[:, :max(1, int(np.ceil(len(rows)/8)))]
    q1, mid, q3 = np.percentile(residual[nearest], [25,50,75], axis=1)
    return np.column_stack((mid, q3-q1, points))


class ConditionReadout:
    def fit(self, x, truth):
        self.model = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=5, random_state=17, n_jobs=1)
        self.model.fit(x.reshape(-1, x.shape[-1]), np.repeat(truth, x.shape[1]))
        return self

    def readings(self, x):
        flat = x.reshape(-1, x.shape[-1])
        trees = np.stack([t.predict(flat).reshape(x.shape[:2]) for t in self.model.estimators_])
        return trees.mean(0), trees.std(0)
