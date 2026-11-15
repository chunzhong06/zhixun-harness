"""仅在源端监督十项部件健康量，现场从航班传感器推算健康量和寿命。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from harness.models.flight_fault_experts import fit_tree, segment_prediction


class WearTransfer:
    def fit(self, views, health, truth, engines):
        flat = views.reshape(-1, views.shape[-1])
        # 仅训练折内可知各部件的量纲。每一发动机的全部航班与片段归入同一折。
        self.health_scale = np.abs(health).max(0).clip(1e-8)
        raw_health = np.repeat(health, 3, axis=0)
        targets = raw_health / self.health_scale
        def readout(x, y):
            return ExtraTreesRegressor(n_estimators=80, min_samples_leaf=4, random_state=31, n_jobs=1).fit(x, y)
        held = np.empty_like(targets)
        duration = np.empty(len(flat))
        owner = np.repeat(engines, 3)
        life = np.repeat(truth, 3)
        for train, test in GroupKFold(n_splits=5).split(flat, groups=owner):
            train_scale = np.abs(raw_health[train]).max(0).clip(1e-8)
            sensor = readout(flat[train], raw_health[train] / train_scale)
            held[test] = sensor.predict(flat[test]) * train_scale / self.health_scale
            duration[test] = fit_tree(flat[train], life[train]).predict(flat[test])
        self.health = readout(flat, targets)
        self.direct = fit_tree(flat, life)
        self.wear_model = fit_tree(held, life)
        self.stack = fit_tree(np.column_stack((flat, held, duration)), life)
        self.scale = StandardScaler().fit(held)
        # 开工与临近失效的样本数随设备不同。保留所有源航班，按设备构建映射用作动态对照。
        self.source_points = self.scale.transform(held.reshape(len(views), 3, -1).mean(1))
        self.source_life = truth
        self.source_engines = engines
        self.error = float(np.sqrt(np.mean((held - targets) ** 2)))
        return self

    def outputs(self, views):
        from scipy.spatial.distance import cdist
        flat = views.reshape(-1, views.shape[-1])
        wear = self.health.predict(flat)
        wear_only = segment_prediction(self.wear_model, wear.reshape(len(views), 3, -1))
        context = np.column_stack((flat, wear, self.direct.predict(flat))).reshape(len(views), 3, -1)
        stack = segment_prediction(self.stack, context)
        locations = self.scale.transform(wear.reshape(len(views), 3, -1).mean(1))
        values, distances = [], []
        for e in np.unique(self.source_engines):
            source = self.source_engines == e
            dist = cdist(locations, self.source_points[source], metric="sqeuclidean")
            ix = np.argmin(dist, axis=1)
            values.append(self.source_life[source][ix])
            distances.append(dist[np.arange(len(ix)), ix])
        order = np.argsort(np.stack(distances, axis=1), axis=1)
        nearest = np.take_along_axis(np.stack(values, axis=1), order, axis=1)[:, :5].mean(1)
        spread = np.std(wear.reshape(len(views), 3, -1), axis=1).mean(1)
        return {"只凭估计损耗": wear_only, "传感器和损耗联合训练": stack,
                "损耗最近五台参照": nearest}, spread
