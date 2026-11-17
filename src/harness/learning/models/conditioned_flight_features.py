"""把单航班传感器与工况配对汇总；拟合仅接收合法训练航班，无标签或身份。"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


SENSORS = ('T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24',
           'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf')
SETTINGS = ('alt', 'Mach', 'TRA', 'T2')
PHYSICAL = SENSORS + SETTINGS
BASE_VALUES = tuple(f'{stat}_{name}' for stat in ('mean', 'std') for name in PHYSICAL) + ('elapsed_flight',)
BASE_COLUMNS = BASE_VALUES + tuple('present_' + name for name in BASE_VALUES)


class ConditionedFlightFeatures:
    """输入是当前完整航班每16点取1点后的(n,18)数组，前14列为传感器。

    训练航班各贡献最多32个等距工况点；6个锚点和所有随机性固定。
    transform不更新状态，打乱臂只破坏同航班传感器块与工况的配对。
    """

    # 每个锚点的28项传感器统计紧接1项距离，与transform_one的行展开一致。
    feature_names = tuple(name for i in range(6) for name in
                          ([f'condition_{i}_{stat}_{sensor}' for stat in ('median', 'iqr') for sensor in SENSORS]
                           + [f'condition_{i}_distance']))

    @staticmethod
    def _array(flight):
        rows = np.asarray(flight, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 18 or not len(rows) or not np.isfinite(rows).all():
            raise ValueError('单航班必须提供非空、有限且按实测顺序排列的18列')
        return rows

    def fit(self, flights):
        """只在合法训练航班调用一次；不接受y，不使用未来工况拟合尺度。"""
        if hasattr(self, 'centers_'):
            raise ValueError('本有界试验的工况参照只允许拟合一次')
        samples = []
        for flight in flights:
            rows = self._array(flight)
            take = np.linspace(0, len(rows) - 1, min(32, len(rows)), dtype=int)
            samples.append(rows[take, 14:])
        if not samples or sum(map(len, samples)) < 6:
            raise ValueError('合法训练工况点不足')
        operating = np.concatenate(samples)
        with threadpool_limits(limits=1):
            self.scaler_ = StandardScaler().fit(operating)
            model = KMeans(n_clusters=6, random_state=42, n_init=1, max_iter=100)
            self.centers_ = model.fit(self.scaler_.transform(operating)).cluster_centers_.copy()
        self.training_flights_ = len(samples)
        self.training_condition_points_ = len(operating)
        return self

    def transform_one(self, flight, *, shuffled=False):
        check_is_fitted(self, ['scaler_', 'centers_'])
        rows = self._array(flight)
        operating = self.scaler_.transform(rows[:, 14:])
        sensors = rows[:, :14]
        if shuffled:
            # 每航班重建生成器，不依赖调用顺序；整块置换保留传感器间的配对。
            sensors = sensors[np.random.default_rng(42).permutation(len(rows))]
        distances = np.linalg.norm(self.centers_[:, None, :] - operating[None, :, :], axis=2)
        neighbors = np.argsort(distances, axis=1, kind='stable')[:, :int(np.ceil(len(rows) / 8))]
        q1, median, q3 = np.percentile(sensors[neighbors], [25, 50, 75], axis=1)
        coverage = np.take_along_axis(distances, neighbors, axis=1).mean(axis=1)
        values = np.column_stack((median, q3 - q1, coverage)).reshape(-1)
        if values.shape != (174,) or not np.isfinite(values).all():
            raise ValueError('工况摘要须恰好返回174项有限数值')
        return values

    def transform(self, flights, *, shuffled=False):
        """批量接口仅逐航班调用，无批次统计或跨航班状态。"""
        return np.asarray([self.transform_one(f, shuffled=shuffled) for f in flights], dtype=np.float64).reshape(-1, 174)
