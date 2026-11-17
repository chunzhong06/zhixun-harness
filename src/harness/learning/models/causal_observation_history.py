"""只用当前及已经到达的同设备观测，更新首次观测差分和全前缀OLS斜率。"""
import numpy as np


class CausalObservationHistory:
    """设备身份仅管理独立状态；首次观测是实测参照，不假定其健康。

    step只接收一条当前观测，时间在各设备内严格递增。OLS使用含当前点的
    全部已到达前缀，按在线中心矩更新；不接收标签、未来数组或设备总长度。
    """

    def __init__(self, dimension=36):
        if type(dimension) is not int or dimension < 1:
            raise ValueError('观测维数必须为正整数')
        self.dimension = dimension
        self._states = {}

    def step(self, group, time, values):
        if not isinstance(group, str) or not group:
            raise ValueError('设备状态键必须为非空字符串')
        timestamp = float(time)
        current = np.asarray(values, dtype=np.float64)
        if current.shape != (self.dimension,) or not np.isfinite(current).all() or not np.isfinite(timestamp):
            raise ValueError('当前观测维数错误或含非有限值')
        old = self._states.get(group)
        if old is None:
            state = {'count': 1, 'first_time': timestamp, 'last_time': timestamp,
                'mean_time': timestamp, 'time_m2': 0., 'first': current.copy(),
                'mean_values': current.copy(), 'cross': np.zeros(self.dimension)}
            delta, slope = np.zeros(self.dimension), np.zeros(self.dimension)
        else:
            if timestamp <= old['last_time']:
                raise ValueError('同设备时间必须严格递增；重复或倒序输入不改变状态')
            count = old['count'] + 1
            dt = timestamp - old['mean_time']
            mean_time = old['mean_time'] + dt / count
            mean_values = old['mean_values'] + (current - old['mean_values']) / count
            time_m2 = old['time_m2'] + dt * (timestamp - mean_time)
            cross = old['cross'] + dt * (current - mean_values)
            if not np.isfinite(time_m2) or time_m2 <= 0 or not np.isfinite(cross).all():
                raise ValueError('时间尺度导致OLS中心矩非有限或退化')
            delta, slope = current - old['first'], cross / time_m2
            state = {**old, 'count': count, 'last_time': timestamp, 'mean_time': mean_time,
                     'mean_values': mean_values, 'time_m2': time_m2, 'cross': cross}
        if not np.isfinite(delta).all() or not np.isfinite(slope).all():
            raise ValueError('历史表示非有限；本条不提交状态')
        self._states[group] = state
        return delta.copy(), slope.copy()

    def snapshot(self):
        """返回可JSON序列化的独立副本，只含已到达观测的充分统计量。"""
        return {'schema_version': 1, 'dimension': self.dimension, 'states': {
            group: {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in state.items()} for group, state in self._states.items()}}

    @classmethod
    def from_snapshot(cls, snapshot):
        if set(snapshot) != {'schema_version', 'dimension', 'states'} or snapshot['schema_version'] != 1:
            raise ValueError('状态快照版本或字段不同')
        restored = cls(snapshot['dimension'])
        keys = {'count', 'first_time', 'last_time', 'mean_time', 'time_m2', 'first', 'mean_values', 'cross'}
        for group, original in snapshot['states'].items():
            if not isinstance(group, str) or not group or set(original) != keys:
                raise ValueError('快照设备键或状态字段不同')
            state = dict(original)
            if type(state['count']) is not int or state['count'] < 1:
                raise ValueError('快照已见点数无效')
            for key in ('first', 'mean_values', 'cross'):
                state[key] = np.asarray(state[key], dtype=np.float64).copy()
                if state[key].shape != (restored.dimension,) or not np.isfinite(state[key]).all():
                    raise ValueError('快照观测统计量无效')
            for key in ('first_time', 'last_time', 'mean_time', 'time_m2'):
                state[key] = float(state[key])
                if not np.isfinite(state[key]):
                    raise ValueError('快照时间统计量非有限')
            if not state['first_time'] <= state['mean_time'] <= state['last_time']:
                raise ValueError('快照时间范围无效')
            if state['count'] == 1:
                if (state['first_time'] != state['last_time'] or state['time_m2'] != 0
                        or np.any(state['cross'] != 0) or not np.array_equal(state['first'], state['mean_values'])):
                    raise ValueError('单点快照不满足零斜率条件')
            elif state['time_m2'] <= 0 or state['first_time'] >= state['last_time']:
                raise ValueError('多点快照时间未递增')
            restored._states[group] = state
        return restored
