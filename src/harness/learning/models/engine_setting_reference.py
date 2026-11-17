"""从源训练输入学习冗余运行设置的取值，未知情况保留现场观测。"""
from collections import defaultdict

import numpy as np
from harness.core.contracts import digest
from harness.learning.models.source_combinations import serve_transformed


def setting_key(row):
    return (round(float(row[-3])), round(float(row[-2]), 2), float(row[0]))


class EngineSettingReference:
    """输入列按赛方发动机路径排列，训练和单条服务均不读取目标或设备号。"""

    def __init__(self, values):
        values = self.validate(values)
        settings = defaultdict(set)
        counts = defaultdict(int)
        for row in values:
            key = setting_key(row)
            settings[key].add(float(row[-1]))
            counts[key] += 1
        self.mapping = {k: next(iter(v)) for k, v in settings.items() if counts[k] >= 8 and len(v) == 1}
        self.identity = digest(sorted((*k, v) for k, v in self.mapping.items()))

    @staticmethod
    def validate(values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != 24 or not len(x) or not np.isfinite(x).all():
            raise ValueError("需要有效的发动机 21 个传感器和 3 个运行设置")
        return x

    def transform(self, values):
        x = self.validate(values).copy()
        for row in x:
            row[-1] = self.mapping.get(setting_key(row), row[-1])
        return x

    def advance_current(self, model, explore, confirm, serve):
        """只处理服务行；逐条预测时不需要任何本地寿命反馈。"""
        return serve_transformed(model, serve, self.transform, "source-setting")
