"""在已交接读数的尺度上选有代表性的可回标测量，沿用既有分类与发布流程。"""
from hashlib import sha256
from pathlib import Path

from harness.contracts import digest
from harness.methods import feedback_coverage
from harness.models import served_scale_gas

MODE = "coverage-served-scale"


def implementation():
    return digest([served_scale_gas.implementation(), sha256(Path(feedback_coverage.__file__).read_bytes()).hexdigest(),
                   sha256(Path(__file__).read_bytes()).hexdigest()])


class CoverageScaleGas(served_scale_gas.ServedScaleGas):
    def __init__(self, x, y, bank_names, classes, mode, state=None):
        if mode != MODE:
            raise ValueError("未知覆盖查询方式")
        super().__init__(x, y, bank_names, classes, served_scale_gas.MODE, state)

    def select(self, keys, x, votes, budget, seed):
        if not self.served:
            raise ValueError("当前服务预测及读数尚未交接")
        ix = feedback_coverage.coverage_queries(keys, x, None, self.served, self.history,
                                                self.answers, budget, seed, "固定均匀覆盖")
        return [keys[i] for i in ix]
