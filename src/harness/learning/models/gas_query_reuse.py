"""对同一官方权重的相同筛选续训复用产物，探索和采用仍在每个批次单独评价。"""
from hashlib import sha256
from pathlib import Path

from harness.core.contracts import digest
from harness.learning.models.gas_observed_queries import ObservedGasQueryTools


class ReusableGasQueryTools:
    reuse_identical_fit = True

    def __init__(self, *args):
        self.tools = ObservedGasQueryTools(*args)
        self.identity = digest([self.tools.identity, sha256(Path(__file__).read_bytes()).hexdigest()])

    def __getattr__(self, name):
        # 提案的恢复身份和官方训练请求保持原义；外层身份只改变是否重用已训练权重。
        return getattr(self.tools, name)

    def fit_identity(self, task, action, rows, parent):
        """赛方此入口固定种子和优化器，探索标签只写评分，不参与梯度或早停。"""
        ids = tuple(r.sample_id for r in rows)
        if (task != self.package.task or action.epochs != 1 or action.focus_label is not None
                or action.gas_pair or ids != self.catalog.selected.get(action.fit_manifest)
                or any(self.train.get(r.sample_id) != r for r in rows)):
            raise ValueError("复用筛选训练需要相同的有序官方行和一次普通续训")
        manifest,_ = self.runner.artifacts.read(parent)
        return digest([self.identity, self.package.train_sha256, manifest["sha256"], ids, action.epochs])
