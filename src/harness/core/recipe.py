"""可执行的成熟组件组合；声明式配方随模型保存，不执行计划中的任意代码。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

CATALOG_VERSION = 3
CATBOOST_DEFAULT_DEPTH = 6


@dataclass(frozen=True)
class PipelineRecipe:
    """处理、特征与预测器整体重建；所有有状态组件仅在fit资料上拟合。

    predictor描述模型族，分类/回归由TaskSpec决定。seed属于公共训练
    后端，避免把更换随机种子当成一个新的处理机制。
    """
    processor: str = "standard"
    features: str = "all"
    predictor: str = "linear"
    regularization: float = 1.0
    max_iter: int = 1000
    trees: int = 100
    max_depth: int | None = None
    learning_rate: float = .05
    class_weight: str = "none"
    regression_loss: str = "default"

    def __post_init__(self):
        if self.processor not in {"none", "standard", "robust", "condition_standard"}:
            raise ValueError("未知处理器")
        if self.features not in {"all", "nonconstant"} or self.predictor not in {"linear", "forest", "catboost"}:
            raise ValueError("未知特征或预测器")
        if (type(self.regularization) not in {int, float} or not math.isfinite(self.regularization)
                or not 0 < self.regularization <= 10000):
            raise ValueError("正则强度无效")
        if type(self.max_iter) is not int or not 1 <= self.max_iter <= 10000:
            raise ValueError("迭代上限无效")
        if type(self.trees) is not int or not 1 <= self.trees <= 512:
            raise ValueError("树数量无效")
        if self.max_depth is not None and (type(self.max_depth) is not int or not 1 <= self.max_depth <= 64):
            raise ValueError("树深度无效")
        if (type(self.learning_rate) not in {int, float} or not math.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= 1):
            raise ValueError("提升树学习率无效")
        if self.class_weight not in {"none", "balanced"} or self.regression_loss not in {"default", "mae"}:
            raise ValueError("训练权重或回归损失无效")
        if self.predictor == "catboost":
            # 本地CPU候选限制树深，减少指数增长的叶数；并非作者推荐的最优配置。
            if self.max_depth is not None and self.max_depth > 10:
                raise ValueError("当前CPU提升树深度上限为10")
        elif self.learning_rate != .05 or self.regression_loss != "default":
            raise ValueError("学习率和可选回归损失仅用于catboost")

    def validate_task_type(self, task_type):
        """跨任务复用配方时检查目标相容性，避免无效字段被后端静默忽略。"""
        if task_type == "regression" and self.class_weight != "none":
            raise ValueError("回归任务不使用类别权重")
        if task_type == "classification" and self.regression_loss != "default":
            raise ValueError("分类任务不使用回归损失")

    def validate_task(self, task):
        self.validate_task_type(task.task_type)
        if self.processor == "condition_standard" and not task.condition_features:
            raise ValueError("按工况处理需要任务明确提供预测时可见的工况字段")

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("组件配方须为对象")
        try:
            return cls(**value)
        except TypeError:
            raise ValueError("组件配方含未知字段") from None

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def component_catalog():
    """计划器可见的能力目录；扩展工具时同时维护本模块和后端工厂。"""
    return {"version": CATALOG_VERSION, "processor": ["none", "standard", "robust", "condition_standard"],
            "features": ["all", "nonconstant"], "predictor": ["linear", "forest", "catboost"],
            "class_weight": ["none", "balanced"], "regression_loss": ["default", "mae"],
            "parameters": {"regularization": "0 < value <= 10000", "max_iter": [1, 10000],
                           "trees": [1, 512], "max_depth": "null or integer 1..64; catboost <=10",
                           "learning_rate": "catboost only: 0 < value <= 1"},
            "conditions": {"class_weight": "仅分类；balanced按实际fit标签计算逆频率样本权重",
                           "condition_standard": "任务须声明condition_features和精度；组内至少8行，否则使用fit全局统计；保留工况输入",
                           "regression_loss": "mae仅用于catboost回归；default沿用各模型原损失，catboost为RMSE",
                           "catboost": "CPU单线程、固定迭代；需要另有catboost依赖；无隐式验证划分"},
            "default": asdict(PipelineRecipe()),
            "execution": "完整组件组合在拟合资料上重建，评分与采用由公共工具完成"}
