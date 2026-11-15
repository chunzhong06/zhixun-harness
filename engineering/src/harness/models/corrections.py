"""冻结主预测器上的局部修正契约与组合；机制依据见N089。

回归学习合法新资料的残差，分类在冻结区域内替换预测。区域只读预测时
可见的特征，范围外保持基础输出；首版固定一层，不递归叠加修正器。
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CorrectionSpec:
    base_model_id: str
    mode: str = "residual"
    region_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.base_model_id, str) or not self.base_model_id:
            raise ValueError("修正需要已知基础模型")
        if self.mode not in {"residual", "region_replace"}:
            raise ValueError("未知修正方式")
        if self.region_id is not None and (not isinstance(self.region_id, str) or not self.region_id):
            raise ValueError("修正区域身份无效")
        if self.mode == "region_replace" and self.region_id is None:
            raise ValueError("分类替换需要冻结区域")

    def validate_task(self, task):
        if (task.task_type == "regression") != (self.mode == "residual"):
            raise ValueError("修正方式与任务目标不匹配")


def validate_predicates(predicates, features):
    if not isinstance(predicates, (list, tuple)) or len(predicates) > 2:
        raise ValueError("修正范围最多两个冻结条件")
    for p in predicates:
        if (not isinstance(p, dict) or type(p.get("index")) is not int
                or not 0 <= p["index"] < features or p.get("side") not in {"le", "gt"}
                or type(p.get("cut")) not in {int, float} or not math.isfinite(p["cut"])):
            raise ValueError("冻结区域条件无效")


def inside(values, predicates):
    return all(values[p["index"]] <= p["cut"] if p["side"] == "le"
               else values[p["index"]] > p["cut"] for p in predicates)


def combine(base, correction, inputs, mode, predicates):
    """只改范围内已计算修正的输入；标签编号从不作为数值残差相加。"""
    result = dict(base)
    for row in inputs:
        sid = row["sample_id"]
        if inside(row["values"], predicates):
            result[sid] = base[sid] + correction[sid] if mode == "residual" else correction[sid]
    return result


@dataclass
class CorrectedPipeline:
    """整套保存冻结基础对象与修正对象；搬移后不依赖另一目录的权重文件。"""
    base: object
    correction: object
    mode: str
    predicates: tuple


def selection_key(batch_id, action):
    key = f"{batch_id}:selection:{action.recipe}:{action.ratio:g}"
    # 修正排除基础训练样本并按区域取数，与普通同资料配方分开记清单。
    return key + ":" + action.key if action.correction or action.fit_manifest else key
