from dataclasses import dataclass

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
