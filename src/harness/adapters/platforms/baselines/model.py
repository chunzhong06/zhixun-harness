"""部署入口加载全部组件；独立导出时由同一推理实现生成自包含源码。"""
from .core_model import Model as BaseModel
from harness.learning.inference.complete_checkpoint import load_checkpoint

class Model(BaseModel):
    @classmethod
    def load(cls, path):
        return load_checkpoint(path, BaseModel)
