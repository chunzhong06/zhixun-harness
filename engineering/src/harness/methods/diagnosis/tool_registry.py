"""登记经过本地检查的只读调查工具；计划内容只能引用已登记身份。"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import inspect
import importlib.util
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CheckCapability:
    name: str
    version: str
    input: str
    signal: str
    threshold_min: float = 0.
    cost_demo_units: int = 1
    dependencies: tuple[str, ...] = ()


class CheckRegistry:
    """由本地开发注册函数，试调用通过后启用；Qwen没有导入或执行源码权限。

    executor接收task、fit、explore、当前探索预测、阈值，返回signal/observations。
    来源文件和依赖声明进入运行身份。注册检查验证接口，工业效用由后续共同试改评价。
    """
    def __init__(self):
        self.entries = {}

    def register(self, capability, executor, sample_call):
        if not capability.name.isidentifier() or not capability.version or capability.name in {"feature_scale","error_slices","tail_profile","error_regions"}:
            raise ValueError("工具身份无效或与内置工具重名")
        if capability.name in self.entries:
            raise ValueError("已登记工具不可原地覆盖")
        if type(capability.cost_demo_units) is not int or capability.cost_demo_units < 1 or not math.isfinite(capability.threshold_min):
            raise ValueError("工具成本或阈值声明无效")
        source = inspect.getsourcefile(executor)
        if not source or not Path(source).is_file():
            raise ValueError("登记工具需要可核对的本地源文件")
        if any(importlib.util.find_spec(name) is None for name in capability.dependencies):
            raise ValueError("登记工具缺少已声明依赖")
        observed = executor(**deepcopy(sample_call))
        self.validate_result(observed)
        manifest = {**asdict(capability),"source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    "interface_verified":True,"effect_verified":False}
        self.entries[capability.name] = {"manifest":manifest,"execute":executor,"source":source}
        return deepcopy(manifest)

    @staticmethod
    def validate_result(value):
        if not isinstance(value,dict) or set(value) != {"signal","observations"} or (value["signal"] is not None and type(value["signal"]) is not bool) or not isinstance(value["observations"],list):
            raise ValueError("登记工具必须返回signal和observations")
        json.dumps(value,allow_nan=False)

    def catalog(self):
        return {key:deepcopy(value["manifest"]) for key,value in self.entries.items()}

    def execute(self, name, **inputs):
        entry = self.entries[name]
        if hashlib.sha256(Path(entry["source"]).read_bytes()).hexdigest() != entry["manifest"]["source_sha256"]:
            raise ValueError("登记工具源码已变化，请重新核对并登记")
        result = entry["execute"](**deepcopy(inputs))
        self.validate_result(result)
        return result


CHECK_TOOLS = CheckRegistry()
