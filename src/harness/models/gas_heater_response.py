"""历史模块兼容别名；实现已迁入 learning.models.gas_heater_response。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("harness.learning.models.gas_heater_response")
