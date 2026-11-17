"""历史模块兼容别名；实现已迁入 research.methods.planning.error_slice。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("harness.research.methods.planning.error_slice")
