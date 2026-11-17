"""历史模块兼容别名；实现已迁入 engine.research_recovery。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("harness.engine.research_recovery")
