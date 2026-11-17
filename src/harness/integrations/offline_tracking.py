"""历史模块兼容别名；实现已迁入 adapters.research.offline_tracking。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("harness.adapters.research.offline_tracking")
