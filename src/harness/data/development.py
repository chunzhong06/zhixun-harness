"""历史模块兼容别名；实现已迁入 adapters.datasets.development。"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("harness.adapters.datasets.development")
