"""历史工程替身路径兼容。"""
import sys,importlib
sys.modules[__name__]=importlib.import_module("harness.adapters.executors.fixture_backend")
