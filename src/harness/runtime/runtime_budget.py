"""旧模块路径兼容入口。"""
import sys,runpy,importlib
if __name__=="__main__":runpy.run_module("harness.engine.legacy.runtime_budget",run_name="__main__")
else:sys.modules[__name__]=importlib.import_module("harness.engine.legacy.runtime_budget")
