"""历史对照模块别名；作者控制器不属于生产内核。"""
import sys,importlib
sys.modules[__name__]=importlib.import_module("experiments.baselines.author_comind")
