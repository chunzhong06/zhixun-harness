"""SDK加载的选手模块，实际训练、预测和状态接续由项目适配层实现。"""
from pathlib import Path
import sys

# 官方入口harness.py与项目包同名；优先固定源码根，避免把入口当成库导入。
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from harness.adapters.platforms.sdk_submission import TASKS, train, predict
