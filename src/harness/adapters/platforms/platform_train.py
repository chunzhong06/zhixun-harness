"""保留历史 python -m 训练命令，导入时与新入口共用同一模块。"""
import sys
from harness.contest import training

if __name__ == "__main__":
    training.main()
else:
    sys.modules[__name__] = training
