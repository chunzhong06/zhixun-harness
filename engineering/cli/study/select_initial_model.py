"""核对已有首批筛选结果并固定初始配方；不调用训练或预测。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.evaluation.initial_selection import freeze_initial

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=freeze_initial(args.study,args.output)
    print({"status":result["status"],"selected":result["selected"]})
