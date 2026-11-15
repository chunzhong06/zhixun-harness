"""现场工况入口：从已完成的同型设备寿命拟合厂家关系，对当前每行输入给出剩余分钟。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from harness.models.bearing_load_life import BearingLoadLife


def infer(history, requests):
    model=BearingLoadLife(history)
    for request in requests:
        if not isinstance(request,dict) or set(request)-{"task","inputs"} or request.get("task","regression")!="regression":
            raise ValueError("工况预测只接收回归任务和当前输入")
        yield {"outputs":{"value":model.predict(request.get("inputs"))}}


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,required=True,help="同型设备已完成档案的 JSON 数组")
    args=parser.parse_args(argv)
    requests=(json.loads(line) for line in sys.stdin if line.strip())
    for result in infer(json.loads(args.history.read_bytes()),requests):
        print(json.dumps(result,allow_nan=False),flush=True)
