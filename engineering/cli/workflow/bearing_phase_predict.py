"""从已交付的阶段树读取比赛单条双轴振动输入，返回剩余分钟。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

from harness.contracts import digest
from harness.data.official_bearing import CHANNELS, MEASURES
from harness.models.model_artifacts import ModelArtifacts
from harness.models.training_tools import SklearnBackend


def infer(folder, requests):
    report=json.loads((folder/"run.json").read_bytes())
    if report.get("task")!="bearing" or report.get("policy")!="phase-fixed" or not report["accepted"]:
        raise ValueError("需要已实际交付的轴承阶段树目录")
    entry=report["accepted"][-1]
    ack=json.loads((folder/"accepted"/(entry["request"]+".json")).read_bytes())
    p=ack["payload"]
    if ack["request_id"]!=entry["request"] or digest([report["identity"],p])!=entry["request"]:
        raise ValueError("服务回执身份错误")
    manifest,blob=ModelArtifacts(folder/"references").read(report["tree_artifact"])
    spec=dict(manifest["request"])
    scale=spec.get("scale")
    if (spec.get("policy")!="phase-fixed" or scale!=report["scale_minutes"] or digest([spec,manifest["sha256"]])!=p["bank"]):
        raise ValueError("训练树及尺度与实际交付不一致")
    # 模型仅来自本机已知可信的工作区运行产物。每行无需访问行号、寿命、设备或后续测量。
    model=SklearnBackend(execution_enabled=True).load(blob)
    for req in requests:
        if (not isinstance(req,dict) or set(req)-{"task","inputs"} or req.get("task","regression")!="regression" or
                not isinstance(req.get("inputs"),dict) or set(req["inputs"])!={"features"}):
            raise ValueError("需要轴承回归输入")
        f=req["inputs"]["features"]
        if not isinstance(f,dict) or set(f)!=set(CHANNELS) or any(not isinstance(f[c],dict) or set(f[c])!=set(MEASURES) for c in CHANNELS):
            raise ValueError("需要官方两方向各十七项特征")
        x=[f[c][n] for c in CHANNELS for n in MEASURES]
        if any(type(v) not in (int,float) for v in x) or not np.isfinite(x).all():
            raise ValueError("振动特征必须是有限数值")
        v=float(model.predict([x],thread_count=1)[0])
        yield {"outputs":{"value":float(np.clip(v,0,1)*scale)}}


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args(argv)
    for r in infer(args.run,(json.loads(line) for line in sys.stdin if line.strip())):
        print(json.dumps(r,allow_nan=False),flush=True)
