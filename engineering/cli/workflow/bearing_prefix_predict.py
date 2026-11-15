"""一个进程服务一台轴承。每行送入200个频带与上次测量以来至本行的实际采集时间戳。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys
import joblib

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))


def predict(stream, record):
    obs=record["inputs"]
    if set(obs)!={"spectrum","timestamps"}:
        raise ValueError("仅接收实测频谱及本次时间块")
    remaining=stream.predict(obs["spectrum"],obs["timestamps"])
    return {"outputs":{"remaining_calendar_minutes":remaining}}


if __name__=="__main__":
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("--artifact",type=Path,required=True)
    ap.add_argument("--sha256",required=True)
    args=ap.parse_args()
    with args.artifact.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=args.sha256:raise ValueError("装机权重身份不同")
        f.seek(0)
        stream=joblib.load(f).stream()
    for line in sys.stdin:
        if line.strip():print(json.dumps(predict(stream,json.loads(line)),allow_nan=False),flush=True)
