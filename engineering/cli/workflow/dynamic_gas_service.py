"""从已标定源流启动，接收原始十分钟测量或随后返回的标签，逐批发放乙烯预测。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.dynamic_gas_readout import DynamicGasService


def serve(source,site,calibrate_lag):
    with np.load(source,allow_pickle=False) as d:
        service = DynamicGasService().fit(d["seconds"],d["windows"],d["labels"],calibrate_lag)
    for line in sys.stdin:
        request = json.loads(line)
        if set(request)=={"batch"}:
            with np.load(request["batch"],allow_pickle=False) as b:
                if set(b.files)!={"clock","sensors"}:raise ValueError("原始服务采样只能包含传感器与时间戳")
                ids,out = service.predict(b["clock"],b["sensors"],site)
            reply = {"ids":ids,"predictions":out,"used":service.models[0].count}
        elif set(request)=={"feedback"}:
            service.update(request["feedback"])
            reply = {"used":service.models[0].count}
        else:
            raise ValueError("请求须为预测或回标")
        print(json.dumps(reply),flush=True)
    if service.pending is not None:raise ValueError("最后一次输出尚未确认反馈")


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--site",required=True)
    parser.add_argument("--calibrate-lag",action="store_true")
    args = parser.parse_args()
    serve(args.source,args.site,args.calibrate_lag)
