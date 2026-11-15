"""逐槽接受本机风速、转速和温度。功率参考只能在对应预测已确认之后归还。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.models.wind_power import WindMaintenance
from harness.models.wind_torque import CombinedWindRelease, TorqueMaintenance
from harness.models.wind_regime import RegimeRelease, RegimeMaintenance
from harness.models.wind_rotor import RotorRelease, RotorMaintenance


def serve(path, checksum, rules):
    with path.open("rb") as release:
        if file_digest(release,"sha256").hexdigest()!=checksum:
            raise ValueError("装机源权重摘要变化")
        release.seek(0)
        model = joblib.load(release)
    if hasattr(model,"controller"):
        state = model.controller()
    elif isinstance(model,RotorRelease):
        state = RotorMaintenance()
    elif isinstance(model,RegimeRelease):
        state = RegimeMaintenance()
    else:
        state = TorqueMaintenance() if isinstance(model,CombinedWindRelease) else WindMaintenance()
    last = None
    pending = False
    count = 0
    for line in sys.stdin:
        req = json.loads(line)
        if set(req)=={"时刻","测量"} and not pending:
            t = np.asarray(req["时刻"])
            x = np.asarray(req["测量"],dtype=float)
            if (t.ndim!=1 or t.dtype.kind not in "iu" or not 0<len(t)<=36 or x.shape!=(len(t),3)
                    or np.any(np.diff(t)<=0) or (last is not None and t[0]//21600<=last//21600)
                    or t[0]//21600!=t[-1]//21600 or not np.isfinite(x).all()):
                raise ValueError("测量必须来自同一六小时内且严格晚于已发批次")
            base = model.predict(x)
            predictions = state.predict_timed(t,x,base) if hasattr(state,"predict_timed") else state.predict(x,base)
            out = {"时刻":t.tolist(),"预测":{n:predictions[n].tolist() for n in rules or predictions},"已回标":count}
            last = int(t[-1])
            pending = True
        elif set(req)=={"已确认末窗","参考"} and pending and req["已确认末窗"]==last:
            reference = req["参考"]
            if reference is not None:
                if (last+600)%21600 or not np.isfinite(reference):
                    raise ValueError("该时刻不能回传功率参考")
                state.update(float(reference))
                count += 1
            out = {"已回标":count}
            pending = False
        else:
            raise ValueError("预测和确认的先后顺序错误")
        print(json.dumps(out,ensure_ascii=False,allow_nan=False),flush=True)
    if pending:
        raise ValueError("最后一批仍未确认")


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--release",type=Path,required=True)
    cli.add_argument("--sha256",required=True)
    cli.add_argument("--rule",action="append",default=[])
    args = cli.parse_args()
    serve(args.release,args.sha256,args.rule)
