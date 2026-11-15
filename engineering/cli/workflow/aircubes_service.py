"""独立进程接收一台空气探头的原始区间。回执确认后才允许补交该六小时槽的末窗参考。"""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from harness.models.aircubes import HumidityHistory
from harness.models.aircubes_pair import PairedMaintenance


def joint_controller(model):
    """按实际装机的训练工具恢复服务，兼容此前发布过的权重包。"""
    if "history" in model:
        from harness.models.aircubes_history import HistoryMaintenance
        return HistoryMaintenance(model)
    if "precision" in model:
        from harness.models.aircubes_precision import PrecisionMaintenance
        return PrecisionMaintenance(model)
    if "changes" in model:
        from harness.models.aircubes_changes import ChangeMaintenance
        return ChangeMaintenance(model)
    if "response" in model:
        from harness.models.aircubes_response import ResponseMaintenance
        return ResponseMaintenance(model)
    if "stack" in model:
        from harness.models.aircubes_stack import StackMaintenance
        return StackMaintenance(model)
    from harness.models.aircubes_bias import DriftMaintenance
    return DriftMaintenance(model)


def serve(path,checksum,rules=()):
    with path.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=checksum:raise ValueError("源校准权重不符")
        f.seek(0);model=joblib.load(f)
    hist=HumidityHistory()
    joint=isinstance(model,dict) and "joint" in model
    if joint:
        state=joint_controller(model)
        paper=model["paper"]
    else:
        state=PairedMaintenance()
        paper=model
    count=0
    last=None; pending=None
    for line in sys.stdin:
        msg=json.loads(line)
        if set(msg)=={"起始时刻","启动湿度"} and last is None:
            raw=np.asarray(msg["启动湿度"],dtype=float)
            if raw.shape!=(52,):raise ValueError("需52个连续的已知历史湿度以重建500分钟变化")
            for r in raw:hist.next(r)
            last=int(msg["起始时刻"])-600
            response={"下个时刻":last+600}
        elif set(msg)=={"时刻","读数"} and last is not None and pending is None:
            t=np.asarray(msg["时刻"],dtype=np.int64);x=np.asarray(msg["读数"],dtype=float)
            if (t.ndim!=1 or not len(t) or len(t)>36 or x.shape!=(len(t),7 if joint else 4) or t[0]!=last+600 or
                    np.any(np.diff(t)!=600) or t[0]//21600!=t[-1]//21600):
                raise ValueError("每批只接受同一六小时槽的连续原始测量")
            # 湿度历史照时钟递进，即便本时刻电极缺失。每行只依赖自己与过去。
            columns=np.column_stack((x[:,:4],[hist.next(h) for h in x[:,3]]))
            valid=np.isfinite(columns).all(1)
            if joint and not np.isfinite(x[valid,4:]).all():
                raise ValueError("可用原电极时刻的辅助读数缺失")
            issued=t[valid].tolist();out={}
            if len(issued):
                v=columns[valid]
                baseline=paper.predict(v)
                p=state.predict(t[valid],np.column_stack((v,x[valid,4:])),baseline) if joint else state.predict(baseline,v[:,2])
                out={n:p[n].tolist() for n in (rules or p)}
            last=int(t[-1]);pending=issued[-1] if issued else -1
            response={"区间起点":issued,"预测":out,"已取参考":count}
        elif set(msg)=={"确认槽末","参考"} and pending is not None and msg["确认槽末"]==last:
            if msg["参考"] is not None:
                if pending!=last or (last+600)%21600 or not np.isfinite(msg["参考"]):
                    raise ValueError("此时不能返回指定时刻的参考")
                state.update(float(msg["参考"]))
                count+=1
            pending=None
            response={"已取参考":count}
        else:
            raise ValueError("原始服务请求与上一批状态不符")
        print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)
    if pending is not None:raise ValueError("最后发出的批次未确认")


if __name__=="__main__":
    arg=ArgumentParser(description=__doc__);arg.add_argument("--release",type=Path,required=True)
    arg.add_argument("--sha256",required=True)
    arg.add_argument("--rule",action="append",default=[],help="仅输出指定读法，可重复；默认发放全部")
    args=arg.parse_args()
    serve(args.release,args.sha256,args.rule)
