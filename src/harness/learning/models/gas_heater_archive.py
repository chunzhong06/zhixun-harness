"""按作者的采集时间与冻结出的暴露边界读取单日测量。只打开温控和传感器列。"""
import numpy as np
import pandas as pd
from harness.learning.models.gas_heater_response import cycle_summary


def day_exposures(inner, name):
    result = {}
    # 不读取真实CO、湿度、流量，保留与旧入口相同的原始14路电阻及可实测加热电压。
    with inner.open(name) as handle:
        frame = pd.read_csv(handle,usecols=[0,5,*range(6,20)]).sort_values("Time (s)",kind="stable")
    raw = frame.replace([np.inf,-np.inf],np.nan).to_numpy()
    t = raw[:,0]-raw[0,0]
    group = np.floor(t/900).astype(int)
    limit = int(np.floor((t[-1]+1/3.5)/900))
    for n in range(1,limit):
        select = group==n
        if not select.any() or not np.all(np.isfinite(raw[select,2:]).any(0)):
            continue
        result[f"{name[:-4]}/{n}"] = cycle_summary(t[select],raw[select,1],raw[select,2:])
    return result
