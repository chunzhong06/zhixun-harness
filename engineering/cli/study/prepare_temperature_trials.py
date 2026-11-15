"""从日期文件构建互不相交的整段暴露，每条输入只含本段传感器平均读数。"""
from hashlib import file_digest, sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from prepare_dynamic_mixtures import ROOT, atomic_json


RAW = ROOT / "data/public/uci-temperature-modulation/uci487.zip"
DEST = ROOT / "data/prepared/gas/temperature-trials"
IDENTITY = "cb5ce4a6af1a51b933d1979952d7845f0e9baac54e15a81a1e3599e8b85905d4"


def trials(inner, filename):
    with inner.open(filename) as stream:
        frame = pd.read_csv(stream, usecols=[0,1,*range(6,20)])
    clock = frame.iloc[:,0].to_numpy()
    if not np.isfinite(clock).all():
        raise ValueError("采集时钟缺失")
    # 十五分钟时间归属以发布者时间戳为准。源端第三天有一次约七秒的文件行序倒退。
    frame = frame.sort_values(frame.columns[0],kind="stable")
    clock = frame.iloc[:,0].to_numpy()
    time = clock-clock[0]
    number = np.floor(time/900).astype(int)
    limit = int(np.floor((time[-1]+1/3.5)/900))
    rows, rejected = [], []
    for n in range(1, limit):
        group = frame.loc[number==n].iloc[:,1:].replace([np.inf,-np.inf],np.nan)
        key = f"{filename[:-4]}/{n}"
        if group.empty or group.notna().sum().min()==0:
            rejected.append(key)
            continue
        concentration = group.iloc[:,0].median()
        rows.append({"id":key,"values":group.iloc[:,1:].mean().tolist(),
                     "label":1+int(concentration>5)+int(concentration>15)})
    return rows,rejected


def prepare():
    with RAW.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=IDENTITY:
            raise ValueError("原始归档摘要不同")
    all_rows={}
    rejected=[]
    with ZipFile(RAW) as archive, ZipFile(BytesIO(archive.read("gas-sensor-array-temperature-modulation.zip"))) as inner:
        for name in sorted(f for f in inner.namelist() if f.endswith(".csv")):
            date=name.split("_")[0]
            if date in all_rows:
                raise ValueError("一天有多个文件，需检查冻结日期划分")
            all_rows[date], omitted=trials(inner,name)
            rejected.extend(omitted)
    dates=list(all_rows)
    if len(dates)!=13:
        raise ValueError("未取得十三个日期")
    info={}
    layouts={"full":(None,None),"replacement":(slice(0,3),slice(3,6)),
             "a-to-a":(slice(0,3),slice(0,3)),"b-to-a":(slice(3,6),slice(0,3)),
             "b-to-b":(slice(3,6),slice(3,6))}
    for site,(source_channels,target_channels) in layouts.items():
        loc=DEST/site
        def project(row, source):
            channels=source_channels if source else target_channels
            if channels is None:
                return row
            return {**row,"values":row["values"][channels]}
        train=[project(r,True) for d in dates[:3] for r in all_rows[d]]
        atomic_json(loc/"source.json",train)
        batches=[]
        truth={}
        for d in dates[3:]:
            rows=[project(r,False) for r in all_rows[d]]
            inputs,feedback=loc/(d+".json"),loc/(d+"-feedback.json")
            atomic_json(inputs,[{"id":r["id"],"values":r["values"]} for r in rows])
            labels={r["id"]:r["label"] for r in rows}
            truth.update(labels)
            atomic_json(feedback,labels)
            batches.append({"batch":d,"inputs":inputs.name,"feedback":feedback.name,
                            "inputs_sha256":sha256(inputs.read_bytes()).hexdigest(),
                            "feedback_sha256":sha256(feedback.read_bytes()).hexdigest()})
        atomic_json(loc/"evaluation.json",truth)
        atomic_json(loc/"stream.json",{"classes":3,"source":"source.json",
                    "source_sha256":sha256((loc/"source.json").read_bytes()).hexdigest(),"batches":batches})
        info[site]={"source":len(train),"service":len(truth),"source_dates":dates[:3],"test_dates":dates[3:]}
    atomic_json(DEST/"preparation.json",{"raw_sha256":IDENTITY,"excluded_windows":rejected,"splits":info})
    print(info["full"],"omitted",len(rejected))


if __name__=="__main__":
    prepare()
