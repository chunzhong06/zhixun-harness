"""把原始连续读数按已冻结的十秒窗口聚合，推理输入、交付后反馈与评分分别写出。"""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import hashlib

import numpy as np
import pandas as pd

from harness.runtime.storage import atomic_json


ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/"data/public/uci-dynamic-mixtures/uci322.zip"
DEST=ROOT/"data/prepared/gas/dynamic-mixtures"
EXPECTED="8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7"


def aggregate(stream):
    totals, positive, samples = {}, {}, {}
    previous_time=-np.inf
    for frame in pd.read_csv(stream,sep=r"\s+",skiprows=1,header=None,chunksize=100000):
        values=frame.to_numpy(dtype=float)
        if values.shape[1]!=19 or not np.isfinite(values).all():
            raise ValueError("原始读数缺失或传感器数量不同")
        clock=values[:,0]
        if clock[0]<previous_time or np.any(np.diff(clock)<0):
            raise ValueError("原始时钟回退")
        previous_time=clock[-1]
        windows=np.floor(clock/10).astype(int)
        indices=np.r_[0,np.flatnonzero(np.diff(windows))+1]
        counts=np.diff(np.r_[indices,len(clock)])
        sensors=np.add.reduceat(values[:,3:],indices,axis=0)
        ethylene=np.add.reduceat((values[:,2]>0).astype(int),indices)
        for w,n,x,y in zip(windows[indices],counts,sensors,ethylene,strict=True):
            k=int(w)
            totals[k]=totals.get(k,0)+x
            positive[k]=positive.get(k,0)+int(y)
            samples[k]=samples.get(k,0)+int(n)
    return [(i,(totals[i]/samples[i]).tolist(), 2 if 2*positive[i]>samples[i] else 1,
             samples[i],positive[i]) for i in sorted(totals)]


def prepare():
    with RAW.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=EXPECTED:
            raise ValueError("UCI 322 来源内容变化")
    manifest={}
    with ZipFile(RAW) as z:
        for name,filename in (("co","ethylene_CO.txt"),("methane","ethylene_methane.txt")):
            with z.open(filename) as f:
                records=aggregate(f)
            source=[{"id":f"{name}/{i}","values":x,"label":label} for i,x,label,_,_ in records]
            atomic_json(DEST/name/"source.json",source)
            truth={r["id"]:r["label"] for r in source}
            atomic_json(DEST/name/"evaluation.json",truth)
            groups=sorted(set(i//60 for i,*_ in records))
            names=[]
            for g in groups:
                rows=[r for r,(i,*_) in zip(source,records,strict=True) if i//60==g]
                stem=f"batch-{g:03}"
                atomic_json(DEST/name/(stem+".json"), [{"id":r["id"],"values":r["values"]} for r in rows])
                atomic_json(DEST/name/(stem+"-feedback.json"), {r["id"]:r["label"] for r in rows})
                names.append(stem)
            manifest[name]={"batches":names,"windows":len(records),"raw_samples":sum(r[3] for r in records),
                            "mixed_windows":sum(0<r[4]<r[3] for r in records),
                            "source_sha256":sha256((DEST/name/"source.json").read_bytes()).hexdigest()}
    for target,source in (("co","methane"),("methane","co")):
        folder=DEST/target
        items=[]
        for stem in manifest[target]["batches"]:
            inp=folder/(stem+".json")
            fb=folder/(stem+"-feedback.json")
            items.append({"batch":stem,"inputs":inp.name,"feedback":fb.name,"inputs_sha256":sha256(inp.read_bytes()).hexdigest(),
                          "feedback_sha256":sha256(fb.read_bytes()).hexdigest()})
        atomic_json(folder/"stream.json",{"classes":2,"source":f"../{source}/source.json",
                    "source_sha256":manifest[source]["source_sha256"],"batches":items})
    atomic_json(DEST/"preparation.json",{"raw_sha256":EXPECTED,"streams":manifest})
    for k,v in manifest.items():
        print(k,len(v["batches"]),v["windows"],v["raw_samples"],v["mixed_windows"],flush=True)


if __name__=="__main__":
    prepare()
