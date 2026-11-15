"""从 NASA 原始循环号、FEMTO 波形时间戳重新计分，不使用模拟器的标签对象。"""
import json
from collections import defaultdict
from math import fsum, isclose, sqrt
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/"engineering/src"))
from harness.runtime.storage import atomic_json

BASE=ROOT/"results/experiments/unseen-agents"


def load(path):
    return json.loads(path.read_bytes())


def remaining_minutes(manifest="manifest.json"):
    source=ROOT/"data/public/pronostia-sampled"
    times=defaultdict(dict)
    for rec in load(source/manifest):
        with (source/rec["source"]).open("rb") as fp:
            first=fp.readline().replace(b";",b",").split(b",")
        hh,mm,ss,us=map(float,first[:4])
        times[rec["unit"]][rec["index"]]=3600*hh+60*mm+ss+us/1_000_000
    truth={}
    for uid,by_index in times.items():
        end=max(by_index)
        for i,time in by_index.items():
            seconds=(by_index[end]-time)%86400
            if abs(seconds-10*(end-i))>0.05:
                raise ValueError("波形时钟间隔不符")
            if (i-1)%30==0:truth[f"PHM12/{uid}/{i}"]=seconds/60
    return truth


def remaining_cycles():
    ends={}
    with (ROOT/"data/public/phm08/train.txt").open() as stream:
        for line in stream:
            uid,cycle,*_=line.split()
            ends[uid]=max(ends.get(uid,0),int(cycle))
    return {f"PHM08/{uid}/{j}":end-j for uid,end in ends.items() for j in range(1,end+1)}


def verify(task,truth):
    groups=load(BASE/"primary-scores"/(task+".json"))
    groups.update(load(BASE/"common-scores"/(task+".json")))
    expected=None; scores={}
    for name,result in groups.items():
        issued={}
        for n in range(1,11):
            part=load(BASE/"issued"/task/name/f"{n:02}.json")["predictions"]
            if issued.keys() & part.keys():raise ValueError("服务测点重复")
            issued.update(part)
        if expected is None:expected=set(issued)
        if set(issued)!=expected:raise ValueError("各组服务测点不齐")
        residual=[v-truth[k] for k,v in issued.items()]
        score=sqrt(fsum(x*x for x in residual)/len(residual))
        if not isclose(score,result["rmse"],abs_tol=0.001):
            raise ValueError(f"{task} {name} 的分数未能复算")
        scores[name]={"rmse_from_source":score,"mean_signed_error":fsum(residual)/len(residual)}
    y=[truth[k] for k in expected]
    return {"rows":len(y),"target_min":min(y),"target_max":max(y),
            "always_zero_rmse":sqrt(fsum(v*v for v in y)/len(y)),"policies":scores}


if __name__=="__main__":
    atomic_json(BASE/"原始来源复算.json",{"bearing":verify("bearing",remaining_minutes()),"engine":verify("engine",remaining_cycles())})
