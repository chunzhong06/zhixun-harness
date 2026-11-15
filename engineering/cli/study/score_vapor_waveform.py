"""把采样内波形直接用于逐半秒发放，保留相同原始输入、时钟与反馈的论文对照。"""
from hashlib import sha256
import json

import joblib
import numpy as np

from score_vapor_field import DATA, LocalAcceptance, LocalStore, atomic_json, digest, read_flight
from score_vapor_correction import DEST as PRIOR, clock_groups, deliver, load_verified, score, PRIMARY as CORRECTION
from harness.models.vapor_boundary import PRIMARY as MIXED
from harness.models.vapor_waveform import Waveform, WaveReader, PRIMARY, FIXED_PAPER

DEST=PRIOR.parent/"vapor-waveform"


def train(manifest,tools):
    features=[]; truth=[]
    for i in manifest["源日"]:
        raw,y,base=read_flight(i,manifest)
        ends=np.r_[0,np.cumsum([len(a) for a in raw])]
        clocks=clock_groups(DATA/manifest["记录"][str(i)]["文件"],ends)
        sensor=Waveform(base,tools["scale"])
        features.extend([sensor.next(a,ts) for a,ts in zip(raw,clocks,strict=True)][20:])
        truth.extend(y[20:]/tools["cut"])
    return WaveReader().fit(np.stack(features),np.array(truth))


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    tools,previous=load_verified(PRIOR)
    tools["wave"]=train(manifest,tools)
    release=DEST/"release.joblib"
    joblib.dump(tools,release,compress=3)
    checksum=sha256(release.read_bytes()).hexdigest()
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    requests={}; reports={}; all_y=[]; all_cols={};spent={}
    for i in manifest["开发日"]:
        seq,y,spent[i]=deliver(i,manifest,previous,release,checksum,receiver,clocked=True)
        rows=[];cols={}
        for key in seq:
            pkt=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(pkt)!=key:
                raise ValueError("原始采样预测在交付后改变")
            rows.extend(pkt["窗口"])
            for name,values in pkt["阈值倍数"].items():
                cols.setdefault(name,[]).extend(values)
        if rows!=list(range(20,len(y))):
            raise ValueError("逐半秒发放不完整")
        labels=y[rows]/tools["cut"]>=1
        requests[i]=seq;reports[i]=score(labels,cols);all_y.extend(labels)
        for name,values in cols.items():
            all_cols.setdefault(name,[]).extend(values)
    means=score(all_y,all_cols)
    fixed=max((k for k in means if k not in (PRIMARY,MIXED,CORRECTION)),key=lambda k:means[k]["F1"])
    paper=max((FIXED_PAPER,*(k for k in means if "与论文" in k)),key=lambda k:means[k]["F1"])
    gap=means[PRIMARY]["F1"]-max(means[k]["F1"] for k in (fixed,paper))
    d={"现场":PRIMARY,"固定":fixed,"论文":paper,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
       "论文F1":means[paper]["F1"],"差值":gap,
       "进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":reports,"requests":requests,
        "源权重SHA256":checksum,"窗口":len(all_y),"各场已回参考":spent})
    print(json.dumps({"原始采样快慢变化逐窗交付":d,"实际预测数":len(all_y),"总回标":sum(spent.values())},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("原始采样已有完整部署")
        run()
