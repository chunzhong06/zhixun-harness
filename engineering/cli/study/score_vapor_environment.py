"""实发同层原始温湿度及三个慢通道，保留仅慢通道的已验收对照。"""
from hashlib import sha256
import json
import joblib
import numpy as np
import pandas as pd

from score_vapor_correction import DATA,DEST as PRIOR,LocalAcceptance,LocalStore,atomic_json,digest,load_verified,read_flight,deliver,score
from harness.models.vapor_environment import CHANNELS,EnvironmentalProbe,EnvironmentReadout,PRIMARY
from harness.models.vapor_waveform import FIXED_PAPER
from harness.models.vapor_boundary import PRIMARY as OLD_MIX
from harness.models.vapor_correction import PRIMARY as OLD_CORRECTION

DEST=PRIOR.parent/"vapor-environment"


def measurements(number,manifest,channels=CHANNELS):
    raw,y,_=read_flight(number,manifest)
    data=pd.read_csv(DATA/manifest["记录"][str(number)]["文件"],usecols=channels)[list(channels)].to_numpy()
    bounds=np.r_[0,np.cumsum([len(a) for a in raw])]
    if len(data)!=bounds[-1]:
        raise ValueError("环境传感器与慢读数采样数不匹配")
    return raw,y,[data[a:b] for a,b in zip(bounds[:-1],bounds[1:],strict=True)]


def run():
    manifest=json.loads((DATA/"source.json").read_bytes())
    tools,previous=load_verified(PRIOR)
    records=[]
    for number in manifest["源日"]:
        raw,y,env=measurements(number,manifest)
        encoder=EnvironmentalProbe(raw[:20],env[:20],tools["scale"])
        full=[encoder.next(v,e) for v,e in zip(raw,env,strict=True)]
        records.append(([np.stack(col) for col in zip(*full,strict=True)],y/tools["cut"]))
    tools["environment"]=EnvironmentReadout().fit(records)
    release=DEST/"release.joblib"
    joblib.dump(tools,release,compress=3)
    checksum=sha256(release.read_bytes()).hexdigest()
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    requests={};all_y=[];cols={};reports={};spent={}
    for number in manifest["开发日"]:
        _,_,env=measurements(number,manifest)
        keys,y,spent[number]=deliver(number,manifest,previous,release,checksum,receiver,environment=env)
        names={};rows=[]
        for key in keys:
            pkt=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest(pkt)!=key:raise ValueError("温湿校正交付身份不符")
            rows.extend(pkt["窗口"])
            for n,v in pkt["阈值倍数"].items():names.setdefault(n,[]).extend(v)
        if rows!=list(range(20,len(y))):raise ValueError("环境版本缺少半秒测量")
        label=y[rows]/tools["cut"]>=1
        requests[number]=keys;reports[number]=score(label,names)
        all_y.extend(label)
        for n,v in names.items():cols.setdefault(n,[]).extend(v)
    means=score(all_y,cols)
    # 全部新读法都接回同源储备池，固定全库与同工具论文在这里可以同名。
    fixed=max((n for n in means if n not in (PRIMARY,OLD_CORRECTION,OLD_MIX)),key=lambda n:means[n]["F1"])
    gap=means[PRIMARY]["F1"]-means[fixed]["F1"]
    d={"现场":PRIMARY,"固定":fixed,"论文":fixed,"现场F1":means[PRIMARY]["F1"],"固定F1":means[fixed]["F1"],
       "论文F1":means[fixed]["F1"],"差值":gap,"最弱召回":means[PRIMARY]["最弱召回"],
       "进入独立":gap>=.01 and means[PRIMARY]["最弱召回"]>=means[fixed]["最弱召回"]-.02}
    atomic_json(DEST/"scores.json",{"decision":d,"means":means,"scores":reports,"requests":requests,
        "源权重SHA256":checksum,"窗口":len(all_y),"各场已回参考":spent})
    print(json.dumps({"温湿实测校正":d,"原三通道固定F1":means[FIXED_PAPER]["F1"],"总窗数":len(all_y)},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("该版已有实发结果")
        run()
