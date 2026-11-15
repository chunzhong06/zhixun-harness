"""以作者的多器件共享及每日洁净参照构建同标签对照，评价整批交付和回标价值。"""
from io import BytesIO
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from score_temperature_heater_response import RAW, INPUT, CHANNELS, DEST as FIRST, POLICIES, read, digest, LocalStore, LocalAcceptance, atomic_json
from verify_temperature_heater_entry import DEST as ENTRY
from harness.models.gas_global_calibration import heating_curve, readings, ordered_label

DEST = FIRST.parent/"gas-global-calibration"
PAPER = "论文多器件洁净校准"
ANCHOR = "相位电阻合训"
PRIMARY = "依已回标分歧选择洁净校准"


def extract():
    bank = {}
    with ZipFile(RAW) as outer, ZipFile(BytesIO(outer.read("gas-sensor-array-temperature-modulation.zip"))) as inner:
        for day in sorted(x for x in inner.namelist() if x.endswith(".csv")):
            with inner.open(day) as stream:
                frame = pd.read_csv(stream,usecols=[0,5,*range(6,20)]).sort_values("Time (s)",kind="stable")
            raw = frame.to_numpy()
            t = raw[:,0]-raw[0,0]
            bins = np.floor(t/900).astype(int)
            limit = int(np.floor((t[-1]+1/3.5)/900))
            day_base = None
            for n in range(limit):
                mask=bins==n
                r=raw[mask,2:]
                if not mask.any() or not np.all(np.isfinite(r).any(0)):
                    if n==0:
                        raise ValueError("未取得作者日初洁净参照")
                    continue
                curve=heating_curve(t[mask],raw[mask,1],r)
                if n==0:
                    day_base=curve
                else:
                    bank[f"{day[:-4]}/{n}"]=np.stack((curve,curve-day_base))
    return bank


def by_type(x,sy,target,components,family="pls"):
    groups=[slice(0,7),slice(7,14)] if x.shape[1]==14 else [slice(None)]
    preds=[]
    spread=[]
    for g in groups:
        expected, uncertainty=readings(x[:,g],sy,components,target[:,g],family)
        preds.append(expected)
        spread.append(uncertainty)
    return np.mean(preds,axis=0),np.mean(spread,axis=0)


def run_site(site,policy,view,bank):
    layout,sc,tc=CHANNELS[site]
    folder=INPUT/layout
    spec=read(folder/"stream.json")
    src=read(folder/spec["source"])
    x=np.stack([bank[r["id"]] for r in src])[:,:,sc]
    y=np.array([r["label"] for r in src])
    days=np.array([r["id"].split("/")[0] for r in src])
    # 每日100个测量整组验证，保证相同暴露的多个器件不会分在供选成分的两侧。
    choices=[]
    for n in (1,2,3,5,8):
        held=np.zeros(len(y))
        for d in np.unique(days):
            ix=days==d
            held[ix]=by_type(x[~ix,1],y[~ix],x[ix,1],n)[0]
        choices.append((f1_score(y,ordered_label(held),average="macro"),-n))
    components=-max(choices)[1]
    parent=ENTRY/site/policy/view
    previous=read(parent/"run.json")
    ident=digest([previous["identity"],components,"global-calibration-01"])
    out=DEST/site/policy/view
    receiver=LocalAcceptance()
    receiver.output=out/"accepted"
    shown=[]; labels=[]; events=[]; past=[]
    for b,e in zip(spec["batches"],previous["events"],strict=True):
        old=read(parent/"accepted"/(e["request"]+".json"))["payload"]
        if digest([previous["identity"],old])!=e["request"] or old["已见反馈"]!=len(labels):
            raise ValueError("温控发放和回标历史不同")
        ids=old["ids"]
        target=np.stack([bank[k] for k in ids])[:,:,tc]
        sx=np.concatenate([x,*shown]); sy=np.r_[y,np.asarray(labels,dtype=int)]
        raw,spread=by_type(sx[:,1],sy,target[:,1],components)
        fixed=np.asarray(old["预测"][ANCHOR])
        paper=ordered_label(raw)
        source_raw=by_type(x[:,1],y,target[:,1],components)[0]
        nonlinear=by_type(sx[:,1],sy,target[:,1],components,"svm")[0]
        uncorrected=by_type(sx[:,0],sy,target[:,0],components)[0]
        pred={**old["预测"],PAPER:paper,"论文校准仅源":ordered_label(source_raw),
              "共享模型无洁净参照":ordered_label(uncorrected),"多器件洁净核":ordered_label(nonlinear)}
        for share in (.25,.5,.75):
            pred[f"论文与相位固定{share}"]=ordered_label(share*raw+(1-share)*fixed)
        result=fixed.copy()
        if past:
            old_p,old_anchor,old_y,old_w,old_sp=map(np.concatenate,zip(*past))
            for a in (1,2,3):
                for d in (1,2,3):
                    if a==d:continue
                    historical=(old_p==a)&(old_anchor==d)
                    if historical.sum()<2:continue
                    acc=(old_y==a).astype(float)-(old_y==d)
                    local=(paper==a)&(fixed==d)
                    for j in np.flatnonzero(local):
                        w=old_w[historical]/(1+4*abs(old_sp[historical]-spread[j]))
                        if np.sum(w*acc[historical])/(w.sum()+4)>.15:
                            result[j]=paper[j]
        pred[PRIMARY]=result
        payload={"ids":ids,"预测":{k:np.asarray(v).tolist() for k,v in pred.items()},"已见反馈":len(labels),"原回执":e["request"]}
        key=digest([ident,payload])
        if not receiver.submit(key,payload)["accepted"]:
            raise ValueError("校准方案未发出")
        events.append(key)
        if e["selected"]:
            ix=[ids.index(k) for k in e["selected"]]; yy=np.array(e["labels"],dtype=int)
            returned=read(folder/b["feedback"])
            if yy.tolist()!=[returned[k] for k in e["selected"]]:
                raise ValueError("同预算查询标签错位")
            shown.append(target[ix]);labels.extend(yy.tolist())
            w=np.ones(len(ix))
            if policy=="混合抽查":
                half=len(ix)//2
                w[half:]=(len(ids)-half)/(len(ix)-half)
            past.append((paper[ix],fixed[ix],yy,w,spread[ix]))
    truth=read(folder/"evaluation.json"); ys=[];cols={}
    for key in events:
        p=read(receiver.output/(key+".json"))["payload"]
        if digest([ident,p])!=key:raise ValueError("发布身份改变")
        ys.extend(truth[k] for k in p["ids"])
        for k,v in p["预测"].items():cols.setdefault(k,[]).extend(v)
    scores={k:{"F1":float(f1_score(ys,v,average="macro")),"最差召回":float(recall_score(ys,v,average=None).min())} for k,v in cols.items()}
    atomic_json(out/"run.json",{"identity":ident,"components":components,"requests":events,"scores":scores})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("论文同条件校准已完成")
        bank=extract()
        configs=[("混合抽查","均值"),("BADGE","均值"),*[(q,"加热") for q in POLICIES]]
        streams={s:{q+"-"+view:run_site(s,q,view,bank) for q,view in configs} for s in CHANNELS}
        prior=read(ENTRY/"scores.json")
        means={a:{k:{m:float(np.mean([s[a][k][m] for s in streams.values()])) for m in ("F1","最差召回")} for k in item}
               for a,item in streams["全通道"].items()}
        get=lambda arm:means[arm[0]][arm[1]]["F1"]
        f=max(((a,k) for a in means if not a.startswith("BADGE") for k in means[a] if k!=PRIMARY),key=get)
        p=max(((a,k) for a in means for k in means[a] if k!=PRIMARY and (a.startswith("BADGE") or k.startswith("论文"))),key=get)
        a=("混合抽查-均值",PRIMARY)
        gap=get(a)-max(get(f),get(p),prior["decision"]["论文F1"],prior["decision"]["固定F1"])
        decision={"主候选":a,"现场F1":get(a),"固定":f,"固定F1":get(f),"同工具论文":p,"论文F1":get(p),"最小领先":gap,
                  "进独立":gap>=.01 and means[a[0]][PRIMARY]["最差召回"]>=means[f[0]][f[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":decision})
        print(json.dumps({"成熟校准与实际发放":decision},ensure_ascii=False))


if __name__=="__main__":main()
