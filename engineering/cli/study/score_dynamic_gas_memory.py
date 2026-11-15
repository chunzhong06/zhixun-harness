"""从原始连续采样重建秒观测；同一批次提交后才能取既定的32个回标答案。"""
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score
from prepare_dynamic_mixtures import RAW,ROOT,DEST as DATA,EXPECTED
from harness.contracts import digest
from harness.runtime.storage import LocalStore,atomic_json
from harness.runtime.handoff import LocalAcceptance
from harness.models.dynamic_gas_readout import DynamicEncoder,StreamRegression,MotionChoice,PRIMARY,VIEWS

PRIOR = ROOT/"results/experiments/local-defaults/whole-maintenance/field-delivery"
DEST = PRIOR.parent/"dynamic-gas-memory"
MODES = ("只用源端","累计回标","近期回标")


def seconds(stream):
    counts,totals = {},{}
    cursor = -np.inf
    for f in pd.read_csv(stream,sep=r"\s+",skiprows=1,header=None,chunksize=120000,usecols=[0,*range(3,19)]):
        a = f.to_numpy(dtype=float)
        t = a[:,0]
        if t[0]<cursor or np.any(np.diff(t)<0) or not np.isfinite(a).all():raise ValueError("时钟或传感器无效")
        cursor=t[-1]
        keys = np.floor(t).astype(int)
        starts = np.r_[0,np.flatnonzero(np.diff(keys))+1]
        n = np.diff(np.r_[starts,len(t)])
        block = np.add.reduceat(a[:,1:],starts,axis=0)
        for sec,row,amt in zip(keys[starts],block,n,strict=True):
            totals[sec]=totals.get(sec,0)+row
            counts[sec]=counts.get(sec,0)+amt
    order = np.array(sorted(counts))
    return order,np.stack([totals[k]/counts[k] for k in order]),np.array([counts[k] for k in order])


def encode(encoder,raw,ids,prepared):
    times,signal,counts = raw
    windows = times//10
    buckets = np.r_[0,np.flatnonzero(np.diff(windows))+1,len(windows)]
    if len(ids)!=len(buckets)-1:raise ValueError("十秒服务边界不同")
    result = [[] for _ in VIEWS];motion=[]
    for uid,left,right in zip(ids,buckets[:-1],buckets[1:],strict=True):
        if int(uid.split("/")[-1])!=windows[left]:raise ValueError("时间窗口错位")
        actual = np.average(signal[left:right],weights=counts[left:right],axis=0)
        if not np.allclose(actual,prepared[uid],rtol=2e-10,atol=1e-8):raise ValueError("原始窗口与开发测量不一致")
        values,speed = encoder.encode(signal[left:right])
        for j,v in enumerate(values):result[j].append(v)
        motion.append(speed)
    return [np.stack(v) for v in result],np.array(motion)


def compare(site,source,raw,lists):
    src,dev = lists[source],lists[site]
    ref = json.loads((PRIOR/site/"ongoing/run.json").read_bytes())
    old_report = json.loads((PRIOR/site/"scores.json").read_bytes())
    encoder = DynamicEncoder(raw[source][1])
    ids = [row["id"] for row in src]
    training,sm = encode(encoder,raw[source],ids,{r["id"]:r["values"] for r in src})
    y = np.array([r["label"]==2 for r in src])
    models = [StreamRegression().fit(v,y) for v in training]
    select = MotionChoice(sm,len(models)*len(MODES))
    encoder.reset()
    keys = [r["id"] for r in dev]
    target,motion = encode(encoder,raw[site],keys,{r["id"]:r["values"] for r in dev})
    # 此映射仅在确认本批交付后按旧回标名单查询。编码器仅接收测量，不接收标签。
    answers = {r["id"]:int(r["label"]) for r in dev}
    receiver = LocalAcceptance();receiver.output = DEST/site/"accepted"
    emitted=[];cursor=0
    for e in ref["accepted"]:
        old = json.loads((PRIOR/site/"ongoing/accepted"/(e["request"]+".json")).read_bytes())["payload"]
        if digest([ref["identity"],old])!=e["request"]:raise ValueError("原始反馈时序身份改变")
        block = old["candidate_ids"]
        cut = len(block)
        if block!=keys[cursor:cursor+cut]:raise ValueError("服务时序错位")
        sl = slice(cursor,cursor+cut)
        if old["feedback_seen"]!=models[0].count:raise ValueError("反馈成本不同")
        cols = {k:list(v) for k,v in old["candidate_predictions"].items()}
        predictions = [m.outputs(v[sl]) for m,v in zip(models,target,strict=True)]
        bank = np.column_stack(predictions)
        names = [view+"/"+mode for view in VIEWS for mode in MODES]
        values = dict(zip(names,bank.T,strict=True));values.update(select.predict(motion[sl],bank))
        anchor = np.asarray(cols[old_report["strongest_fixed"]])==2
        for view in VIEWS:
            for mode in MODES:
                name = view+"/"+mode
                values["原固定与"+name+"各半"] = (.5+(.5*anchor-.25)+np.clip(values[name],0,1))/2
        cols.update({k:(1+(v>=.5)).tolist() for k,v in values.items()})
        payload={"ids":block,"预测":cols,"回标已用":models[0].count,"上轮":e["request"]}
        req=digest([site,payload])
        if not receiver.submit(req,payload)["accepted"]:raise ValueError("动态预测未确认")
        emitted.append(req)
        returned=e["returned"]
        if len(returned)!=len(set(returned)) or len(returned)>32 or not set(returned)<=set(block):raise ValueError("原回标分配无效")
        if returned:
            ix=np.array([block.index(i) for i in returned])
            yy=np.array([answers[i]==2 for i in returned])
            select.update(motion[sl][ix],bank[ix],yy)
            for m,v in zip(models,target,strict=True):m.update(v[sl][ix],yy)
        cursor+=cut
    if cursor!=len(keys) or models[0].count!=old_report["labels_used"]:raise ValueError("服务未覆盖或反馈数量变化")
    columns = {}
    seen=[]
    for req in emitted:
        p=json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest([site,p])!=req:raise ValueError("交付身份变化")
        seen.extend(p["ids"])
        for k,v in p["预测"].items():columns.setdefault(k,[]).extend(v)
    if seen!=keys:raise ValueError("重复或漏发窗口")
    true=np.array([answers[i] for i in keys])
    table={k:{"F1":float(f1_score(true,v,average="macro")),"最弱召回":float(recall_score(true,v,average=None).min())}
           for k,v in columns.items()}
    if not np.isclose(table[old_report["strongest_fixed"]]["F1"],old_report["scores"][old_report["strongest_fixed"]]["macro_f1"]):
        raise ValueError("先前完整固定分数未复现")
    fixed=max((k for k in table if k!=PRIMARY),key=lambda k:table[k]["F1"])
    paper=max((k for k in table if "论文" in k),key=lambda k:table[k]["F1"])
    gap=table[PRIMARY]["F1"]-max(table[k]["F1"] for k in (fixed,paper))
    d={"现场F1":table[PRIMARY]["F1"],"固定":fixed,"固定F1":table[fixed]["F1"],"论文":paper,"论文F1":table[paper]["F1"],"差值":gap,
       "通过":gap>=.01 and table[PRIMARY]["最弱召回"]>=table[fixed]["最弱召回"]-.02,"回标":models[0].count}
    return {"decision":d,"scores":table,"requests":emitted,"窗口":len(keys)}


def run():
    with RAW.open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=EXPECTED:raise ValueError("UCI322原始包不同")
    with ZipFile(RAW) as z:
        raw={}
        for site,member in (("co","ethylene_CO.txt"),("methane","ethylene_methane.txt")):
            with z.open(member) as f:raw[site]=seconds(f)
    lists={s:json.loads((DATA/s/"source.json").read_bytes()) for s in raw}
    results={s:compare(s,t,raw,lists) for s,t in (("co","methane"),("methane","co"))}
    atomic_json(DEST/"scores.json",results)
    print(json.dumps({site+"连续乙烯识别":x["decision"] for site,x in results.items()},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("该连续动态试验已完成")
        run()
