"""从四列原始电极及环境输入重新部署四台仪器。用真实回执和作者参考检查最终输出。"""
import json
import subprocess
import sys
import numpy as np
import pandas as pd

from score_aircubes import DATA,ROOT,START,known,LocalAcceptance,LocalStore,atomic_json,digest,RAW
from score_aircubes_intervals import DEST as PRIOR,PRIMARY

DEST=PRIOR.parent/"aircubes-entry"
CLI=ROOT/"engineering/cli/workflow/aircubes_service.py"


def values(array):
    a=np.asarray(array)
    return np.where(np.isfinite(a),a.astype(object),None).tolist()


def ask(proc,req):
    proc.stdin.write(json.dumps(req,ensure_ascii=False,allow_nan=False)+"\n")
    proc.stdin.flush()
    line=proc.stdout.readline()
    if not line:raise ValueError("原始空气监测入口没有回答")
    return json.loads(line)


def run_site(site,manifest,previous,ref,receiver,prior_dir=PRIOR,fields=RAW,rules=()):
    frame=pd.read_csv(known(site+".csv",manifest),usecols=["timestamp",*fields])
    t=frame["timestamp"].to_numpy(dtype=np.int64)
    raw=frame[list(fields)].to_numpy()
    ix=int(np.searchsorted(t,START))
    clock=t[ix:];x=raw[ix:]
    bounds=np.r_[0,np.flatnonzero(np.diff(clock//21600))+1,len(clock)]
    cmd=[sys.executable,"-B",str(CLI),"--release",str(prior_dir/(site+".joblib")),
         "--sha256",previous["源模型SHA256"][site]]
    for rule in rules:cmd.extend(["--rule",rule])
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8")
    events=[];count=0;prior=iter(previous["requests"][site]);max_diff=0
    try:
        if ask(proc,{"起始时刻":int(clock[0]),"启动湿度":values(raw[ix-52:ix,3])})!={"下个时刻":int(clock[0])}:
            raise ValueError("已有湿度未正确启动")
        for a,b in zip(bounds[:-1],bounds[1:],strict=True):
            resp=ask(proc,{"时刻":clock[a:b].tolist(),"读数":values(x[a:b])})
            ids=resp["区间起点"]
            if resp["已取参考"]!=count:raise ValueError("新入口回标时序发生变化")
            if ids:
                before=next(prior);old=json.loads((prior_dir/"accepted"/(before+".json")).read_bytes())["payload"]
                if digest(old)!=before or ids!=old["区间起点"] or set(rules or old["预测"])!=set(resp["预测"]):
                    raise ValueError("已发测量与先前评分身份不一致")
                for n,p in resp["预测"].items():
                    diff=float(np.max(abs(np.array(p)-old["预测"][n])))
                    max_diff=max(max_diff,diff)
                    if diff>1e-7:raise ValueError("原始服务与旧发放数值不符")
            else:
                before=None
            new={"仪器":site,"区间起点":ids,"预测":resp["预测"],"已取参考":count,"前序":before}
            h=digest(new)
            if not receiver.submit(h,new)["accepted"]:raise ValueError("原始服务输出未接收")
            events.append(h)
            last=int(clock[b-1]);label=None
            if b<len(clock) and ids and ids[-1]==last and (last+600)%21600==0:
                y=ref.get(last+600,np.nan)
                if np.isfinite(y):label=float(y);count+=1
            if ask(proc,{"确认槽末":last,"参考":label})!={"已取参考":count}:
                raise ValueError("已确认反馈未进入下一批")
        if next(prior,None) is not None:raise ValueError("旧回执仍有未发项")
        proc.stdin.close()
        if proc.wait(timeout=20):raise ValueError("原始进程未正常结束")
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        if not proc.stdin.closed:proc.stdin.close()
        proc.stdout.close()
    sq={};total=0;issued=0
    for h in events:
        p=json.loads((receiver.output/(h+".json")).read_bytes())["payload"]
        if digest(p)!=h:raise ValueError("原始部署回执更改")
        ids=p["区间起点"];issued+=len(ids)
        truth=ref.reindex(np.asarray(ids,dtype=np.int64)+600).to_numpy();ok=np.isfinite(truth)
        total+=int(ok.sum())
        for n,v in p["预测"].items():
            sq[n]=sq.get(n,0.)+np.sum((np.array(v)[ok]-truth[ok])**2)
    return {"requests":events,"实发":issued,"可评分":total,"已回标":count,"最大复现差值":max_diff,
            "scores":{n:float(np.sqrt(v/total)) for n,v in sq.items()}}


def run():
    meta=json.loads((DATA/"source.json").read_bytes())
    prev=json.loads((PRIOR/"scores.json").read_bytes())
    ref=pd.read_csv(known("reference-first.csv",meta),usecols=["timestamp","NO2_ppb"]).set_index("timestamp")["NO2_ppb"]
    receiver=LocalAcceptance();receiver.output=DEST/"accepted"
    reports={s:run_site(s,meta,prev,ref,receiver) for s in prev["requests"]}
    means={n:float(np.mean([v["scores"][n] for v in reports.values()])) for n in prev["means"]}
    for n,v in means.items():
        if abs(v-prev["means"][n])>1e-7:raise ValueError("两种入口的实际分数不同")
    d=prev["decision"]
    summary={"实发":sum(r["实发"] for r in reports.values()),"可评分":sum(r["可评分"] for r in reports.values()),
        "已回标":sum(r["已回标"] for r in reports.values()),"现场误差":means[PRIMARY],
        "固定误差":means[d["固定"]],"论文误差":means[d["论文"]],"复现最大差":max(r["最大复现差值"] for r in reports.values())}
    atomic_json(DEST/"scores.json",{"summary":summary,"sites":reports,"means":means,"源模型SHA256":prev["源模型SHA256"]})
    print(json.dumps({"原始电极入口实发":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("原始发放已完成")
        run()
