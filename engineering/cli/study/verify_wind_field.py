"""从记录中的三列本机输入重启功率服务，对齐实际回标与逐条实发功率。"""
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

from score_wind_field import DEST as PRIOR,DATA,ROOT,START,FIELDS,LocalAcceptance,LocalStore,atomic_json,digest
from prepare_wind_field import FILES

DEST = PRIOR.parent/"wind-field-entry"


def verify(parent, destination, additional=()):
    source = json.loads((parent/"scores.json").read_bytes())
    path = DATA/"kelmarsh.csv.gz"
    with path.open("rb") as f:
        if file_digest(f,"md5").hexdigest()!=FILES[path.name]:
            raise ValueError("原始输入不符")
    frame = pd.read_csv(path,usecols=["turbine_id","timestamp","power_pu",*FIELDS])
    frame["clock"] = pd.to_datetime(frame.timestamp,utc=True).to_numpy(dtype="datetime64[s]").astype("int64")
    accept = LocalAcceptance()
    accept.output = destination/"accepted"
    d = source["decision"]
    rules = tuple(dict.fromkeys((d["现场"],d["固定"],d["论文"],"同样工具按全机累计误差分配","三变量提升树源训练",*additional)))
    states = {}
    maxdiff = 0.
    for site, expected in source["sites"].items():
        current = frame[(frame.turbine_id==site)&(frame.clock>=START)].sort_values("clock")
        clock = current.clock.to_numpy()
        raw = current[FIELDS].to_numpy()
        y = current.power_pu.to_numpy()
        edges = np.r_[0,np.flatnonzero(np.diff(clock//21600))+1,len(clock)]
        args = [sys.executable,"-B",str(ROOT/"engineering/cli/workflow/wind_power_service.py"),
                "--release",str(parent/"kelmarsh.joblib"),"--sha256",source["权重SHA256"]]
        for rule in rules:
            args.extend(["--rule",rule])
        proc = subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8")
        events = []
        count = 0
        errs = dict.fromkeys(rules,0.)
        def ask(message):
            proc.stdin.write(json.dumps(message,ensure_ascii=False,allow_nan=False)+"\n")
            proc.stdin.flush()
            return json.loads(proc.stdout.readline())
        try:
            for a,b,previous in zip(edges[:-1], edges[1:],expected["回执"],strict=True):
                old = json.loads((parent/"accepted"/(previous+".json")).read_bytes())["payload"]
                out = ask({"时刻":clock[a:b].tolist(),"测量":raw[a:b].tolist()})
                if (digest(old)!=previous or old["时刻"]!=out["时刻"] or out["时刻"]!=clock[a:b].tolist()
                        or out["已回标"]!=count or old["已回标"]!=count or set(out["预测"])!=set(rules)):
                    raise ValueError("原始部署的时刻、参考预算或已交付策略改变")
                key = digest([site,previous,out])
                if not accept.submit(key,out)["accepted"]:
                    raise ValueError("未完成实际交接")
                events.append(key)
                for n in rules:
                    actual = np.array(out["预测"][n])
                    diff = float(np.max(abs(actual-old["预测"][n])))
                    if diff>1e-7:
                        raise ValueError(f"{n}的实发功率不符")
                    maxdiff = max(maxdiff,diff)
                    errs[n] += float(np.sum((actual-y[a:b])**2))
                last = int(clock[b-1])
                ref = None
                if b<len(clock) and (last+600)%21600==0:
                    ref = float(y[b-1])
                    count += 1
                if ask({"已确认末窗":last,"参考":ref})!={"已回标":count}:
                    raise ValueError("功率参考未确认")
            proc.stdin.close()
            if proc.wait(timeout=30)!=0 or count!=expected["回标"]:
                raise ValueError("部署进程退出或累计参考异常")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not proc.stdin.closed:
                proc.stdin.close()
            proc.stdout.close()
        scores = {n:float(np.sqrt(v/len(clock))) for n,v in errs.items()}
        if any(abs(v-expected["scores"][n])>1e-7 for n,v in scores.items()):
            raise ValueError("整机评分未复现")
        states[site] = {"scores":scores,"回标":count,"实发":len(clock),"回执":events}
    means = {n:np.mean([v["scores"][n] for v in states.values()]) for n in rules}
    summary = {role+"误差":means[d[role]] for role in ("现场","固定","论文")}
    summary.update({"实发":sum(v["实发"] for v in states.values()),"回标":sum(v["回标"] for v in states.values()),"最大差":maxdiff})
    atomic_json(destination/"scores.json",{"summary":summary,"means":means,"sites":states,"权重SHA256":source["权重SHA256"]})
    print(json.dumps({"原始风机部署":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("原始风机部署已完成")
        verify(PRIOR, DEST)
