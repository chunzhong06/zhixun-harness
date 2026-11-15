"""让正式入口从实际原件重新发放，再向固定查询与论文询问器开放加热特征。"""
import json
import subprocess
import sys
import numpy as np

from score_temperature_heater_response import RAW, INPUT, DEST as PRIOR, CHANNELS, POLICIES, read, digest, LocalStore, atomic_json, f1_score, recall_score
from score_gas_audit_feedback import DEST as AUDIT
from engineering.cli.workflow.gas_heater_stream import FIXED, LEARN, ROOT

DEST = PRIOR.parent/"gas-heater-entry"


def evaluate(site,policy,view):
    rel = CHANNELS[site][0]
    root = INPUT/rel
    folder = DEST/site/policy/view
    cli = [sys.executable,"-B",str(ROOT/"engineering/cli/workflow/gas_heater_stream.py"),"--archive",str(RAW),
           "--stream",str(root/"stream.json"),"--query",policy,"--query-view",view,"--output",str(folder)]
    if site=="换件" and policy=="混合抽查" and view=="均值":
        subprocess.run([*cli,"--through","4"],cwd=ROOT,check=True)
    subprocess.run(cli,cwd=ROOT,check=True)
    run = read(folder/"run.json")
    baseline = read(PRIOR/site/policy/"run.json")
    qrun = read(AUDIT/site/policy/"run.json")
    truth = read(root/"evaluation.json")
    ids,cols = [],{}
    for event,original,issued in zip(run["events"],baseline["requests"],qrun["events"],strict=True):
        p = read(folder/"accepted"/(event["request"]+".json"))["payload"]
        if digest([run["identity"],p])!=event["request"]:
            raise ValueError("入口回执不一致")
        if view=="均值":
            old = read(PRIOR/site/policy/"accepted"/(original+".json"))["payload"]
            if (digest([baseline["identity"],old])!=original or old["ids"]!=p["ids"] or
                    p["已见反馈"]!=old["已见反馈"] or event["selected"]!=issued["selected"] or
                    any(v!=old["预测"][k] for k,v in p["预测"].items())):
                raise ValueError("原始入口与实验的查询或预测不一致")
        ids.extend(p["ids"])
        for k,v in {**p["预测"], LEARN:p["现场选择预测"]}.items():
            cols.setdefault(k,[]).extend(v)
    if len(set(ids))!=len(truth) or set(ids)!=set(truth) or len(ids)!=len(truth):
        raise ValueError("入口没有完整覆盖测量")
    y=[truth[k] for k in ids]
    return {k:{"F1":float(f1_score(y,v,average="macro")),"最差召回":float(recall_score(y,v,average=None).min())} for k,v in cols.items()}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已复核入口")
        arms = [(q,"加热") for q in POLICIES]+[("混合抽查","均值"),("BADGE","均值")]
        streams = {s:{q+"-"+v:evaluate(s,q,v) for q,v in arms} for s in CHANNELS}
        means={a:{k:{m:float(np.mean([v[a][k][m] for v in streams.values()])) for m in ("F1","最差召回")}
                  for k in streams["全通道"][a]} for a in streams["全通道"]}
        old=read(PRIOR/"scores.json")["decision"]
        f0,p0=old["固定F1"],old["论文F1"]
        metric=lambda a:means[a[0]][a[1]]["F1"]
        fixed=max(((a,k) for a in means if not a.startswith("BADGE") for k in means[a] if k!=LEARN),key=metric)
        paper=max(((a,k) for a in means if a.startswith("BADGE") for k in means[a]),key=metric)
        main_arm=("混合抽查-加热",LEARN)
        gap=metric(main_arm)-max(metric(fixed),metric(paper),f0,p0)
        d={"主方案":main_arm,"现场F1":metric(main_arm),"新查询最强固定":fixed,"固定F1":max(f0,metric(fixed)),
           "论文":paper,"论文F1":max(p0,metric(paper)),"主方案最小领先":gap,
           "进独立":gap>=.01 and means[main_arm[0]][LEARN]["最差召回"]>=means[fixed[0]][fixed[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"streams":streams,"means":means,"decision":d})
        print(json.dumps({"入口比较":d},ensure_ascii=False))


if __name__=="__main__":
    main()
