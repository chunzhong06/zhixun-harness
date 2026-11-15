"""只读已发行的模型与提案。逐小时重载核验，并量化改变训练配方与选站的分差。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow.station_field_driver import START, STOP, SITES, station_inputs
from harness.contracts import digest
from harness.models.station_residual import Rule, error
from harness.runtime.storage import atomic_json


def rule_of(raw):
    rule=Rule(**raw)
    if rule.where=="all":
        return Rule(rule.background, rule.correction, rule.days, rule.strength)
    return rule


def read(path):
    return json.loads(path.read_bytes())


def main():
    cli=ArgumentParser()
    cli.add_argument("--archive",type=Path,required=True)
    cli.add_argument("--run",type=Path,required=True)
    args=cli.parse_args()
    output=args.run/"audit.json"
    if output.exists():
        raise ValueError("完整审计已存在")
    result=read(args.run/"scores.json")
    names={rule_of(r):f"fixed-{i:03d}" for i,r in enumerate(result["fixed_recipes"])}
    for i,r in enumerate(result["frozen_proposals"]):
        names.setdefault(rule_of(r),f"frozen-{i:02d}")
    replay={}
    calls={"project":0,"paper":0}
    tried=set()
    score_delta=0.
    start=pd.Timestamp(result.get("start",START))
    for station in result.get("sites",SITES):
        folder=args.run/station
        tools,y,fit,windows,ids=station_inputs(args.archive,station,start)
        raw=tools.train(tools.static,fit,y)
        with zipfile.ZipFile(args.archive) as z:
            name=next(name for name in z.namelist() if f"Data_{station}_" in name)
            with z.open(name) as f:
                hourly=pd.read_csv(f,usecols=["year","month","day","hour","PM2.5"])
        index=pd.to_datetime(hourly[["year","month","day","hour"]])
        hour_grid=pd.date_range(start,start+pd.Timedelta(days=56,hours=-1),freq="h")
        target=pd.Series(hourly["PM2.5"].values,index=index).reindex(hour_grid).to_numpy()
        mask=np.isfinite(target)
        issued={}
        for packet in ("issued.npz","frozen.npz"):
            with np.load(folder/packet,allow_pickle=False) as f:
                if not np.array_equal(ids,f["rows"]):
                    raise ValueError("发放小时身份不一致")
                issued.update({k:f[k] for k in f.files if k!="rows"})
        if set(issued)!=set(result["macro_rmse"]):
            raise ValueError("策略未完整发放")
        for k,pred in issued.items():
            score_delta=max(score_delta,abs(error(target[mask],pred[mask])-result["station_rmse"][k][station]))
        replay[station]={}
        for arm in ("project","paper"):
            decision=read(folder/(arm+"-decision.json"))
            model=joblib.load(folder/(arm+".joblib"))
            if not np.array_equal(tools.issue(model,ids),issued[arm]):
                raise ValueError("发布文件不能复现逐条预测")
            for i,trial in enumerate(decision["trials"],1):
                req,=list((args.run/"queue/requests"/arm).glob(f"{station}-trial-{i}-*.json"))
                body=read(req)
                if result.get("history_enabled"):
                    hist=read(folder/"history.json")
                    first_eval=np.flatnonzero(windows[0])[0]
                    if not all(b["last_fit_hour"]<b["first_test_hour"]<=b["last_test_hour"]<first_eval for b in hist["blocks"]):
                        raise ValueError("历史比较使用了正式探索或未来参考")
                    top=body["context"]["past_trained_model_comparisons"]["overall"][0]
                    rank=hist["ordered"][0]
                    if (rule_of(top["recipe"])!=rule_of(result["fixed_recipes"][rank]) or not
                            np.allclose(top["fold_rmse"],hist["rmse"][rank],atol=.00051,rtol=0)):
                        raise ValueError("提供给驱动的历史分数与拟合回执不对应")
                messages=[{"role":"system","content":body["instruction"]},{"role":"user","content":
                            json.dumps(body["context"],allow_nan=False,ensure_ascii=False)}]
                reply=read(args.run/"queue/responses"/arm/req.name)
                if not (digest({"model":"gpt-6-luna","messages":messages})==body["request_id"]==reply["request_id"]
                        and reply["plan"]==trial["proposal"]):
                    raise ValueError("试验动作与真实驱动回执不对应")
                calls[arm]+=1
                if arm=="project" and "observed_rmse" in trial:
                    tried.add(names[rule_of(trial["proposal"]["recipe"])])
            direct=tools.issue(raw,ids)
            if not np.array_equal(direct,issued["static"]):
                raise ValueError("共同起点不能重放")
            rule=rule_of(decision["nominated"]) if "nominated" in decision else None
            replay[station][arm]={"adopted":decision["adopted"],"nominated":decision.get("nominated"),
                                  "holding":error(target[mask],direct[mask]),"issued":error(target[mask],issued[arm][mask]),
                                  "changed_hours":int(np.sum(direct!=issued[arm]))}
            if rule is not None:
                key=names[rule]
                if not np.array_equal(issued[key],issued[arm]):
                    raise ValueError("同起点冻结同配方的预测有差异")
                replay[station][arm]["same_recipe_replay"]=key
    if score_delta>1e-10:
        raise ValueError("重新取目标参考后总分不符")
    best=min(tried,key=result["macro_rmse"].get) if tried else "static"
    audit={"逐站重放":replay,"驱动请求数":calls,"评分最大绝对差":score_delta,"实测策略数":len(result["macro_rmse"]),
           "项目提出的配方若全程固定":{"best":best,"rmse":result["macro_rmse"][best],
            "all":{k:result["macro_rmse"][k] for k in sorted(tried)}}}
    atomic_json(output,audit)
    print(json.dumps({"复核策略数":audit["实测策略数"],"评分最大差":score_delta,
                      "双方请求数":sum(calls.values()),"项目所提最强固定":audit["项目提出的配方若全程固定"]["rmse"]},ensure_ascii=False))


if __name__=="__main__":
    main()
