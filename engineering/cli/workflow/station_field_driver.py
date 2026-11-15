"""北京监测站的原始读数入口。站点预测按时发放，本站未来参考仅由终局评价读取。"""
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import zipfile

import joblib
import numpy as np
import pandas as pd

from harness.integrations.proposal_inbox import ProposalInbox
from harness.models.station_residual import Rule, StationTools, error, gate, history_trials, plan_site, tests
from harness.runtime.storage import LocalStore, atomic_json


DONORS = ("Aotizhongxin","Changping","Dongsi","Guanyuan")
SITES = ("Dingling","Gucheng","Tiantan")
WEATHER = ("TEMP","PRES","DEWP","RAIN","WSPM")
START = pd.Timestamp("2016-09-01")
STOP = START + pd.Timedelta(days=56)
FORWARD_SITES = ("Huairou","Nongzhanguan","Shunyi","Wanliu","Wanshouxigong")
FORWARD_START = pd.Timestamp("2016-12-01")


def table(archive, site, columns, start=START):
    member = f"PRSA_Data_20130301-20170228/PRSA_Data_{site}_20130301-20170228.csv"
    with zipfile.ZipFile(archive) as z, z.open(member) as file:
        raw = pd.read_csv(file,usecols=["year","month","day","hour",*columns])
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw[["year","month","day","hour"]]))
    if not raw.index.is_monotonic_increasing or raw.index.has_duplicates:
        raise ValueError("采集时钟缺失或重叠")
    return raw.reindex(pd.date_range(start-pd.Timedelta(days=180),start+pd.Timedelta(days=56,hours=-1),freq="h"))[columns]


def station_inputs(archive, site, start=START):
    raw = pd.concat([table(archive,d,["PM2.5"],start)["PM2.5"].rename(d) for d in DONORS],axis=1)
    weather = table(archive,site,list(WEATHER),start)
    # 所有插补只沿时间向前取已见值；开场尚无读数时使用固定默认值。
    filled = raw.ffill().fillna(0.)
    fields = filled.assign(median=filled.median(axis=1),mean=filled.mean(axis=1)).join(weather.ffill().fillna(0.))
    index=fields.index
    # 标签读取器单独加载；维护侧只返回发放前且到期的参考值。
    reference=table(archive,site,["PM2.5"],start)["PM2.5"]
    known=(index < start) & (index.hour%6==0)
    y=reference.where(known).to_numpy()
    day=((index-start)/pd.Timedelta(days=1)).to_numpy()
    valid=np.isfinite(y)
    fit=(day < -21)&valid
    windows=[(day>=at)&(day<at+7)&valid for at in (-21,-14,-7)]
    if min(int(w.sum()) for w in windows)<8 or int(fit.sum())<100:
        raise ValueError("已到期参考不足以独立训练和确认")
    return StationTools(fields.to_numpy(), DONORS, WEATHER),y,fit,windows,np.flatnonzero(day>=0)


def fixed_result(tools, rule, parent, fit, windows, y, active_errors):
    try:
        model=tools.train(rule,fit,y)
    except ValueError:
        return parent
    return model if gate(active_errors,tests(tools,model,windows,y)) else parent


def run(archive, output, forward=False, queue=None):
    inbox=ProposalInbox(queue or output/"queue")
    sites,start=(FORWARD_SITES,FORWARD_START) if forward else (SITES,START)
    fixed=list(StationTools.fixed([*DONORS,"median","mean"]))
    observations={}
    proposals=set()
    issued={}
    for station in sites:
        site_dir=output/station
        site_dir.mkdir(exist_ok=False)
        tools,y,fit,windows,service=station_inputs(archive,station,start)
        parent=tools.train(tools.static,fit,y)
        old_errors=tests(tools,parent,windows,y)
        results={}
        history=None
        paper_base=None
        if forward:
            history,report=history_trials(tools,fit,y,fixed)
            atomic_json(site_dir/"history.json",report)
            ordered=[fixed[i] for i in report["ordered"]]
            paper_base=ordered[0]
            # 两个重新拟合名额，与真实驱动相同。第二个候选要求使用不同背景或模型。
            r2=next(r for r in ordered if (r.background,r.correction)!=(paper_base.background,paper_base.correction))
            pair=[tools.train(r,fit,y) for r in (paper_base,r2)]
            champion=min(pair,key=lambda m:np.mean(tests(tools,m,windows[:2],y)**2))
            champion=champion if gate(old_errors,tests(tools,champion,windows,y)) else parent
            results["history-numeric"]=tools.issue(champion,service)
        with ThreadPoolExecutor(2) as pool:
            calls={arm:pool.submit(plan_site,inbox,station,arm,tools,parent,fit,windows,y,{},history,paper_base)
                   for arm in ("project","paper")}
            for arm,call in calls.items():
                selected,dec,tried=call.result()
                atomic_json(site_dir/(arm+"-decision.json"),dec)
                joblib.dump(selected,site_dir/(arm+".joblib"))
                results[arm]=tools.issue(joblib.load(site_dir/(arm+".joblib")),service)
                proposals.update(tried)
        results["static"]=tools.issue(parent,service)
        for name in tools.backgrounds:
            results["raw-"+name]=tools.x[service,tools.names.index(name)]
        # 一个无需语言模型的强自适应对手：看前两段最可靠的背景，试两种校正，
        # 只把提名的一种交给末段确认。与 Luna 的两次拟合名额相同。
        raw_errors={n:np.mean([error(y[m],tools.x[m,tools.names.index(n)])**2 for m in windows[:2]])
                    for n in tools.backgrounds}
        base=min(raw_errors,key=raw_errors.get)
        pair=[tools.train(Rule(base,k,180,1.),fit,y) for k in ("weather","network")]
        numeric=min(pair,key=lambda m:np.mean(tests(tools,m,windows[:2],y)**2))
        numeric=numeric if gate(old_errors,tests(tools,numeric,windows,y)) else parent
        results["numeric"]=tools.issue(numeric,service)
        for i,rule in enumerate(fixed):
            selected=fixed_result(tools,rule,parent,fit,windows,y,old_errors)
            results[f"fixed-{i:03d}"]=tools.issue(selected,service)
        np.savez_compressed(site_dir/"issued.npz",rows=service,**results)
        observations[station]=(tools,y,fit,windows,service,parent,old_errors)
        issued[station]=results
    # 所有计划、正式交付均已封存，冻结新提出的条件在各站回放；绝不回到驱动。
    extra=sorted(proposals-set(fixed),key=repr)
    for station,(tools,y,fit,windows,service,parent,old_errors) in observations.items():
        path=output/station
        audit={}
        for i,rule in enumerate(extra):
            candidate=fixed_result(tools,rule,parent,fit,windows,y,old_errors)
            audit[f"frozen-{i:02d}"]=tools.issue(candidate,service)
        np.savez_compressed(path/"frozen.npz",rows=service,**audit)
        issued[station].update(audit)
    scores={key:{} for key in issued[sites[0]]}; counts={}
    for station in sites:
        reference=table(archive,station,["PM2.5"],start).iloc[observations[station][4],0].to_numpy()
        valid=np.isfinite(reference)
        if not valid.any(): raise ValueError("本站无可评参考")
        counts[station]={"issued":len(reference),"scored":int(valid.sum())}
        for key,pred in issued[station].items():
            if not np.isfinite(pred).all() or pred.shape!=reference.shape:
                raise ValueError("服务输出不完整")
            scores[key][station]=error(reference[valid],pred[valid])
    mean={k:float(np.mean(list(v.values()))) for k,v in scores.items()}
    fixed_names=[k for k in mean if k.startswith(("fixed-","raw-")) or k=="static"]
    best=min(fixed_names,key=mean.get)
    extended=min([*fixed_names, *(k for k in mean if k.startswith("frozen-"))],key=mean.get)
    atomic_json(output/"scores.json",{"station_rmse":scores,"macro_rmse":mean,"best_fixed":best,
                "best_shared":extended,"fixed_recipes":[asdict(r) for r in fixed],"frozen_proposals":[asdict(r) for r in extra],
                "counts":counts,"sites":list(sites),"start":start.date().isoformat(),"history_enabled":forward})
    print(json.dumps({"现场":mean["project"],"论文":mean["paper"],"原版":mean["static"],"数值选择":mean["numeric"],
                      "最强固定":mean[best],"冻结提案后最强固定":mean[extended],
                      **({"历史数值选择":mean["history-numeric"]} if forward else {})},ensure_ascii=False),flush=True)


def main(argv):
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--archive",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--forward-history",action="store_true",help="五个新目标站，从冻结的12月时钟发放；提供较早试训记录")
    cli.add_argument("--queue",type=Path,help="按原请求摘要接续已交换的提案，交付目录必须为新目录")
    args=cli.parse_args(argv)
    with LocalStore(args.output).exclusive():
        if any((args.output/site).exists() for site in (*SITES,*FORWARD_SITES)):
            raise ValueError("该目录已经开始过交付，请使用新的运行目录")
        run(args.archive,args.output,args.forward_history,args.queue)
