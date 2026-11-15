"""伦敦 PM2.5 逐小时交付，模型先发布，六小时参考在该时刻交付后进入后续预测。"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import json
import zipfile

import joblib
import numpy as np
import pandas as pd

from harness.integrations.proposal_inbox import ProposalInbox
from harness.models.station_residual import StationTools, Rule, error, gate, history_trials, plan_site, tests
from harness.models.sparse_reference import SETTINGS, correct, name, setting_of, window_scores, choose_from_feedback
from harness.runtime.storage import LocalStore, atomic_json


DONORS=("BL0","BT4","BQ9","WM0")
TARGETS=("BX2","BQ7","BY7","CD1","CR8","CE2")
START=pd.Timestamp("2023-07-01")
CLOCK=pd.date_range(START-pd.Timedelta(days=180),START+pd.Timedelta(days=56,hours=-1),freq="h")
SEASON=("hour_sin","hour_cos","weekend")


def configure(start):
    global START, CLOCK
    START = pd.Timestamp(start)
    CLOCK = pd.date_range(START-pd.Timedelta(days=180), START+pd.Timedelta(days=56,hours=-1), freq="h")


def readings(archive, site):
    with zipfile.ZipFile(archive) as z:
        data=json.loads(z.read(site+".json"))["RawAQData"]
    if data["@SiteCode"]!=site or data["@SpeciesCode"]!="PM25":
        raise ValueError("源站身份或污染物不符")
    rows=data.get("Data") or []
    rows=[rows] if isinstance(rows,dict) else rows
    frame=pd.DataFrame(rows,columns=["@MeasurementDateGMT","@Value"])
    index=pd.DatetimeIndex(pd.to_datetime(frame["@MeasurementDateGMT"]))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("需要唯一且递增的 GMT 时钟")
    return pd.Series(pd.to_numeric(frame["@Value"],errors="coerce").to_numpy(),index=index).reindex(CLOCK)


def prepare(archive, nonlinear=False, flexible=False):
    donors=pd.concat([readings(archive,n).rename(n) for n in DONORS],axis=1)
    # 超过六小时的缺口回到其他源站当前中位，防止把数月未更新的值作为当前值。
    donors=donors.ffill(limit=6)
    median=donors.median(axis=1).ffill().fillna(0.)
    donors=donors.apply(lambda v:v.fillna(median))
    angle=2*np.pi*CLOCK.hour/24
    x=donors.assign(median=median,mean=donors.mean(axis=1),hour_sin=np.sin(angle),
                    hour_cos=np.cos(angle),weekend=(CLOCK.dayofweek>=5).astype(float))
    return StationTools(x.to_numpy(),DONORS,SEASON,nonlinear=nonlinear,flexible=flexible)


def roles(reference):
    days=((CLOCK-START)/pd.Timedelta(days=1)).to_numpy()
    y=np.where((days<0)&(CLOCK.hour%6==0),reference,np.nan)
    fit=(days < -21)&np.isfinite(y)
    windows=[(days>=a)&(days<a+7)&np.isfinite(y) for a in (-21,-14,-7)]
    if fit.sum()<80 or min(mask.sum() for mask in windows)<8:
        raise ValueError("开场前训练或三个核验窗口的参考不足")
    service=np.flatnonzero(days>=0)
    returned=np.where(CLOCK[service].hour%6==0,reference[service],np.nan)
    return y,fit,windows,service,returned


def base_bank(tools,rules,fit,y):
    base={}
    models={}
    for r in rules:
        key=(r.background,r.correction,r.days,r.features,r.learner,r.parameters)
        if key not in models:
            models[key]=tools.train(r,fit,y)
        model=(*models[key][:-1],r.strength)
        base[r]=tools.issue(model,np.arange(len(y)))
    return base


def deployment(parent, memo, tried, decision, parent_feedback=SETTINGS[0]):
    """保存实际可选的模型及反馈设置，重载后仍能按同一时序继续发放。"""
    models = [(parent, parent_feedback)]
    for rule, setting in tried:
        models.append((memo[rule], setting))
    for _, setting in tried:
        models.append((parent, setting))
    nominee = (Rule(**decision["nominated"]), setting_of(decision["feedback"])) if tried else None
    initial = tried.index(nominee) + 1 if decision["adopted"] else 0
    return {"models": models, "initial": initial, "memory_days": 28}


def emit_deployment(tools, packet, y, windows, service, returned):
    clock = np.arange(np.flatnonzero(windows[2])[0], len(y))
    known = np.concatenate((y[clock[clock < service[0]]], returned))
    columns = [correct(tools.issue(m, clock)[:,None], known, clock, s)[:,0] for m,s in packet["models"]]
    stream, choices = choose_from_feedback(np.column_stack(columns), known, clock, service[0],
                                          packet["initial"], packet["memory_days"])
    return stream[clock >= service[0]], choices[clock >= service[0]]


def fixed_stream(base, y, windows, service, returned, setting, adaptive):
    if not adaptive:
        return correct(base[service],returned,service,setting)
    # 固定对手也取得开场前最近一次反馈，以相同状态进入正式服务。
    clock=np.arange(np.flatnonzero(windows[2])[0],len(y))
    known=np.concatenate((y[clock[clock<service[0]]],returned))
    stream=correct(base[clock],known,clock,setting)
    return stream[clock>=service[0]]


def run(archive,out,adaptive=False,inbox=None,flexible=False,initial_spec=None):
    tools=prepare(archive,nonlinear=adaptive,flexible=flexible)
    initial_feedback=SETTINGS[0]
    if initial_spec is not None:
        tools.static=tools.parse({"hypothesis":"已冻结的初始模型","recipe":initial_spec["recipe"]})
        initial_feedback=setting_of(initial_spec["feedback"])
    rules=list(StationTools.fixed(tools.backgrounds))
    if adaptive:
        rules += [replace(r,features=tuple(tools.feature_terms)) for r in rules if r.correction=="network"]
    if flexible:
        from harness.models.station_learners import references
        rules += [Rule(background,"network",days,1.,learner=learner,parameters=params)
                  for learner,params in references() for background in tools.backgrounds for days in (60,180)]
    arms=("project","paper")
    inbox=inbox or ProposalInbox(out/"queue")
    issued={};late={};proposals=set();eligible=[];counts={};excluded={}
    for site in TARGETS:
        ref=readings(archive,site).to_numpy()
        if np.isfinite(ref[CLOCK<START]).mean()<.80:
            excluded[site]="开场前观测覆盖未到80%";continue
        try:
            y,fit,windows,service,returned=roles(ref)
        except ValueError as exc:
            # 已有运行条件在开场检查，不把缺参考误作训练器或整批服务失败。
            excluded[site]=str(exc)
            continue
        eligible.append(site);dest=out/site;dest.mkdir(exist_ok=False)
        parent=tools.train(tools.static,fit,y)
        joblib.dump((parent,initial_feedback),dest/"initial.joblib")
        parent,initial_feedback=joblib.load(dest/"initial.joblib")
        parent_base=tools.issue(parent,np.arange(len(y)))[:,None]
        parent_err=window_scores(parent_base,windows,y,initial_feedback)[:,0]
        original=fixed_stream(parent_base,y,windows,service,returned,initial_feedback,adaptive)[:,0]
        history,report=history_trials(tools,fit,y,rules)
        base=base_bank(tools,rules,fit,y)
        matrix=np.column_stack(list(base.values()))
        # 共用的早期模型已冻结。用三个窗口中前两窗的在线误差给出反馈强度依据；
        # 这里只展示历史首选结构，避免把最后确认窗泄露给两组驱动。
        oldbest=rules[report["ordered"][0]]
        ranks={}
        for setting in SETTINGS:
            err=window_scores(base[oldbest][:,None],windows[:2],y,setting)[:,0]
            ranks[name(setting)]=err.round(4).tolist()
        history["reference_feedback_pilot_on_historical_leader"]=ranks
        report["pilot_feedback"]=ranks
        atomic_json(dest/"history.json",report)
        preds={"static":original}
        memos={a:{} for a in arms}
        with ThreadPoolExecutor(2) as pool:
            parent_setting=initial_feedback if initial_spec is not None else None
            tasks={a:pool.submit(plan_site,inbox,site,a,tools,parent,fit,windows,y,memos[a],history,oldbest,True,parent_setting) for a in arms}
            for a,t in tasks.items():
                model,decision,tried=t.result()
                proposals.update(tried)
                selected=setting_of(decision["feedback"]) if decision["adopted"] else initial_feedback
                atomic_json(dest/(a+"-decision.json"),decision)
                joblib.dump((model,selected),dest/(a+".joblib"))
                model,selected=joblib.load(dest/(a+".joblib"))
                pred=tools.issue(model,service)
                preds[a]=correct(pred[:,None],returned,service,selected)[:,0]
                preds[a+"-never-correct"]=pred
                if adaptive:
                    packet=deployment(parent,memos[a],tried,decision,initial_feedback)
                    joblib.dump(packet,dest/(a+"-adaptive.joblib"))
                    packet=joblib.load(dest/(a+"-adaptive.joblib"))
                    adaptive_prediction, choices=emit_deployment(tools,packet,y,windows,service,returned)
                    initial_model,selected_feedback=packet["models"][packet["initial"]]
                    initial_base=tools.issue(initial_model,np.arange(len(y)))[:,None]
                    preds[a+"-initial-only"]=fixed_stream(initial_base,y,windows,service,returned,selected_feedback,True)[:,0]
                    preds[a]=adaptive_prediction
                    np.savez_compressed(dest/(a+"-choices.npz"),choices=choices)
        # 每条固定策略与驱动使用相同的探索/确认门槛。预先定义的不设门槛直接发布
        # 也是独立强对照，防止只赢过被不当拒绝的固定组合。
        trial_scores={}
        for setting in SETTINGS:
            losses=window_scores(matrix,windows,y,setting)
            accept=(np.mean(losses[:2]**2,axis=0)<np.mean(parent_err[:2]**2))&(losses[2]<parent_err[2])
            stream=fixed_stream(matrix,y,windows,service,returned,setting,adaptive)
            for i in range(len(rules)):
                key=f"{i}-{name(setting)}"
                preds["direct-"+key]=stream[:,i]
                preds["fixed-"+key]=stream[:,i] if accept[i] else original
                trial_scores[(i,setting)]=float(np.mean(losses[:2,i]**2))
            raw=tools.x[:,:len(tools.backgrounds)]
            for i,n in enumerate(tools.backgrounds):
                preds["raw-"+n+"-"+name(setting)]=fixed_stream(raw[:,i:i+1],y,windows,service,returned,setting,adaptive)[:,0]
        top=report["ordered"][0]
        second=next(i for i in report["ordered"] if (rules[i].background,rules[i].correction,rules[i].features,rules[i].learner)!=
                    (rules[top].background,rules[top].correction,rules[top].features,rules[top].learner))
        ix,setting=min((k for k in trial_scores if k[0] in (top,second)),key=trial_scores.get)
        preds["numeric"]=preds[f"fixed-{ix}-{name(setting)}"]
        if adaptive:
            numeric_models=[(parent,initial_feedback)]
            for index in (top,second):
                s=min((k[1] for k in trial_scores if k[0]==index),key=lambda v:trial_scores[(index,v)])
                numeric_models.append((tools.train(rules[index],fit,y),s))
            numeric_models += [(parent,s) for _,s in numeric_models[1:]]
            initial=1+(ix==second) if gate(parent_err,window_scores(base[rules[ix]][:,None],windows,y,setting)[:,0]) else 0
            packet={"models":numeric_models,"initial":int(initial),"memory_days":28}
            joblib.dump(numeric_models[int(initial)],dest/"numeric.joblib")
            joblib.dump(packet,dest/"numeric-adaptive.joblib")
            preds["numeric-adaptive"],_=emit_deployment(tools,packet,y,windows,service,returned)
        # 单独对照“沿用最近一次测量”；开场使用已经发出前的最后一个合法参考。
        last=y[np.flatnonzero(np.isfinite(y))[-1]]
        persistence=np.empty(len(service))
        for i,answer in enumerate(returned):
            persistence[i]=last
            if np.isfinite(answer):last=answer
        preds["raw-last-reference"]=persistence
        # 发布两臂的逐条输出和开发许可以内的参考时序。对照保留逐小时少量可复算基底，
        # 其余组合的分数只在所有现场请求结束后计算。
        np.savez_compressed(dest/"issued.npz",ids=service,project=preds["project"],paper=preds["paper"],
                            numeric=preds["numeric"],static=original,returned=returned,
                            project_base=preds["project-never-correct"],paper_base=preds["paper-never-correct"])
        issued[site]=preds
        late[site]=(y,fit,windows,service,returned,parent_err)
    extra=sorted({(r,s) for r,s in proposals if r not in rules},key=repr)
    for site in eligible:
        y,fit,windows,service,returned,parent_err=late[site]
        preds=issued[site]
        for i,(r,_) in enumerate(dict.fromkeys((r,None) for r,s in extra)):
            try:
                model=tools.train(r,fit,y)
                raw=tools.issue(model,np.arange(len(y)))
            except ValueError:
                raw=None
            for setting in SETTINGS:
                key=f"extra-{i}-{name(setting)}"
                if raw is None:
                    preds["fixed-"+key]=preds["direct-"+key]=preds["static"]
                    continue
                losses=window_scores(raw[:,None],windows,y,setting)[:,0]
                emit=fixed_stream(raw[:,None],y,windows,service,returned,setting,adaptive)[:,0]
                preds["direct-"+key]=emit
                preds["fixed-"+key]=emit if gate(parent_err,losses) else preds["static"]
    errors={k:{} for k in issued[eligible[0]]};names=list(errors)
    for site in eligible:
        ref=readings(archive,site).to_numpy()[late[site][3]]
        valid=np.isfinite(ref)
        if not np.any(valid):raise ValueError("符合开场条件的站点无终局答案")
        counts[site]={"issued":len(ref),"scored":int(valid.sum()),"returned":int(np.isfinite(late[site][4]).sum())}
        for k in names:
            errors[k][site]=error(ref[valid],issued[site][k][valid])
    means={k:float(np.mean(list(v.values()))) for k,v in errors.items()}
    fixed=[k for k in names if k.startswith(("fixed-","direct-","raw-")) or k=="static"]
    if adaptive:
        fixed += ["numeric","numeric-adaptive"]
    winner=min(fixed,key=means.get)
    for site in eligible:
        np.savez_compressed(out/site/"fixed-winner.npz",prediction=issued[site][winner])
    atomic_json(out/"scores.json",{"station_rmse":errors,"macro_rmse":means,"best_fixed":winner,
                "recipes":[asdict(r) for r in rules],"extra":[asdict(r) for r in dict.fromkeys(r for r,_ in extra)],
                "counts":counts,"excluded":excluded,"start":str(START.date()),"adaptive":adaptive,
                "flexible":flexible,"initial_spec":initial_spec})
    base_label="初次发布基模型" if adaptive else "去除现场持续反馈"
    print(json.dumps({"现场":means["project"],"论文":means["paper"],"最强固定":means[winner],"原版":means["static"],
                      "同工具数值选择":means["numeric"],base_label:means["project-never-correct"]},ensure_ascii=False))


def main(argv):
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--archive",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--start",default="2023-07-01")
    cli.add_argument("--adaptive",action="store_true",help="允许现场组装输入乘积并依据到期误差继续选择")
    cli.add_argument("--expand-models",action="store_true",help="现场可选择学习器和标量参数，各方共用历史模型比较")
    cli.add_argument("--initial-spec",type=Path,help="初始模型配方和参考修正配置的冻结JSON")
    args=cli.parse_args(argv)
    configure(args.start)
    with LocalStore(args.output).exclusive():
        if any((args.output/s).exists() for s in TARGETS):raise ValueError("不可覆盖既有交付")
        initial_spec=json.loads(args.initial_spec.read_bytes()) if args.initial_spec is not None else None
        run(args.archive,args.output,args.adaptive,flexible=args.expand_models,initial_spec=initial_spec)
