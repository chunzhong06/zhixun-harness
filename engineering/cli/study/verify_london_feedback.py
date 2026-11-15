"""从伦敦原始小时流重放发布模型、参考到达顺序和最强固定，核对驱动的增量。"""
from argparse import ArgumentParser
import json
from dataclasses import replace
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scenario
from engineering.cli.workflow.london_field_driver import TARGETS, prepare, readings, roles
from harness.contracts import digest
from harness.models.sparse_reference import SETTINGS, name, setting_of
from harness.models.station_residual import Rule, error, gate
from harness.runtime.storage import atomic_json


def read(p):
    return json.loads(p.read_bytes())


def replay(base,clock,feedback,setting):
    """一次只处理一个时点。返回答案前先发放，与入口按段计算的实现独立。"""
    last=None
    issued=[]
    for b,t,y in zip(base,clock,feedback,strict=True):
        value=b if last is None else max(0,b+setting[0]*np.exp(-(t-last[0])/setting[1])*last[1])
        issued.append(value)
        if np.isfinite(y):
            last=(t,y-b)
    return np.asarray(issued)


def read_rule(data):
    r=Rule(**data)
    return replace(r,limit=0.) if r.where=="all" else r


def window_losses(tools,model,windows,y,setting):
    return np.asarray([error(y[ix],replay(tools.issue(model,ix),ix,y[ix],setting))
                       for ix in (np.flatnonzero(mask) for mask in windows)])


def published_trace(tools,model,setting,y,windows,service,after,warm):
    clock=np.arange(np.flatnonzero(windows[2])[0],len(y)) if warm else service
    known=np.concatenate((y[clock[clock<service[0]]],after)) if warm else after
    return replay(tools.issue(model,clock),clock,known,setting)[clock>=service[0]]


def score_with_gate(tools,rule,setting,y,fit,windows,service,after,direct=False,warm=False,
                    parent_feedback=SETTINGS[0]):
    model=tools.train(rule,fit,y)
    if not direct:
        parent=tools.train(tools.static,fit,y)
        old=window_losses(tools,parent,windows,y,parent_feedback)
        if not gate(old,window_losses(tools,model,windows,y,setting)):
            model,setting=parent,parent_feedback
    return published_trace(tools,model,setting,y,windows,service,after,warm)


def replay_adaptive(tools,packet,y,windows,service,returned):
    """逐时重算各候选损失列表，与入口的累计和实现独立。"""
    clock=np.arange(np.flatnonzero(windows[2])[0],len(y))
    known=np.concatenate((y[clock[clock<service[0]]],returned))
    candidates=np.column_stack([replay(tools.issue(model,clock),clock,known,setting)
                                for model,setting in packet["models"]])
    selected=packet["initial"]
    history=[]
    emitted=[]
    selected_ids=[]
    for i,now in enumerate(clock):
        emitted.append(candidates[i,selected])
        selected_ids.append(selected)
        if np.isfinite(known[i]):
            history.append((now,(candidates[i]-known[i])**2))
            history=[(at,loss) for at,loss in history if at>now-packet["memory_days"]*24]
            if now>=service[0]:
                selected=int(np.argmin(np.mean([loss for _,loss in history],axis=0)))
    return np.asarray(emitted)[clock>=service[0]],np.asarray(selected_ids)[clock>=service[0]]


def first_packet(tools,parent,model,setting,y,windows,parent_feedback=SETTINGS[0]):
    return {"models":[(parent,parent_feedback),(model,setting),(parent,setting)],
            "initial":int(gate(window_losses(tools,parent,windows,y,parent_feedback),
                               window_losses(tools,model,windows,y,setting))),"memory_days":28}


def main():
    ap=ArgumentParser()
    ap.add_argument("--archive",required=True,type=Path)
    ap.add_argument("--run",required=True,type=Path)
    ap.add_argument("--report-name",default="audit.json",help="独立复核报告名，已有报告不可覆盖")
    ap.add_argument("--verify-observations",action="store_true",help="核对候选总分与分组说明来自同一逐时预测")
    ap.add_argument("--compare-run",type=Path,help="修复配对时核对其他交付逐条保持")
    ap.add_argument("--changed-pair",action="append",default=[],help="允许改变的组:站点，例如 project:CE2")
    args=ap.parse_args()
    root=args.run
    if Path(args.report_name).name!=args.report_name or not args.report_name.endswith(".json"):
        raise ValueError("核对报告名必须为当前目录的JSON文件")
    if (root/args.report_name).exists():
        raise ValueError("已完成的重放不可覆盖")
    result=read(root/"scores.json")
    scenario.configure(result["start"])
    CLOCK,START=scenario.CLOCK,scenario.START
    adaptive=result.get("adaptive",False)
    tools=prepare(args.archive,nonlinear=adaptive,flexible=result.get("flexible",False))
    parent_feedback=SETTINGS[0]
    if result.get("initial_spec") is not None:
        tools.static=read_rule(result["initial_spec"]["recipe"])
        parent_feedback=setting_of(result["initial_spec"]["feedback"])
    ids={read_rule(r):str(i) for i,r in enumerate(result["recipes"])}
    ids.update({read_rule(r):f"extra-{i}" for i,r in enumerate(result["extra"])})
    lookup={v:k for k,v in ids.items()}
    winner=result["best_fixed"]
    tally={"project":0,"paper":0}
    reused={"project":0,"paper":0}
    fits={"project":0,"paper":0}
    valid_trials={"project":0,"paper":0}
    used=set(); per_site={};max_diff=0.
    for station in TARGETS:
        truth=readings(args.archive,station).to_numpy()
        eligible=np.isfinite(truth[CLOCK<START]).mean()>=.8
        if eligible:
            try:
                prepared=roles(truth)
            except ValueError:
                eligible=False
        if eligible!=(station in result["counts"]):
            raise ValueError("事前站点覆盖条件错误")
        if not eligible:
            continue
        y,fit,windows,service,returned=prepared
        valid=np.isfinite(truth[service])
        count=result["counts"][station]
        if (int(valid.sum()),int(np.isfinite(returned).sum()),len(service))!=(count["scored"],count["returned"],count["issued"]):
            raise ValueError("参考数目或发放行数不符")
        site=root/station
        with np.load(site/"issued.npz",allow_pickle=False) as a:record={k:a[k] for k in a.files}
        if not np.array_equal(record["ids"],service) or not np.array_equal(record["returned"],returned,equal_nan=True):
            raise ValueError("运行时参考时序与源资料不符")
        parent=tools.train(tools.static,fit,y)
        initial_trace=published_trace(tools,parent,parent_feedback,y,windows,service,returned,adaptive)
        if not np.allclose(initial_trace,record["static"],rtol=0,atol=1e-10):
            raise ValueError("初始模型及其反馈状态与冻结配置不符")
        if (site/"initial.joblib").exists():
            saved_parent,saved_feedback=joblib.load(site/"initial.joblib")
            saved_trace=published_trace(tools,saved_parent,saved_feedback,y,windows,service,returned,adaptive)
            if saved_feedback!=parent_feedback or not np.array_equal(saved_trace,initial_trace):
                raise ValueError("初始发布权重重载后不同")
        per_site[station]={}
        for arm in ("project","paper"):
            seen_fits=set()
            dec=read(site/(arm+"-decision.json"))
            expected_parent=window_losses(tools,parent,windows,y,parent_feedback)
            if not np.allclose(expected_parent,dec["parent_errors"],rtol=0,atol=1e-10):
                raise ValueError("换版比较使用了不同的初始模型或反馈")
            model,setting=joblib.load(site/(arm+".joblib"))
            base=tools.issue(model,service)
            emission=replay(base,service,returned,setting)
            if adaptive:
                packet=joblib.load(site/(arm+"-adaptive.joblib"))
                incumbent,incumbent_setting=packet["models"][0]
                incumbent_trace=published_trace(tools,incumbent,incumbent_setting,y,windows,service,returned,True)
                if incumbent_setting!=parent_feedback or not np.array_equal(incumbent_trace,initial_trace):
                    raise ValueError("持续选择中的原模型或反馈被另一组决定改写")
                emission,selected_ids=replay_adaptive(tools,packet,y,windows,service,returned)
                with np.load(site/(arm+"-choices.npz")) as archived:
                    if not np.array_equal(selected_ids,archived["choices"]):
                        raise ValueError("在线选择身份与独立重算不符")
            delta=float(np.max(np.abs(emission-record[arm])))
            max_diff=max(max_diff,delta)
            if delta>1.e-10 or not np.array_equal(base,record[arm+"_base"]):
                raise ValueError("逐小时重放与真实发布不符")
            if abs(error(truth[service][valid],emission[valid])-result["station_rmse"][arm][station])>1e-10:
                raise ValueError("已发预测误差重算不符")
            first=record["static"]
            for i,trial in enumerate(dec["trials"],1):
                file,=list((root/"queue/requests"/arm).glob(f"{station}-trial-{i}-*.json"))
                q=read(file);a=read(root/"queue/responses"/arm/file.name)
                messages=[{"role":"system","content":q["instruction"]},{"role":"user","content":
                    json.dumps(q["context"],ensure_ascii=False,allow_nan=False)}]
                if digest({"model":"gpt-6-luna","messages":messages})!=q["request_id"] or a["request_id"]!=q["request_id"] or a["plan"]!=trial["proposal"]:
                    raise ValueError("真实提案与模型记录不符")
                tally[arm]+=1
                if "replayed_from" in a:
                    prior=read(ROOT/a["replayed_from"])
                    if prior["trials"][a.get("replayed_trial",1)-1]["proposal"]!=a["plan"]:
                        raise ValueError("首案重放未保留原提案")
                    reused[arm]+=1
                if "observed_rmse" in trial:
                    valid_trials[arm]+=1
                    r=read_rule(a["plan"]["recipe"]);s=setting_of(a["plan"]["feedback"])
                    if r not in seen_fits:
                        fits[arm]+=1
                        seen_fits.add(r)
                    if args.verify_observations:
                        observed_model=tools.train(r,fit,y)
                        for wi,mask in enumerate(windows[:2],1):
                            ix=np.flatnonzero(mask)
                            pred=replay(tools.issue(observed_model,ix),ix,y[ix],s)
                            rmse=error(y[ix],pred)
                            if abs(rmse-trial["observed_rmse"][wi-1])>1e-10:
                                raise ValueError("候选实训总分不符")
                            stats=trial["remaining_errors"][f"window_{wi}"]
                            if stats["parent_rmse"]!=round(rmse,3):
                                raise ValueError("分组说明中的总分与已发预测不符")
                            for feature,values in stats["median_and_mean_error_low_high"].items():
                                col=tools.x[ix,tools.names.index(feature)]
                                threshold=float(np.median(col));low=col<=threshold
                                diff=pred-y[ix]
                                expected=[round(threshold,2),round(float(diff[low].mean()),2),round(float(diff[~low].mean()),2)]
                                if values!=expected:
                                    raise ValueError("分组错误说明与实际发出的预测不符")
                    if arm=="project":
                        used.update(prefix+ids[r]+"-"+name(s) for prefix in ("fixed-","direct-"))
                    if i==1:
                        if adaptive:
                            parent=tools.train(tools.static,fit,y)
                            first_model=tools.train(r,fit,y)
                            fp=first_packet(tools,parent,first_model,s,y,windows,parent_feedback)
                            first,_=replay_adaptive(tools,fp,y,windows,service,returned)
                        else:
                            first=score_with_gate(tools,r,s,y,fit,windows,service,returned,parent_feedback=parent_feedback)
            per_site[station][arm]={"adopted":dec["adopted"],"rule":dec.get("nominated"),"feedback":setting,
                     "first_only":error(truth[service][valid],first[valid]),"issued":error(truth[service][valid],emission[valid]),
                     "without_feedback":error(truth[service][valid],base[valid])}
            if adaptive:
                per_site[station][arm]["switches"]=int(np.count_nonzero(np.diff(selected_ids)))
                per_site[station][arm]["uses_new_features"]=any(bool(t["proposal"]["recipe"].get("features")) for t in dec["trials"])
                clock=np.arange(np.flatnonzero(windows[2])[0],len(y))
                known=np.concatenate((y[clock[clock<service[0]]],returned))
                initial_model,selected_feedback=packet["models"][packet["initial"]]
                frozen=replay(tools.issue(initial_model,clock),clock,known,selected_feedback)[clock>=service[0]]
                per_site[station][arm]["frozen_initial_with_same_feedback_state"]=error(truth[service][valid],frozen[valid])
                per_site[station][arm]["initial_base_rmse"]=per_site[station][arm].pop("without_feedback")
        original=record["static"]
        if winner.startswith(("fixed-","direct-")):
            prefix,word=winner.split("-",1);rule_id,gain,hours=word.rsplit("-",2)
            predicted=score_with_gate(tools,lookup[rule_id],(float(gain[1:]),float(hours[1:])),y,fit,windows,service,returned,
                                     direct=prefix=="direct",parent_feedback=parent_feedback)
            if adaptive:
                predicted=score_with_gate(tools,lookup[rule_id],(float(gain[1:]),float(hours[1:])),y,fit,windows,service,returned,
                                         direct=prefix=="direct",warm=True,parent_feedback=parent_feedback)
        elif winner=="raw-last-reference":
            last=y[np.flatnonzero(np.isfinite(y))[-1]]
            predicted=[]
            for val in returned:
                predicted.append(last)
                if np.isfinite(val):last=val
            predicted=np.asarray(predicted)
        elif winner.startswith("raw-"):
            raw,gain,hours=winner[4:].rsplit("-",2)
            raw_clock=np.arange(np.flatnonzero(windows[2])[0],len(y)) if adaptive else service
            known=np.concatenate((y[raw_clock[raw_clock<service[0]]],returned)) if adaptive else returned
            predicted=replay(tools.x[raw_clock,tools.names.index(raw)],raw_clock,known,(float(gain[1:]),float(hours[1:])))
            predicted=predicted[raw_clock>=service[0]]
        elif winner=="numeric-adaptive":
            predicted,_=replay_adaptive(tools,joblib.load(site/"numeric-adaptive.joblib"),y,windows,service,returned)
        elif winner=="numeric":
            if (site/"numeric.joblib").exists():
                model,setting=joblib.load(site/"numeric.joblib")
                predicted=published_trace(tools,model,setting,y,windows,service,returned,adaptive)
            else:
                predicted=record["numeric"]
        else:
            predicted=original
        with np.load(site/"fixed-winner.npz") as w: diff=float(np.max(np.abs(predicted-w["prediction"])))
        if diff>1.e-10:raise ValueError("固定赢家无法重放")
        max_diff=max(max_diff,diff)
    means={k:float(np.mean(list(v.values()))) for k,v in result["station_rmse"].items()}
    if any(abs(v-result["macro_rmse"][k])>1e-10 for k,v in means.items()):
        raise ValueError("逐站汇总存在差异")
    all_fixed=[k for k in means if k.startswith(("fixed-","direct-","raw-")) or k=="static"]
    if adaptive:
        all_fixed += ["numeric","numeric-adaptive"]
    if winner!=min(all_fixed,key=means.get):
        raise ValueError("最强固定身份不符")
    own=min(used,key=means.get) if used else "static"
    audit={"requests":tally,"fits":fits,"replay_max_abs":max_diff,"sites":per_site,"fixed_winner":winner,
           "own_best_fixed":own,"own_fixed_rmse":means[own],"first_only":float(np.mean([v["project"]["first_only"] for v in per_site.values()])),
           "raw_strategies_beaten":sum(means["project"]<means[k] for k in all_fixed),"comparisons":len(all_fixed)}
    audit["replayed_requests"]=reused
    audit["valid_trials"]=valid_trials
    audit["new_requests"]={arm:tally[arm]-reused[arm] for arm in tally}
    audit["candidate_observations_verified"]=args.verify_observations
    audit["own_fixed_includes_direct"]=True
    if args.compare_run is not None:
        checked=0;changed={}
        valid_pairs={arm+":"+site for arm in ("project","paper") for site in per_site}
        if not set(args.changed_pair)<=valid_pairs:
            raise ValueError("变化对象不属于本次已发放的组及站点")
        for station in per_site:
            with np.load(root/station/"issued.npz") as current, np.load(args.compare_run/station/"issued.npz") as prior:
                if not np.array_equal(current["ids"],prior["ids"]):
                    raise ValueError("修复配对发放时刻不同")
                for arm in ("project","paper"):
                    pair=arm+":"+station
                    delta=float(np.max(np.abs(current[arm]-prior[arm])))
                    if pair in args.changed_pair:changed[pair]=delta
                    else:
                        if delta!=0.:raise ValueError("未受影响的交付发生变化："+pair)
                        checked+=len(current[arm])
        audit["paired_repair"]={"original":str(args.compare_run),"unchanged_predictions":checked,
                                "allowed_pairs_max_abs":changed}
    if adaptive:
        audit["frozen_initial_with_same_feedback_state"]=float(np.mean([
            v["project"]["frozen_initial_with_same_feedback_state"] for v in per_site.values()]))
    atomic_json(root/args.report_name,audit)
    print(json.dumps({"重放最大差":max_diff,"请求":sum(tally.values()),"仅首案":audit["first_only"],
                      "项目所提的最强固定":means[own],"固定对照数":len(all_fixed),
                      "同起始反馈且固定初次选择":audit.get("frozen_initial_with_same_feedback_state")},ensure_ascii=False))


if __name__=="__main__":
    main()
