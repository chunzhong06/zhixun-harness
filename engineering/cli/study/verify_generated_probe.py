"""独立核对历史输入检查、原首案身份、生成模型交付与逐时反馈归因。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.study.verify_generated_delivery import losses,accepts,stream,replay
from harness.contracts import digest
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs,source_for_rule
from harness.models.sparse_reference import setting_of,SETTINGS,name
from harness.models.station_residual import Rule,error
from harness.runtime.storage import atomic_json


def read(path):return json.loads(path.read_bytes())


def same(actual,expected,reason):
    if not np.allclose(actual,expected,rtol=0,atol=1e-10,equal_nan=True):raise ValueError(reason)


def predict(model,x,limit=None,fallback=None):
    raw=[];issued=[];failed=[]
    for i,row in enumerate(x):
        try:
            value=np.asarray(model.predict(row[None,:]),dtype=float)
            if value.shape!=(1,) or not np.isfinite(value).all():raise ValueError("无效当前输出")
            number=float(value[0]);raw.append(number)
            if limit is not None and abs(number)>limit:raise ValueError("超限")
            issued.append(max(0.,number))
        except Exception:
            if fallback is None:raise
            if len(raw)==i:raw.append(float("nan"))
            issued.append(float(fallback[i]));failed.append(i)
    return np.asarray(raw),np.asarray(issued),failed


def initial_and_stream(parent_base,parent_setting,candidates,y,windows,service,returned):
    old=losses(parent_base,windows,y,parent_setting)
    selected=min(range(len(candidates)),key=lambda i:np.mean(candidates[i][3][:2]**2)) if candidates else None
    initial=selected+1 if selected is not None and accepts(old,candidates[selected][3]) else 0
    options=[(parent_base,parent_setting)]+[(b,s) for _,b,s,_ in candidates]+[(parent_base,s) for _,_,s,_ in candidates]
    pred,choices,frozen=stream(options,initial,y,windows,service,returned)
    return pred,choices,frozen,initial,selected


def main():
    ap=ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    args=ap.parse_args();out=args.run;result=read(out/"scores.json")
    if (out/"audit.json").exists():raise ValueError("不可覆盖核验结果")
    if not result["historical_probe_enabled"]:raise ValueError("此核验用于历史输入检查运行")
    old=Path(result["first_proposal_origin"]) if result["first_proposal_origin"] else None
    original=Path(result["original"]);editing=result.get("reference_code_enabled",False)
    sharing=result.get("shared_first_enabled",False);has_guarded="guarded-original" in result["macro_rmse"]
    scene.configure(result["start"]);tools=scene.prepare(args.archive,True,True)
    tools.static=Rule(**result["initial_spec"]["recipe"]);parent_setting=setting_of(result["initial_spec"]["feedback"])
    features,feature_names=code_inputs(tools) if editing else (tools.x[:,:9],tools.names[:9])
    model_count=0;calls=0;first_replays=0;scored=0;issued_rows=0;guard_scores={};fallback_rows=0;probe_rejections=[]
    oracle_rows=[]
    first_sources={read(out/site/"decision.json")["trials"][0].get("source_sha256") for site in result["counts"]}
    first_sources.discard(None)
    for site in result["counts"]:
        dest=out/site;truth=scene.readings(args.archive,site).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
        ids=np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&(scene.CLOCK<scene.START))
        probe=scene.CLOCK[ids]<scene.START-np.timedelta64(7,'D');limit=1000.*max(1.,float(np.max(np.abs(y[fit]))))
        parent,_=joblib.load(dest/"initial.joblib");parent_base=tools.issue(parent,np.arange(len(y)))
        with np.load(dest/"training-input.npz") as z:
            for key,value in (("train_x",features[fit]),("train_y",y[fit]),("predict_ids",ids),("predict_x",features[ids])):
                if not np.array_equal(z[key],value):raise ValueError("历史试训输入或标签时序不符")
        with np.load(dest/"service-input.npz") as z:
            if set(z.files)!={"predict_x","predict_ids","fallback","prediction_limit"}:raise ValueError("服务输入包含额外字段")
            same(z["predict_x"],features[service],"服务读数不同");same(z["predict_ids"],service,"服务时钟不同")
            same(z["fallback"],parent_base[service],"回退起点不同");same(z["prediction_limit"],limit,"数值上限不同")
        cache={};diagnostics={};source_folders={}
        decision=read(dest/"decision.json")
        for trial_id,trial in enumerate(decision["trials"],1):
            proposal=trial["proposal"]
            if trial_id==1 and old is not None:
                path=old/site/"trial-1/proposal.json"
                if proposal!=read(path) or trial["original_sha256"]!=sha256(path.read_bytes()).hexdigest():raise ValueError("原首案被改写")
                first_replays+=1
            else:
                task="code-share" if sharing else "code-edit" if editing else "code-probe"
                qpath,=list((out/"queue/requests"/task).glob(f"{site}-trial-{trial_id}-*.json"));q=read(qpath)
                body={"model":"gpt-6-luna","messages":[{"role":"system","content":q["instruction"]},
                    {"role":"user","content":json.dumps(q["context"],ensure_ascii=False,allow_nan=False)}]}
                a=read(out/"queue/responses"/task/qpath.name)
                if digest(body)!=q["request_id"] or a["request_id"]!=q["request_id"] or a["plan"]!=proposal:raise ValueError("新提案身份不同")
                if q["context"]["trials"]!=decision["trials"][:trial_id-1]:raise ValueError("提案没有收到对应实际试训反馈")
                if editing:
                    if q["context"]["reference_python_source"]!=source_for_rule(tools,tools.static):raise ValueError("参考训练源码不同")
                    if q["context"]["feature_names"]!=feature_names:raise ValueError("代码输入名称顺序不同")
                if sharing:
                    entries=q["context"]["shared_first_trials"];expected_sites=[]
                    for origin_site in result["counts"]:
                        first=read(old/origin_site/"decision.json")["trials"][0]
                        if origin_site==site or "rejected" in first or "observed_rmse" not in first:continue
                        expected_sites.append(origin_site)
                        entry=next(item for item in entries if item["origin_site"]==origin_site)
                        if set(entry)!={"origin_site","proposal","source_sha256","origin_observed_rmse","origin_parent_rmse","origin_remaining_errors","scope"}:
                            raise ValueError("共享记录含非首案历史字段")
                        for now,before in (("proposal","proposal"),("source_sha256","source_sha256"),("origin_observed_rmse","observed_rmse"),("origin_remaining_errors","remaining_errors")):
                            if entry[now]!=first[before]:raise ValueError("共享记录不是来源首案原观察")
                        origin_q,=list((old/"queue/requests/code-edit").glob(origin_site+"-trial-1-*.json"))
                        past=read(origin_q)["context"]["past"]
                        if entry["origin_parent_rmse"]!=[past[f"window_{i}"]["parent_rmse"] for i in (1,2)]:raise ValueError("共享来源起点分数不同")
                    if [item["origin_site"] for item in entries]!=expected_sites:raise ValueError("共享首案范围不同")
                calls+=1
            if "source_sha256" not in trial:continue
            identity=sha256(proposal["python_source"].encode()).hexdigest()
            if identity!=trial["source_sha256"]:raise ValueError("源码身份不同")
            source_folders.setdefault(identity,dest/f"trial-{trial_id}")
        for folder in sorted((dest/"fixed").glob("*")) if (dest/"fixed").exists() else []:
            if (folder/"model.joblib").exists() and (folder/"base.npz").exists():source_folders[folder.name]=folder
        # 每份已保存的本站和跨站权重都重新加载。这里不重新训练或请求新提案。
        for identity,folder in source_folders.items():
            source=(folder/"source.py").read_text(encoding="utf-8")
            if sha256(source.encode()).hexdigest()!=identity:raise ValueError("训练源码与缓存身份不同")
            load_module(folder/"source.py");model=joblib.load(folder/"model.joblib")
            raw,values,_=predict(model,features[ids]);base=np.zeros(len(y));base[ids]=values
            with np.load(folder/"base.npz") as z:
                same(z["ids"],ids,"历史行不同");same(z["prediction"],values,"历史权重重载不同");same(z["raw_prediction"],raw,"历史未裁剪值不同")
            maximum=float(np.max(np.abs(raw[probe])));passed=maximum<=limit
            diagnostics[identity]=(maximum,passed,raw)
            if passed:
                load_module(folder/"source.py");model=joblib.load(folder/"model.joblib")
                raw_service,values,bad=predict(model,features[service],limit,parent_base[service])
                with np.load(folder/"service/base.npz") as z:
                    same(z["ids"],service,"服务行不同");same(z["prediction"],values,"服务权重重载不同");same(z["raw_prediction"],raw_service,"服务原预测不同")
                if [v["id"] for v in read(folder/"service/prediction-failures.json")]!=list(map(int,service[bad])):raise ValueError("当前行回退不符")
                base[service]=values;fallback_rows+=len(bad)
            cache[identity]=base;model_count+=1
        candidates=[]
        for index,trial in enumerate(decision["trials"],1):
            if "source_sha256" not in trial:continue
            identity=trial["source_sha256"];base=cache[identity];setting=setting_of(trial["proposal"]["feedback"])
            measured=losses(base,windows,y,setting);same(trial["observed_rmse"],measured[:2],"探索成绩不同")
            for n,mask in enumerate(windows[:2],1):
                ix=np.flatnonzero(mask);residual=replay(base[ix],ix,y[ix],setting)-y[ix]
                observed=trial["remaining_errors"][f"window_{n}"]
                same(observed["mean_prediction_minus_answer"],float(residual.mean()),"试训偏差方向不同")
                for field,values in observed["below_above_median_errors"].items():
                    col=tools.x[ix,tools.names.index(field)];cut=float(np.median(col));low=col<=cut
                    expected=[round(cut,3),round(float(residual[low].mean()),3),round(float(residual[~low].mean()),3)]
                    if expected!=values:raise ValueError("试训分组说明不同")
            maximum,passed,raw=diagnostics[identity];reported=trial["historical_input_probe"]
            same(reported["prediction_abs_max"],maximum,"历史最大值不同");same(reported["allowed_abs_max"],limit,"历史上限不同")
            if reported["passed"]!=passed or reported["input_rows"]!=int(probe.sum()):raise ValueError("历史检查结果不同")
            probe_ids=ids[probe];probe_raw=raw[probe];order=np.argsort(-np.abs(probe_raw))[:3]
            expected=[{"time":str(scene.CLOCK[probe_ids[j]]),"input":features[probe_ids[j]].tolist(),
                       "prediction":float(probe_raw[j]),"answer_available":bool(np.isfinite(y[probe_ids[j]]))} for j in order]
            if reported["largest_predictions"]!=expected:raise ValueError("提供给驱动的异常读数不同")
            if passed:
                if "rejected" in trial:raise ValueError("有效候选意外被拒绝")
                candidates.append((identity,base,setting,measured))
            else:
                if "rejected" not in trial:raise ValueError("无效候选进入发布")
                probe_rejections.append({"site":site,"trial":index,"source":identity})
        pred,choices,frozen,initial,selected=initial_and_stream(parent_base,parent_setting,candidates,y,windows,service,returned)
        packet=read(dest/"deployment.json")
        if initial!=packet["initial"] or selected!=decision["selected"]:raise ValueError("开场采用不同")
        first=candidates[:1] if "rejected" not in decision["trials"][0] else []
        first_pred=initial_and_stream(parent_base,parent_setting,first,y,windows,service,returned)[0]
        # 原两案的检查独立从原权重计算，不直接相信事后诊断摘要中的通过名单。
        guarded=[]
        for index,trial in enumerate(read(old/site/"decision.json")["trials"] if has_guarded else [],1):
            if "observed_rmse" not in trial:continue
            folder=old/site/f"trial-{index}";load_module(folder/"source.py");model=joblib.load(folder/"model.joblib")
            raw,_,_=predict(model,tools.x[ids[probe],:9])
            if np.max(np.abs(raw))>limit:continue
            with np.load(folder/"base.npz") as z:base=np.zeros(len(y));base[z["ids"]]=z["prediction"]
            setting=setting_of(trial["proposal"]["feedback"])
            guarded.append((trial["source_sha256"],base,setting,losses(base,windows,y,setting)))
        guarded_pred=initial_and_stream(parent_base,parent_setting,guarded,y,windows,service,returned)[0] if has_guarded else None
        valid=np.isfinite(truth[service]);scored+=int(valid.sum());issued_rows+=len(service)
        # 事后只用于判断修错空间，使用了终局答案，绝不能作为合法交付成绩。
        clock=np.arange(np.flatnonzero(windows[2])[0],len(y));known=np.concatenate((y[clock[clock<service[0]]],returned))
        options=[(parent_base,parent_setting)]+[(b,s) for _,b,s,_ in candidates]+[(parent_base,s) for _,_,s,_ in candidates]
        columns=np.column_stack([replay(b[clock],clock,known,s)[clock>=service[0]] for b,s in options])
        own_errors=[error(truth[service][valid],v[valid]) for v in columns.T]
        weekly=np.empty(len(service))
        for start in range(0,len(service),7*24):
            block=slice(start,min(start+7*24,len(service)));mask=valid[block]
            winner=min(range(columns.shape[1]),key=lambda i:error(truth[service][block][mask],columns[block,i][mask]))
            weekly[block]=columns[block,winner]
        oracle={"site":site,"own_static":min(own_errors),"own_weekly":error(truth[service][valid],weekly[valid])}
        if has_guarded:guard_scores[site]=error(truth[service][valid],guarded_pred[valid])
        with np.load(dest/"issued.npz") as z:
            same(z["choices"],choices,"持续选择身份不同");same(z["project"],pred,"完整发放不同")
        with np.load(dest/"all-issued.npz") as z:
            attribution=[("first-only",first_pred),("frozen-initial",frozen)]
            if has_guarded:attribution.append(("guarded-original",guarded_pred))
            for key,expected in attribution:
                same(z[key],expected,"归因预测不同："+key)
            with np.load(original/site/"issued.npz") as previous:
                for now,before in (("previous-project","project"),("paper","paper"),("numeric","numeric"),("static","static")):
                    same(z[now],previous[before],"既有对手预测变化")
            with np.load(original/site/"fixed-winner.npz") as previous:same(z["previous-best-fixed"],previous["prediction"],"既有最强固定变化")
            if "prior-code-project" in z.files:
                with np.load(old/site/"issued.npz") as previous:
                    same(z["prior-code-project"],previous["project"],"原代码组发放变化")
                    same(z["first-only"],previous["first-only"],"重放首案的实际发放发生变化")
            for identity in result["source_ids"]:
                valid_source=identity in cache and diagnostics[identity][1]
                base=cache.get(identity)
                for setting in SETTINGS:
                    key=identity[:16]+"-"+name(setting)
                    direct=(stream([(base,setting)],0,y,windows,service,returned)[0] if valid_source else z["static"])
                    gate=valid_source and accepts(losses(parent_base,windows,y,parent_setting),losses(base,windows,y,setting))
                    same(z["direct-"+key],direct,"固定直接发放不同");same(z["fixed-"+key],direct if gate else z["static"],"固定门槛发放不同")
            for key in z.files:same(error(truth[service][valid],z[key][valid]),result["station_rmse"][key][site],"最终评分不同")
            pool=[key for key in z.files if key.startswith(tuple("direct-"+identity[:16]+"-" for identity in first_sources))]
            oracle["first_source_pool_static"]=min([result["station_rmse"][key][site] for key in pool]+[result["station_rmse"]["static"][site]])
            oracle_rows.append(oracle)
    means={k:float(np.mean(list(v.values()))) for k,v in result["station_rmse"].items()}
    for k,value in means.items():same(value,result["macro_rmse"][k],"平均分不同")
    controls=[k for k in means if k.startswith(("fixed-","direct-")) or k in ("static","numeric","previous-best-fixed")]
    if min(controls,key=means.get)!=result["best_fixed"]:raise ValueError("最强固定身份不同")
    audit={"new_luna_calls":calls,"first_proposals_replayed":first_replays,"models_reloaded":model_count,
           "issued":issued_rows,"scored":scored,"probe_rejections":probe_rejections,"current_row_fallbacks":fallback_rows,
           "historical_inputs_only_before_proposals":True,"full_predictions_and_fixed_controls_verified":True,
           "project":means["project"],"guarded_original":float(np.mean(list(guard_scores.values()))) if guard_scores else None,
           "reference_code_verified":editing,
           "shared_first_history_verified":sharing,"prior_code_project":means.get("prior-code-project"),
           "first_only":means["first-only"],"frozen_initial":means["frozen-initial"],
           "best_fixed":means[result["best_fixed"]],"paper":means["paper"],
           "hindsight_candidate_diagnostic":{"scope":"使用终局答案的事后修错空间，不是现场成绩或可发布策略",
               "per_site":oracle_rows,"macro":{key:float(np.mean([row[key] for row in oracle_rows]))
                    for key in ("own_static","own_weekly","first_source_pool_static")}},
           "scope":"开发同批；全部已保存候选和跨站固定权重重载逐行复核，未重新训练或重新提案"}
    atomic_json(out/"audit.json",audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
