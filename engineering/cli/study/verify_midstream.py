"""独立核对中途现场请求、合法训练、实际权重、确认与前后交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.study.verify_generated_delivery import replay
from engineering.cli.study.verify_generated_probe import predict, same
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs
from harness.integrations.candidate_edit import named_scores, resolve_edit
from harness.models.sparse_reference import setting_of
from harness.runtime.storage import atomic_json


def read(path):return json.loads(path.read_bytes())


def accepts(old,new):
    """回执来自JSON列表，先转数组再按事前两探索周和确认周条件独立判断。"""
    old,new=np.asarray(old),np.asarray(new)
    return bool(np.mean(new[:2]**2)<np.mean(old[:2]**2) and new[2]<old[2])


def plan_id(plan):
    return sha256(json.dumps({k:plan[k] for k in ("components","filters")},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def same_observation(actual, expected):
    """对象和时间身份严格相同；独立数值重构沿原same容差比较。"""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError("观察字段不符")
        for key in expected: same_observation(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise ValueError("观察条数不符")
        for a,b in zip(actual, expected, strict=True): same_observation(a,b)
    elif isinstance(expected, float):
        same(actual, expected, "观察分值不同")
    elif actual != expected:
        raise ValueError("观察对象或时间身份不同")


def main():
    ap=ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    args=ap.parse_args();out=args.run;scores=read(out/"scores.json");baseline=Path(scores["baseline"])
    if (out/"audit.json").exists():raise ValueError("不可覆盖中途核验")
    scene.configure(scores["start"]);tools=scene.prepare(args.archive,True,True);features,names=code_inputs(tools)
    registry=read(out/"plan-bank.json");reference=read(out/"reference-plan.json");expected_registry={}
    replay_records=read(out/"replay-provenance.json") if (out/"replay-provenance.json").exists() else []
    for entry in replay_records:
        original_request=read(Path(entry["source_request"]));response_path=Path(entry["source_response"])
        if sha256(response_path.read_bytes()).hexdigest()!=entry["source_response_sha256"]:raise ValueError("重放原回复变化")
        raw=response_path.read_text(encoding="utf-8");original_response,end=json.JSONDecoder().raw_decode(raw)
        if raw[end:].strip() not in ('',r'\n'):raise ValueError("重放吞掉了未定义内容")
        current_request=out/"queue/requests"/entry["task"]/Path(entry["source_request"]).name
        current_response=out/"queue/responses"/entry["task"]/Path(entry["source_request"]).name
        if read(current_request)!=original_request or read(current_response)!=original_response:raise ValueError("重放改变了原上下文或计划")
    for path in (out/"driver-original").glob('*.json'):
        site,arm,turn=path.stem.split('-');current,=list((out/"queue/responses"/('mid-'+arm)).glob(site+'-trial-'+turn+'-*.json'))
        if read(current)["plan"]!=read(path):raise ValueError("转交修改了Luna原始最终JSON")
    models=0;requests=0;rows=0;scored=0;changes={};all_scores={};training_count=0
    for site in scores["counts"]:
        dest=out/site;truth=scene.readings(args.archive,site).to_numpy();old_y,_,_,service,returned=scene.roles(truth)
        start=int(service[0]);at=start+28*24;labels=np.r_[old_y[:start],returned];clock=np.arange(len(labels))
        fit=(clock<at-21*24)&np.isfinite(labels);history=np.arange(at-21*24-6,at);future=np.arange(at,len(labels))
        windows=[(clock>=at+a*24)&(clock<at+(a+7)*24)&np.isfinite(labels) for a in (-21,-14,-7)]
        with np.load(baseline/site/"best-fixed.npz") as z:original=z["prediction"].copy();same(z["ids"],service,"起点时钟不同")
        parent=[float(np.sqrt(np.mean((original[np.flatnonzero(m)-start]-labels[m])**2))) for m in windows]
        timing=read(dest/"timing.json");same(timing["parent_errors"],parent,"父版真实交付观察不同")
        if timing["maintenance_hour"]!=at or timing["training_last_hour"]!=int(np.flatnonzero(fit)[-1]):raise ValueError("维护时序不同")
        records={};outputs={}
        for record_path in dest.rglob("training.json"):
            folder=record_path.parent;plan=read(folder/"plan.json");record=read(record_path);key=plan_id(plan)
            selected=fit.copy()
            for f in plan["filters"]:
                values=features[:,names.index(f["feature"])];selected &= (values<=f["threshold"]) if f["operator"]=="<=" else (values>f["threshold"])
            ids=np.flatnonzero(selected)
            if len(ids)<12 or ids[-1]>=at-21*24:raise ValueError("训练用了探索或未来答案")
            with np.load(folder/"training-input.npz") as z:
                same(z["train_ids"],ids,"训练行不同");same(z["train_x"],features[selected],"训练输入不同")
                same(z["train_y"],labels[selected],"训练目标不同");same(z["predict_ids"],history,"历史时钟不同")
                same(z["predict_x"],features[history],"历史输入不同")
            limit=1000.*max(1.,float(np.max(np.abs(labels[selected]))));same(record["prediction_limit"],limit,"量级阈值不同")
            combined=np.zeros(len(history));later=np.zeros(len(future));has_future=(folder/"future-issued.npz").exists();pieces=[]
            if has_future:
                with np.load(folder/"future-input.npz") as z:
                    if set(z.files)!={"predict_x","predict_ids","fallback","prediction_limit"}:raise ValueError("未来输入存在标签")
                    same(z["predict_x"],features[future],"未来推理输入不同");same(z["predict_ids"],future,"未来时钟不同")
                    same(z["fallback"],original[28*24:],"回退版不同")
            for i,c in enumerate(plan["components"]):
                unit=folder/f"component-{i}";source=(unit/"source.py").read_text(encoding="utf-8");weights=unit/"model.joblib"
                if source!=c["python_source"] or sha256(weights.read_bytes()).hexdigest()!=record["components"][i]["weights_sha256"]:
                    raise ValueError("训练代码或权重身份不同")
                load_module(unit/"source.py");model=joblib.load(weights)
                raw,base,bad=predict(model,features[history],limit,np.zeros(len(history)))
                if bad:raise ValueError("历史输入数值失败")
                with np.load(unit/"base.npz") as z:same(z["prediction"],base,"历史重载预测不同");same(z["raw_prediction"],raw,"历史原始值不同")
                corrected=replay(base,history,labels[history],setting_of(c["feedback"]))
                combined+=c["weight"]*corrected;pieces.append(corrected)
                if has_future:
                    model=joblib.load(weights);raw,values,bad=predict(model,features[future],limit,original[28*24:])
                    with np.load(unit/"service/base.npz") as z:same(z["prediction"],values,"未来重载预测不同");same(z["raw_prediction"],raw,"未来原始值不同")
                    if len(bad)!=len(read(unit/"service/prediction-failures.json")):raise ValueError("推理回退记录不同")
                    stream=replay(np.r_[base,values],np.r_[history,future],labels[np.r_[history,future]],setting_of(c["feedback"]))
                    later+=c["weight"]*stream[len(history):]
                models+=1
            losses=[float(np.sqrt(np.mean((combined[m[history]]-labels[history][m[history]])**2))) for m in windows]
            same(losses,record["history_rmse"],"探索或确认评分不同")
            if scores.get("edit_contract"):
                ablations=[]
                if len(pieces)>1:
                    for i,c in enumerate(plan["components"]):
                        ablated=(combined-c["weight"]*pieces[i])/(1.-c["weight"])
                        values=[float(np.sqrt(np.mean((ablated[m[history]]-labels[history][m[history]])**2))) for m in windows[:2]]
                        ablations.append({"removed_component":i,"remaining_weights_renormalized":True,
                            "scores":named_scores(values,windows[:2],history,prediction_object="ensemble_without_component_"+str(i)+"_after_feedback")})
                same_observation(record["component_ablations"],ablations)
            with np.load(folder/"historical-issued.npz") as z:same(z["prediction"],combined,"历史组合不同")
            if has_future:
                with np.load(folder/"future-issued.npz") as z:same(z["prediction"],later,"交接组合不同")
                outputs[str(folder.relative_to(dest))]=later
            records[str(folder.relative_to(dest))]=(key,plan,losses);training_count+=1
        expected={"baseline":original.copy()};refit=records["refit"]
        expected["refit-direct"]=np.r_[original[:28*24],outputs["refit"]]
        expected["refit-gate"]=expected["refit-direct"] if accepts(parent,refit[2]) else original.copy()
        observed=np.flatnonzero((clock>=start)&(clock<at-7*24)&np.isfinite(labels))
        for arm in ("project","paper"):
            decision=read(dest/(arm+"-decision.json"));candidates=["refit"]
            for turn,trial in enumerate(decision["trials"],1):
                request_path,=list((out/"queue/requests"/('mid-'+arm)).glob(f"{site}-trial-{turn}-*.json"))
                request=read(request_path);response=read(out/"queue/responses"/('mid-'+arm)/request_path.name);context=request["context"]
                if response["request_id"]!=request["request_id"] or response["plan"]!=trial["proposal"]:raise ValueError("原始现场提案不同")
                if context["trials"]!=decision["trials"][:turn-1]:raise ValueError("下一请求没有使用真实上次反馈")
                if context["reference_plan"]!=reference or context["feature_names"]!=names:raise ValueError("起点和输入改变")
                if scores.get("edit_contract"):
                    if context["reference_identity"]!=plan_id(reference):raise ValueError("编辑父版身份改变")
                    same_observation(context["parent_scores"],named_scores(parent[:2],windows[:2],history,prediction_object="deployed_parent_ensemble_after_feedback"))
                    same_observation(context["refit_observation"]["scores"],named_scores(refit[2][:2],windows[:2],history,prediction_object="complete_refit_ensemble_after_feedback"))
                    if context["refit_observation"]["component_ablations"]!=read(dest/"refit/training.json")["component_ablations"]:
                        raise ValueError("驱动没有收到实际消融")
                else:
                    same(context["parent_history_rmse"],parent[:2],"向驱动披露了额外父版评分")
                    same(context["refit_observation"]["rmse"],refit[2][:2],"补训观察含确认或不同分数")
                supplied=context["legal_service_observations"]
                if [v["hour"] for v in supplied]!=observed.tolist():raise ValueError("驱动可见答案时钟不同")
                for row in supplied:
                    hour=row["hour"];same(row["answer"],labels[hour],"驱动可见目标不同")
                    same(row["issued_prediction"],original[hour-start],"驱动看到的不是原实际交付")
                    same(row["raw_inputs"],features[hour,:9].round(5),"可见输入不同")
                requests+=1
                if trial.get("stopped"):
                    if turn!=len(decision["trials"]):raise ValueError("停止后继续调用")
                    continue
                if "rejected" in trial:continue
                name=str(Path(arm)/f"trial-{turn}");key,plan,losses=records[name];candidates.append(name)
                if key!=trial["identity"]:raise ValueError("方案身份不同")
                if scores.get("edit_contract"):
                    same_observation(trial["scores"],named_scores(losses[:2],windows[:2],history,prediction_object="complete_ensemble_after_feedback"))
                    resolved,delta=resolve_edit(trial["proposal"],reference)
                    if resolved!=plan or delta!=trial["actual_change"]:raise ValueError("实际编辑与提案不符")
                    if trial["component_ablations"]!=read(dest/name/"training.json")["component_ablations"]:
                        raise ValueError("试训返回的消融不同")
                else:
                    same(trial["observed_rmse"],losses[:2],"试训反馈不是两探索周")
                if arm=="paper" and not scores.get("edit_contract"):
                    for c,r in zip(plan["components"],reference["components"],strict=True):
                        if c["python_source"]!=r["python_source"] or c["weight"]!=r["weight"]:raise ValueError("论文组改变了学习器或配比")
            best=min(candidates,key=lambda name:np.mean(np.square(records[name][2][:2])))
            if best!=decision["nominated"] or accepts(parent,records[best][2])!=decision["adopted"]:raise ValueError("选案使用了其他信息")
            expected[arm]=original.copy()
            if decision["adopted"]:expected[arm][28*24:]=outputs[best]
            direct_nominee=np.r_[original[:28*24],outputs[best]]
            terminal=truth[service];valid_terminal=np.isfinite(terminal)
            changes.setdefault(site,{})[arm]={"nominated":best,"adopted":decision["adopted"],
                "changed_after_maintenance":int(np.count_nonzero(np.abs(expected[arm][28*24:]-original[28*24:])>1e-10)),
                "differs_from_refit_gate":int(np.count_nonzero(np.abs(expected[arm]-expected["refit-gate"])>1e-10)),
                "nominated_direct_hindsight_rmse":float(np.sqrt(np.mean((direct_nominee[valid_terminal]-terminal[valid_terminal])**2))),
                "diagnostic_scope":"事后检查已提名但可能被拒绝的模型，不是实际现场成绩"}
            for name in candidates:
                key,plan,_=records[name];expected_registry.setdefault(key,{"plan":plan,"origin_site":site,"origin_arm":arm,"origin_folder":name})
        valid=np.isfinite(truth[service]);rows+=len(service);scored+=int(valid.sum())
        with np.load(dest/"issued.npz") as z:
            same(z["ids"],service,"完整服务时钟不同");same(z["returned"],returned,"反馈不同")
            for key,value in expected.items():
                same(z[key],value,"完整交接不同");same(value[:28*24],original[:28*24],"回改了前半段")
                all_scores.setdefault(key,{})[site]=float(np.sqrt(np.mean((value[valid]-truth[service][valid])**2)))
        failures=read(dest/"fixed-failures.json")
        with np.load(dest/"fixed-issued.npz") as z:
            for key,entry in registry.items():
                if key in failures:direct=original.copy();gated=original.copy()
                else:
                    name=next(n for n,(k,_,_) in records.items() if k==key and n in outputs)
                    direct=np.r_[original[:28*24],outputs[name]];gated=direct if accepts(parent,records[name][2]) else original.copy()
                for prefix,value in (("direct-",direct),("gated-",gated)):
                    same(z[prefix+key],value,"固定方案交付不同")
                    all_scores.setdefault(prefix+key,{})[site]=float(np.sqrt(np.mean((value[valid]-truth[service][valid])**2)))
        if scores.get("prior_run"):
            previous=Path(scores["prior_run"])
            prior_scores=read(previous/"scores.json")
            if prior_scores["baseline"]!=scores["baseline"] or prior_scores["start"]!=scores["start"]:
                raise ValueError("复用固定对照不属于同一起点")
            with np.load(previous/site/"fixed-issued.npz") as z:
                same(z["ids"],service,"历史固定对照时钟不同")
                for key in z.files:
                    if key=="ids":continue
                    same(z[key][:28*24],original[:28*24],"历史固定改动了前半段")
                    all_scores.setdefault("prior-"+key,{})[site]=float(np.sqrt(np.mean((z[key][valid]-truth[service][valid])**2)))
    if registry!=expected_registry:raise ValueError("固定库混入未产生的提案")
    if requests!=scores["field_requests"]:raise ValueError("现场请求计数不同")
    for key,values in all_scores.items():
        for site,value in values.items():same(value,scores["station_rmse"][key][site],"评分不同")
        same(np.mean(list(values.values())),scores["macro_rmse"][key],"汇总不同")
    fixed=[k for k in all_scores if k.startswith(("direct-","gated-","prior-")) or k in ("baseline","refit-direct","refit-gate")]
    if min(fixed,key=lambda k:np.mean(list(all_scores[k].values())))!=scores["best_fixed"]:raise ValueError("非最强固定对照")
    review=[]
    for site,arms in changes.items():
        for arm,detail in arms.items():
            score=all_scores[arm][site];old=all_scores["baseline"][site];refit_score=all_scores["refit-gate"][site]
            review.append({"site":site,"arm":arm,**detail,"full_rmse":score,"baseline_rmse":old,
                           "same_data_refit_rmse":refit_score,"improvement_over_baseline_percent":100*(1-score/old),
                           "improvement_over_same_data_refit_percent":100*(1-score/refit_score),
                           "new_proposal_actually_published":bool(detail["adopted"] and detail["nominated"]!="refit")})
    atomic_json(out/"decision-review.json",{"scope":"完整56日，前半段相同；新提案与同资料补训收益分开，百分比不相加", "rows":review})
    audit={"issued":rows,"scored":scored,"models_reloaded":models,"training_plans":training_count,"field_requests":requests,
           "identical_requests_replayed":len(replay_records),"new_requests":requests-len(replay_records),
           "replay_and_relay_originals_verified":True,
           "legal_training_and_request_context_verified":True,"first_half_unchanged":True,"independent_handoff_verified":True,
           "all_fixed_and_scores_verified":True,"changes":changes,
           "project":scores["macro_rmse"]["project"],"paper":scores["macro_rmse"]["paper"],
           "refit_gate":scores["macro_rmse"]["refit-gate"],"best_fixed":scores["macro_rmse"][scores["best_fixed"]]}
    atomic_json(out/"audit.json",audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
