"""从首案原件、发布权重及原始回执独立重构首案池的全部选择。"""
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
from engineering.cli.study.verify_generated_probe import predict,same
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs
from harness.models.station_residual import Rule,error
from harness.models.sparse_reference import SETTINGS,setting_of,name,correct,choose_from_feedback
from harness.runtime.storage import atomic_json


def read(path):return json.loads(path.read_bytes())


def main():
    ap=ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    args=ap.parse_args();out=args.run;result=read(out/"scores.json")
    if (out/"audit.json").exists():raise ValueError("不可覆盖池核验")
    generated=Path(result["generated"]);current=Path(result["current"]);registry=read(out/"source-pool.json")
    expected={}
    for site in result["counts"]:
        first=read(generated/site/"decision.json")["trials"][0]
        if "rejected" not in first and "source_sha256" in first:
            expected.setdefault(first["source_sha256"],{"origin_site":site,"python_source":first["proposal"]["python_source"],"proposal":first["proposal"]})
    if registry!=expected or list(registry)!=result["source_ids"]:raise ValueError("池中出现非首案或来源不符")
    scene.configure(result["start"]);tools=scene.prepare(args.archive,True,True)
    tools.static=Rule(**result["initial_spec"]["recipe"]);features,feature_names=code_inputs(tools)
    parent_setting=setting_of(result["initial_spec"]["feedback"]);models=0;issued_rows=0;scored=0;diagnostics=[]
    for site in result["counts"]:
        dest=out/site;deployment=read(dest/"deployment.json")
        truth=scene.readings(args.archive,site).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
        history=np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&(scene.CLOCK<scene.START))
        ids=np.r_[history,service];limit=1000.*max(1.,float(np.max(np.abs(y[fit]))))
        if (dest/"initial.joblib").read_bytes()!=(generated/site/"initial.joblib").read_bytes():raise ValueError("起点权重被更换")
        parent,_=joblib.load(dest/"initial.joblib");parent_base=tools.issue(parent,np.arange(len(y)))
        with np.load(dest/"prediction-input.npz") as z:
            if set(z.files)!={"predict_x","predict_ids","fallback","prediction_limit"}:raise ValueError("只推理输入包含额外字段")
            for key,value in (("predict_x",features[ids]),("predict_ids",ids),("fallback",parent_base[ids]),("prediction_limit",limit)):
                same(z[key],value,"推理输入不符："+key)
        bases={"initial":parent_base}
        if len(deployment["published"])!=len(registry):raise ValueError("发布权重数量不同")
        for item in deployment["published"]:
            identity=item["source"];folder=dest/identity;weights=Path(item["weights"])
            if not weights.resolve().is_relative_to((generated/site).resolve()):raise ValueError("权重不在本站原训练记录内")
            if sha256(weights.read_bytes()).hexdigest()!=item["weights_sha256"]:raise ValueError("权重身份不同")
            source=(folder/"source.py").read_text(encoding="utf-8")
            if source!=registry[identity]["python_source"] or sha256(source.encode()).hexdigest()!=identity:raise ValueError("重载代码不是首案")
            load_module(folder/"source.py");model=joblib.load(weights)
            raw,values,bad=predict(model,features[ids],limit,parent_base[ids])
            if bad or read(folder/"prediction-failures.json"):raise ValueError("出现未说明的逐行回退")
            with np.load(folder/"base.npz") as z:
                same(z["ids"],ids,"权重预测时钟不同");same(z["prediction"],values,"重载预测不同");same(z["raw_prediction"],raw,"原预测不同")
            base=np.zeros(len(y));base[ids]=values;bases[identity]=base;models+=1
        options=[];pairs=[];errors=[]
        for identity,base in bases.items():
            for setting in SETTINGS:
                options.append({"source":identity,"feedback":list(setting),"name":identity+"-"+name(setting)})
                pairs.append((base,setting));errors.append(losses(base,windows,y,setting))
        errors=np.asarray(errors)
        if options!=deployment["options"] or feature_names!=deployment["feature_names"]:raise ValueError("候选顺序或输入名字不同")
        same(errors,deployment["history_rmse"],"历史评价不同")
        parent_index=next(i for i,v in enumerate(options) if v["source"]=="initial" and tuple(v["feedback"])==parent_setting)
        best=int(np.argmin(np.mean(errors[:,:2]**2,axis=1)))
        initial=best if accepts(errors[parent_index],errors[best]) else parent_index
        if (parent_index,best,initial)!=(deployment["parent_index"],deployment["history_best"],deployment["initial"]):raise ValueError("历史选案或确认不同")
        online,choices,frozen=stream(pairs,initial,y,windows,service,returned)
        clock=np.arange(np.flatnonzero(windows[2])[0],len(y));known=np.concatenate((y[clock[clock<service[0]]],returned))
        columns=np.column_stack([replay(base[clock],clock,known,setting) for base,setting in pairs]);use=clock>=service[0]
        # 实际选择器也须通过未来答案扰动：第28天的答案变化不能改写此前及当时预测。
        cut=28*24;shifted=known.copy();after=clock>=service[cut];shifted[after&np.isfinite(shifted)]+=137.
        changed_columns=np.column_stack([correct(base[clock,None],shifted,clock,setting)[:,0] for base,setting in pairs])
        changed,changed_choices=choose_from_feedback(changed_columns,shifted,clock,service[0],initial,28)
        same(changed[use][:cut+1],online[:cut+1],"未来答案改变了既往输出")
        if not np.array_equal(changed_choices[use][:cut+1],choices[:cut+1]):raise ValueError("未来答案改变了既往采用身份")
        valid=np.isfinite(truth[service]);reference=valid&np.isfinite(returned);issued_rows+=len(service);scored+=int(valid.sum())
        with np.load(dest/"issued.npz") as z:
            same(z["ids"],service,"交付时钟不同");same(z["returned"],returned,"回标时序不同")
            if not np.array_equal(z["choices"],choices):raise ValueError("持续选择身份不同")
            for key,value in (("pool-online",online),("pool-history-direct",columns[use,best]),("pool-history-gate",frozen),("static",columns[use,parent_index])):
                same(z[key],value,"完整交付不同："+key)
            for i,option in enumerate(options):
                key="bank-"+option["name"];same(z[key],columns[use,i],"固定发放不同")
                gated=columns[use,i] if accepts(errors[parent_index],errors[i]) else columns[use,parent_index]
                same(z["gate-"+key],gated,"固定确认发放不同")
            with np.load(generated/site/"issued.npz") as old:
                for now,before in (("previous-code","project"),("first-only","first-only"),("previous-project","previous-project"),
                                   ("paper","paper"),("numeric","numeric"),("previous-best-fixed","previous-best-fixed")):
                    same(z[now],old[before],"原对手预测变化")
            with np.load(current/site/"issued.npz") as old:same(z["current-code"],old["project"],"当前代码候选预测变化")
            for key in result["station_rmse"]:same(error(truth[service][valid],z[key][valid]),result["station_rmse"][key][site],"最终评分不同")
        full=[error(truth[service][valid],columns[use,i][valid]) for i in range(len(options))]
        partial=[error(truth[service][reference],columns[use,i][reference]) for i in range(len(options))]
        a,b=int(np.argmin(full)),int(np.argmin(partial))
        diagnostics.append({"site":site,"full_best":options[a]["name"],"reference_best":options[b]["name"],
                            "full_oracle":full[a],"reference_winner_full_rmse":full[b]})
        original_diagnostic=next(v for v in result["hindsight_diagnostics"] if v["site"]==site)
        if original_diagnostic["full_hindsight_best"]!=options[a]["name"] or original_diagnostic["reference_hindsight_best"]!=options[b]["name"]:
            raise ValueError("事后诊断赢家不同")
    means={key:float(np.mean(list(values.values()))) for key,values in result["station_rmse"].items()}
    for key,value in means.items():same(value,result["macro_rmse"][key],"汇总评分不同")
    fixed=[key for key in means if key.startswith(("bank-","gate-bank-")) or key in ("static","numeric","previous-best-fixed")]
    if min(fixed,key=means.get)!=result["best_fixed"] or min([*fixed,"pool-history-direct","pool-history-gate"],key=means.get)!=result["best_simple"]:
        raise ValueError("强对手身份不同")
    audit={"published_models_reloaded":models,"issued":issued_rows,"scored":scored,"new_training":0,"new_luna_calls":0,
           "source_pool_only_first_proposals":True,"history_and_choices_verified":True,"future_answers_do_not_change_past":True,
           "all_controls_and_scores_verified":True,"online_pool":means["pool-online"],"history_direct":means["pool-history-direct"],
           "history_gate":means["pool-history-gate"],"best_fixed":means[result["best_fixed"]],"best_simple":means[result["best_simple"]],
           "hindsight_diagnostic":{"scope":"事后诊断，不是现场成绩","rows":diagnostics}}
    atomic_json(out/"audit.json",audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
