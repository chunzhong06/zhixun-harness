"""从生成源码、已发布权重及原始回执独立核对完整发放，不重新提案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.study.verify_london_feedback import replay
from harness.contracts import digest
from harness.integrations.generated_regressor import load_module
from harness.models.sparse_reference import setting_of
from harness.models.station_residual import Rule,error
from harness.runtime.storage import atomic_json


def read(path):return json.loads(path.read_bytes())


def losses(base,windows,y,setting):
    return np.asarray([error(y[ix],replay(base[ix],ix,y[ix],setting)) for ix in (np.flatnonzero(mask) for mask in windows)])


def accepts(old,new):return np.mean(new[:2]**2)<np.mean(old[:2]**2) and new[2]<old[2]


def stream(options,initial,y,windows,service,returned):
    clock=np.arange(np.flatnonzero(windows[2])[0],len(y));known=np.concatenate((y[clock[clock<service[0]]],returned))
    columns=np.column_stack([replay(base[clock],clock,known,setting) for base,setting in options])
    selected=initial;history=[];pred=[];choices=[]
    for i,now in enumerate(clock):
        pred.append(columns[i,selected]);choices.append(selected)
        if np.isfinite(known[i]):
            history.append((now,(columns[i]-known[i])**2));history=[v for v in history if v[0]>now-28*24]
            if now>=service[0]:selected=int(np.argmin(np.mean([v[1] for v in history],axis=0)))
    use=clock>=service[0]
    return np.asarray(pred)[use],np.asarray(choices)[use],columns[use,initial]


def main():
    ap=ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--archive",type=Path,required=True)
    args=ap.parse_args();root=args.run;result=read(root/"scores.json")
    if (root/"audit.json").exists():raise ValueError("不可覆盖独立核验")
    scene.configure(result["start"]);tools=scene.prepare(args.archive,True,True)
    tools.static=Rule(**result["initial_spec"]["recipe"]);parent_setting=setting_of(result["initial_spec"]["feedback"])
    maximum=0.;requests=0;checked_models=0;checked_predictions=0;scored=0;first_scores={};first_valid_scores={};frozen_scores={}
    best=result["best_fixed"];original=Path(result["original"]);source_registry={};behaviour=[]
    for station in result["counts"]:
        dest=root/station;truth=scene.readings(args.archive,station).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
        valid=np.isfinite(truth[service]);scored+=int(valid.sum())
        expected_ids=np.flatnonzero(np.logical_or.reduce(windows)|(np.arange(len(y))>=service[0]))
        with np.load(dest/"training-input.npz") as data:
            for key,expected in (("train_x",tools.x[fit,:9]),("train_y",y[fit]),("predict_ids",expected_ids),("predict_x",tools.x[expected_ids,:9])):
                if not np.array_equal(data[key],expected):raise ValueError("训练或逐时输入与合法时序不符")
        parent,_=joblib.load(dest/"initial.joblib");parent_base=tools.issue(parent,np.arange(len(y)))
        parent_loss=losses(parent_base,windows,y,parent_setting)
        memo={};candidates=[];dec=read(dest/"decision.json");packet=read(dest/"deployment.json")
        for i,trial in enumerate(dec["trials"],1):
            qpath,=list((root/"queue/requests/code").glob(f"{station}-trial-{i}-*.json"))
            q=read(qpath);a=read(root/"queue/responses/code"/qpath.name)
            body={"model":"gpt-6-luna","messages":[{"role":"system","content":q["instruction"]},
                  {"role":"user","content":json.dumps(q["context"],ensure_ascii=False,allow_nan=False)}]}
            if digest(body)!=q["request_id"] or a["request_id"]!=q["request_id"] or a["plan"]!=trial["proposal"]:
                raise ValueError("提案与请求身份不同")
            requests+=1
            if "observed_rmse" not in trial:continue
            source=trial["proposal"]["python_source"];identity=sha256(source.encode()).hexdigest()
            if identity!=trial["source_sha256"]:raise ValueError("发布源码身份不同")
            source_registry[identity]=source;setting=setting_of(trial["proposal"]["feedback"])
            if identity not in memo:
                folder=dest/f"trial-{i}"
                if (folder/"source.py").read_text(encoding="utf-8")!=source:raise ValueError("发布源码不是原提案")
                load_module(folder/"source.py");model=joblib.load(folder/"model.joblib")
                prediction=np.asarray([max(0.,float(np.asarray(model.predict(x[None,:]))[0])) for x in tools.x[expected_ids,:9]])
                with np.load(folder/"base.npz") as archive:
                    if not np.array_equal(archive["ids"],expected_ids):raise ValueError("发布预测时钟不同")
                    delta=float(np.max(np.abs(prediction-archive["prediction"])));maximum=max(maximum,delta)
                    if delta>1e-10:raise ValueError("生成模型重载后预测不同")
                base=np.zeros(len(y));base[expected_ids]=prediction;memo[identity]=base;checked_models+=1
                # 只用开场前两个探索周的输入，检查稀疏有答案时刻之外的数值行为。
                probe=np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&
                                    (scene.CLOCK<scene.START-np.timedelta64(7,'D')))
                probe_prediction=np.asarray([float(np.asarray(model.predict(x[None,:]))[0]) for x in tools.x[probe,:9]])
                info={"site":station,"trial":i,"source":identity,"historical_probe_rows":len(probe),
                      "historical_prediction_max_abs":float(np.max(np.abs(probe_prediction))),
                      "training_target_max_abs":float(np.max(np.abs(y[fit]))),
                      "label_hours":sorted(set(map(int,scene.CLOCK[fit].hour)))}
                fitted=getattr(model,"model_",model)
                steps=getattr(fitted,"named_steps",{})
                if "standardscaler" in steps:
                    scale=np.asarray(steps["standardscaler"].scale_)
                    info["minimum_scale"]=float(scale.min());tiny=np.flatnonzero(scale<1e-8)
                    if "polynomialfeatures" in steps:
                        names=steps["polynomialfeatures"].get_feature_names_out(tools.names[:9])
                        info["near_constant_features"]=[str(names[j]) for j in tiny]
                behaviour.append(info)
            base=memo[identity];measured=losses(base,windows,y,setting)
            if not np.allclose(measured[:2],trial["observed_rmse"],rtol=0,atol=1e-10):raise ValueError("试训总分不同")
            for n,mask in enumerate(windows[:2],1):
                ix=np.flatnonzero(mask);prediction=replay(base[ix],ix,y[ix],setting);err=prediction-y[ix]
                observed=trial["remaining_errors"][f"window_{n}"]
                if abs(observed["mean_prediction_minus_answer"]-float(err.mean()))>1e-10:raise ValueError("试训错误方向不同")
                for field,values in observed["below_above_median_errors"].items():
                    col=tools.x[ix,tools.names.index(field)];cut=float(np.median(col));low=col<=cut
                    expected=[round(cut,3),round(float(err[low].mean()),3),round(float(err[~low].mean()),3)]
                    if expected!=values:raise ValueError("分组试训反馈不符")
            candidates.append((identity,base,setting,measured))
        selected=min(range(len(candidates)),key=lambda i:np.mean(candidates[i][3][:2]**2)) if candidates else None
        initial=selected+1 if selected is not None and accepts(parent_loss,candidates[selected][3]) else 0
        if initial!=packet["initial"] or selected!=dec["selected"]:raise ValueError("初次采用决定不同")
        options=[(parent_base,parent_setting)]+[(b,s) for _,b,s,_ in candidates]+[(parent_base,s) for _,_,s,_ in candidates]
        pred,choices,frozen=stream(options,initial,y,windows,service,returned)
        with np.load(dest/"issued.npz") as archive:
            if not np.array_equal(archive["choices"],choices):raise ValueError("持续选择身份不同")
            delta=float(np.max(np.abs(pred-archive["project"])));maximum=max(maximum,delta)
            if delta>1e-10:raise ValueError("完整发放重放不一致")
        first=next(iter(dec["trials"]));first_pred=stream([(parent_base,parent_setting)],0,y,windows,service,returned)[0]
        if "source_sha256" in first:
            base=memo[first["source_sha256"]];setting=setting_of(first["proposal"]["feedback"])
            first_pred=stream([(parent_base,parent_setting),(base,setting),(parent_base,setting)],
                              int(accepts(parent_loss,losses(base,windows,y,setting))),y,windows,service,returned)[0]
        first_scores[station]=error(truth[service][valid],first_pred[valid]);frozen_scores[station]=error(truth[service][valid],frozen[valid])
        first_valid=stream([(parent_base,parent_setting)],0,y,windows,service,returned)[0]
        if candidates:
            _,base,setting,measured=candidates[0]
            first_valid=stream([(parent_base,parent_setting),(base,setting),(parent_base,setting)],
                               int(accepts(parent_loss,measured)),y,windows,service,returned)[0]
        first_valid_scores[station]=error(truth[service][valid],first_valid[valid])
        with np.load(dest/"all-issued.npz") as archive:
            for key in archive.files:
                score=error(truth[service][valid],archive[key][valid]);delta=abs(score-result["station_rmse"][key][station]);maximum=max(maximum,delta)
                if delta>1e-10:raise ValueError("保存的预测与最终评分不同")
            for key,expected in (("first-only",first_pred),("frozen-initial",frozen)):
                if not np.allclose(archive[key],expected,rtol=0,atol=1e-10):raise ValueError("归因发放不同")
            with np.load(original/station/"issued.npz") as old:
                for now,before in (("previous-project","project"),("paper","paper"),("numeric","numeric"),("static","static")):
                    if not np.array_equal(archive[now],old[before]):raise ValueError("旧对手复用预测发生变化")
            checked_predictions+=len(service)
    means={key:float(np.mean(list(v.values()))) for key,v in result["station_rmse"].items()}
    if any(abs(means[k]-v)>1e-10 for k,v in result["macro_rmse"].items()):raise ValueError("汇总得分不同")
    controls=[k for k in means if k.startswith(("fixed-","direct-")) or k in ("static","numeric","previous-best-fixed")]
    if min(controls,key=means.get)!=best:raise ValueError("固定赢家身份不同")
    audit={"replay_max_abs":maximum,"requests":requests,"published_models_reloaded":checked_models,
           "issued":checked_predictions,"scored":scored,"first_only":float(np.mean(list(first_scores.values()))),
           "first_executable_only":float(np.mean(list(first_valid_scores.values()))),
           "frozen_initial":float(np.mean(list(frozen_scores.values()))),"best_fixed":best,
           "candidate_observations_verified":True,"previous_controls_unchanged":True,
           "historical_input_diagnostics":behaviour,
           "scope":"生成候选已重载；全部保存的固定发放已重新计分，未对每项跨站固定权重再训练"}
    atomic_json(root/"audit.json",audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
