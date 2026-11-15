"""真实重训配对核对已有配方与可编辑源码；只验证表达等价，不计现场新发现。"""
from argparse import ArgumentParser
from dataclasses import replace,asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.study.verify_london_feedback import replay
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs,source_for_rule
from harness.models.station_residual import Rule,error
from harness.models.sparse_reference import setting_of
from harness.runtime.storage import LocalStore,atomic_json


def main():
    ap=ArgumentParser();ap.add_argument("--archive",required=True,type=Path);ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path);args=ap.parse_args()
    original=json.loads((args.original/"scores.json").read_bytes());scene.configure(original["start"])
    tools=scene.prepare(args.archive,True,True);x,names=code_inputs(tools)
    parent=Rule(**original["initial_spec"]["recipe"])
    best_id=int(original["best_fixed"].split("-")[1]);best=Rule(**original["recipes"][best_id])
    cases={"initial":parent,"known-best":best,"bias":replace(best,correction="bias",days=60),
           "products":replace(best,features=tuple(tools.feature_terms)),
           "filtered-tree":replace(best,days=60,where="BQ9:low",limit=20.,learner="ExtraTreesRegressor")}
    with LocalStore(args.output).exclusive():
        if (args.output/"scores.json").exists():raise ValueError("不可覆盖等价实测")
        atomic_json(args.output/"cases.json",{key:asdict(rule) for key,rule in cases.items()})
        scores={};rows=[];maximum=0.;fits=0
        for site in original["counts"]:
            truth=scene.readings(args.archive,site).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
            historical=np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&(scene.CLOCK<scene.START))
            dest=args.output/site;dest.mkdir();data=dest/"historical-input.npz"
            np.savez_compressed(data,train_x=x[fit],train_y=y[fit],predict_x=x[historical],predict_ids=historical)
            clock=np.arange(np.flatnonzero(windows[2])[0],len(y))
            visible=np.concatenate((y[clock[clock<service[0]]],returned))
            for key,rule in cases.items():
                folder=dest/key;source=source_for_rule(tools,rule)
                old=tools.train(rule,fit,y);old_base=tools.issue(old,np.arange(len(y)));fits+=1
                ids,prediction=fit_source(source,data,folder);base=np.zeros(len(y));base[ids]=prediction;fits+=1
                joblib.dump(old,folder/"original.joblib");load_module(folder/"source.py");model=joblib.load(folder/"model.joblib")
                selected=np.flatnonzero(fit);selected=selected[selected>=selected[-1]-24*rule.days+1]
                if rule.where!="all":
                    field,side=rule.where.split(":");values=tools.x[selected,tools.names.index(field)]
                    selected=selected[values<=rule.limit] if side=="low" else selected[values>rule.limit]
                if not np.array_equal(model.training_hours_,selected):raise ValueError("源码选择的训练行或真实时间不同")
                weights_before=sha256((folder/"model.joblib").read_bytes()).hexdigest()
                service_data=folder/"service-input.npz"
                np.savez_compressed(service_data,predict_x=x[service],predict_ids=service,fallback=old_base[service],
                                    prediction_limit=1000.*max(1.,float(np.max(np.abs(y[fit])))))
                ids,prediction=fit_source(source,service_data,folder/"service",folder/"model.joblib");base[ids]=prediction
                if sha256((folder/"model.joblib").read_bytes()).hexdigest()!=weights_before:raise ValueError("只推理过程修改了权重")
                if json.loads((folder/"service/prediction-failures.json").read_bytes()):raise ValueError("源码通过回退掩盖差别")
                delta=float(np.max(np.abs(base[np.r_[historical,service]]-old_base[np.r_[historical,service]])))
                maximum=max(maximum,delta)
                if delta>1e-10:raise ValueError(f"源码与原模型逐行预测不同：{site}/{key}/{delta}")
                setting=setting_of(original["initial_spec"]["feedback"]) if key=="initial" else (.5,6.)
                pred=replay(base[clock],clock,visible,setting)[clock>=service[0]]
                before=scene.fixed_stream(old_base[:,None],y,windows,service,returned,setting,True)[:,0]
                if not np.allclose(pred,before,rtol=0,atol=1e-10):raise ValueError("完整反馈交付不同")
                if key=="known-best":
                    with np.load(args.original/site/"fixed-winner.npz") as z:
                        if not np.allclose(pred,z["prediction"],rtol=0,atol=1e-10):raise ValueError("未复现原最强固定")
                if key=="initial":
                    with np.load(args.original/site/"issued.npz") as z:
                        if not np.allclose(pred,z["static"],rtol=0,atol=1e-10):raise ValueError("未复现原起点")
                valid=np.isfinite(truth[service]);scores.setdefault(key,{})[site]=error(truth[service][valid],pred[valid])
                np.savez_compressed(folder/"issued.npz",ids=service,prediction=pred,original=before)
                rows.append({"site":site,"case":key,"selected_rows":len(selected),"last_training_hour":int(selected[-1]),
                             "first_training_hour":int(selected[0]),"prediction_max_abs_difference":delta,
                             "source_sha256":sha256(source.encode()).hexdigest(),"weights_sha256":weights_before})
            print(site+" 原工具与可编辑源码的训练及完整交付一致",flush=True)
        means={key:float(np.mean(list(v.values()))) for key,v in scores.items()}
        result={"scope":"已见2025开发资料，开发助手导出现有配方的能力等价实测；无新Luna调用，无新增性能收益", 
                "macro_rmse":means,"station_rmse":scores,"cases":rows,"training_executions":fits,
                "code_features":names,"max_prediction_difference":maximum,"matched_known_best":True}
        atomic_json(args.output/"scores.json",result)
        print(json.dumps({"原最强固定等价误差":means["known-best"],"原起点等价误差":means["initial"],
                          "最大逐行差":maximum,"真实训练":fits,"现场调用":0},ensure_ascii=False))


if __name__=="__main__":main()
