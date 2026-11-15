"""重载现场首案的跨站权重，用合法历史及到期反馈选择完整交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_code_driver import fit_source
from harness.integrations.recipe_regressor import code_inputs
from harness.models.station_residual import Rule,error,gate
from harness.models.sparse_reference import SETTINGS,correct,window_scores,choose_from_feedback,name,setting_of
from harness.runtime.storage import LocalStore,atomic_json


def read(path):return json.loads(path.read_bytes())


def model_folder(generated,site,identity):
    """同源码在本站试训过时用该实际权重，否则用已冻结的跨站重训权重。"""
    for trial_id,trial in enumerate(read(generated/site/"decision.json")["trials"],1):
        folder=generated/site/f"trial-{trial_id}"
        if trial.get("source_sha256")==identity and (folder/"model.joblib").exists():return folder
    folder=generated/site/"fixed"/identity
    if not (folder/"model.joblib").is_file():raise ValueError("首案池缺少在本站合法训练的权重")
    return folder


def main(argv):
    ap=ArgumentParser(description=__doc__);ap.add_argument("--generated",type=Path,required=True)
    ap.add_argument("--archive",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--current",type=Path,required=True,help="同条件的当前较好代码候选，仅作评价对照")
    args=ap.parse_args(argv);original=read(args.generated/"scores.json")
    current=read(args.current/"scores.json")
    if any(current[key]!=original[key] for key in ("start","initial_spec","counts")):
        raise ValueError("当前代码对照不是同一输入与起点")
    if not original.get("reference_code_enabled"):raise ValueError("需已核验的等价源码入口")
    verified=read(args.generated/"audit.json")
    if not verified["full_predictions_and_fixed_controls_verified"]:raise ValueError("原交付尚未核验")
    scene.configure(original["start"]);tools=scene.prepare(args.archive,True,True)
    tools.static=Rule(**original["initial_spec"]["recipe"]);parent_setting=setting_of(original["initial_spec"]["feedback"])
    features,feature_names=code_inputs(tools);registry={}
    for site in original["counts"]:
        trial=read(args.generated/site/"decision.json")["trials"][0]
        if "rejected" in trial or "source_sha256" not in trial:continue
        identity=trial["source_sha256"];source=trial["proposal"]["python_source"]
        if sha256(source.encode()).hexdigest()!=identity:raise ValueError("首案源码身份不同")
        registry.setdefault(identity,{"origin_site":site,"python_source":source,"proposal":trial["proposal"]})
    with LocalStore(args.output).exclusive():
        if (args.output/"scores.json").exists() or (args.output/"source-pool.json").exists():raise ValueError("不可覆盖池试用记录")
        atomic_json(args.output/"source-pool.json",registry)
        all_scores={};diagnostics=[];count_options=None
        for site in original["counts"]:
            dest=args.output/site;dest.mkdir()
            truth=scene.readings(args.archive,site).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
            history=np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&(scene.CLOCK<scene.START))
            ids=np.r_[history,service]
            parent,_=joblib.load(args.generated/site/"initial.joblib");parent_base=tools.issue(parent,np.arange(len(y)))
            shutil.copyfile(args.generated/site/"initial.joblib",dest/"initial.joblib")
            data=dest/"prediction-input.npz"
            np.savez_compressed(data,predict_x=features[ids],predict_ids=ids,fallback=parent_base[ids],
                                prediction_limit=1000.*max(1.,float(np.max(np.abs(y[fit])))))
            bases={"initial":parent_base};published=[]
            for identity,entry in registry.items():
                folder=model_folder(args.generated,site,identity)
                if (folder/"source.py").read_text(encoding="utf-8")!=entry["python_source"]:raise ValueError("权重源码不是首案原件")
                weights=folder/"model.joblib";weight_hash=sha256(weights.read_bytes()).hexdigest()
                generated_ids,values=fit_source(entry["python_source"],data,dest/identity,weights)
                if sha256(weights.read_bytes()).hexdigest()!=weight_hash:raise ValueError("只推理修改了发布权重")
                if read(dest/identity/"prediction-failures.json"):raise ValueError("已核验首案出现新的推理失败")
                base=np.zeros(len(y));base[generated_ids]=values;bases[identity]=base
                published.append({"source":identity,"origin_site":entry["origin_site"],"weights":str(weights),"weights_sha256":weight_hash})
            clock=np.arange(np.flatnonzero(windows[2])[0],len(y));known=np.concatenate((y[clock[clock<service[0]]],returned))
            options=[];errors=[];columns=[]
            for identity,base in bases.items():
                for setting in SETTINGS:
                    options.append({"source":identity,"feedback":list(setting),"name":identity+"-"+name(setting)})
                    errors.append(window_scores(base[:,None],windows,y,setting)[:,0])
                    columns.append(correct(base[clock,None],known,clock,setting)[:,0])
            errors=np.asarray(errors);columns=np.column_stack(columns);use=clock>=service[0]
            parent_index=next(i for i,v in enumerate(options) if v["source"]=="initial" and tuple(v["feedback"])==parent_setting)
            best=int(np.argmin(np.mean(errors[:,:2]**2,axis=1)))
            initial=best if gate(errors[parent_index],errors[best]) else parent_index
            online,choices=choose_from_feedback(columns,known,clock,service[0],initial,28)
            issued={"pool-online":online[use],"pool-history-direct":columns[use,best],
                    "pool-history-gate":columns[use,initial],"static":columns[use,parent_index]}
            for i,option in enumerate(options):
                key="bank-"+option["name"]
                issued[key]=columns[use,i]
                issued["gate-"+key]=columns[use,i] if gate(errors[parent_index],errors[i]) else issued["static"]
            # 这些已保存的同条件对手只进入评价，不用于池中选择。
            with np.load(args.generated/site/"issued.npz") as previous:
                for now,before in (("previous-code","project"),("first-only","first-only"),("previous-project","previous-project"),
                                   ("paper","paper"),("numeric","numeric"),("previous-best-fixed","previous-best-fixed")):
                    issued[now]=previous[before]
                if not np.allclose(issued["static"],previous["static"],rtol=0,atol=1e-10):raise ValueError("原起点发放不一致")
            with np.load(args.current/site/"issued.npz") as previous:
                if not np.array_equal(previous["ids"],service):raise ValueError("当前代码对照的服务时钟不同")
                issued["current-code"]=previous["project"]
            atomic_json(dest/"deployment.json",{"options":options,"parent_index":parent_index,"history_best":best,
                        "initial":initial,"history_rmse":errors.tolist(),"published":published,"feature_names":feature_names})
            np.savez_compressed(dest/"issued.npz",ids=service,choices=choices[use],returned=returned,**issued)
            valid=np.isfinite(truth[service]);reference=valid&np.isfinite(returned)
            for key,prediction in issued.items():all_scores.setdefault(key,{})[site]=error(truth[service][valid],prediction[valid])
            # 事后检查“只看能收到答案的时刻”与完整评价是否会排出不同赢家。
            all_errors=[error(truth[service][valid],columns[use,i][valid]) for i in range(len(options))]
            reference_errors=[error(truth[service][reference],columns[use,i][reference]) for i in range(len(options))]
            oracle=int(np.argmin(all_errors));reference_winner=int(np.argmin(reference_errors))
            diagnostics.append({"site":site,"history_best":options[best]["name"],"full_hindsight_best":options[oracle]["name"],
                "reference_hindsight_best":options[reference_winner]["name"],"full_hindsight_rmse":all_errors[oracle],
                "reference_winner_full_rmse":all_errors[reference_winner],"online_rmse":all_scores["pool-online"][site],
                "hindsight_note":"两种事后赢家都使用服务期答案，只用于诊断，不进入交付选择"})
            count_options=len(options);print(site+" 首案池合法选择与完整交付完成",flush=True)
        means={k:float(np.mean(list(v.values()))) for k,v in all_scores.items()}
        fixed=[k for k in means if k.startswith(("bank-","gate-bank-")) or k in ("static","numeric","previous-best-fixed")]
        best_fixed=min(fixed,key=means.get)
        best_simple=min([*fixed,"pool-history-direct","pool-history-gate"],key=means.get)
        result={"scope":"已见开发，重载Luna原首案在各站的已训练权重，新增历史试用与选择；无新训练或Luna调用",
                "generated":str(args.generated),"current":str(args.current),"start":original["start"],"initial_spec":original["initial_spec"],
                "counts":original["counts"],"source_ids":list(registry),"options_per_site":count_options,
                "macro_rmse":means,"station_rmse":all_scores,"best_fixed":best_fixed,"best_simple":best_simple,"hindsight_diagnostics":diagnostics}
        atomic_json(args.output/"scores.json",result)
        print(json.dumps({"首案池持续选择":means["pool-online"],"历史首选直接发布":means["pool-history-direct"],
            "历史确认后发布":means["pool-history-gate"],"原现场":means["previous-project"],"论文适配":means["paper"],
            "当前代码候选":means["current-code"],"最强固定":means[best_fixed],"最强简单方案":means[best_simple],
            "候选源码":len(registry),"新增训练":0,"新增现场调用":0},ensure_ascii=False))

