"""在相同合法历史上由Luna生成训练代码，重载发布后比较完整交付与固定代码方案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

import joblib
import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from harness.integrations.generated_regressor import IMPORTS,NUMPY,inspect_source
from harness.integrations.recipe_regressor import code_inputs,source_for_rule
from harness.integrations.proposal_inbox import ProposalInbox
from harness.integrations.pytrio_planner import ChatRequest
from harness.models.station_residual import Rule,error,gate
from harness.models.sparse_reference import SETTINGS,correct,window_scores,setting_of,name,choose_from_feedback
from harness.runtime.storage import LocalStore,atomic_json

ROOT=Path(__file__).resolve().parents[3]


def fit_source(source,data,destination,reuse_model=None):
    """代码检查、单线程子进程和超时共同限制一次试训，失败回到驱动观察。"""
    destination.mkdir(parents=True,exist_ok=True)
    started=monotonic()
    try:
        inspect_source(source)
        path=destination/"source.py";path.write_text(source,encoding="utf-8")
        command=[sys.executable,"-B",str(ROOT/"engineering/cli/study/fit_generated_regressor.py"),
            "--source",str(path.resolve()),"--data",str(data.resolve()),"--output",str(destination.resolve())]
        if reuse_model is not None:command.extend(["--reuse-model",str(reuse_model.resolve())])
        result=subprocess.run(command,
            cwd=destination,capture_output=True,text=True,encoding="utf-8",timeout=90)
        (destination/"stdout.log").write_text(result.stdout,encoding="utf-8")
        (destination/"stderr.log").write_text(result.stderr,encoding="utf-8")
        if result.returncode:raise ValueError(result.stderr[-1200:])
        with np.load(destination/"base.npz",allow_pickle=False) as z:ids=z["ids"];prediction=z["prediction"]
        return ids,prediction
    finally:
        atomic_json(destination/"cost.json",{"elapsed_seconds":monotonic()-started})


def history_probe(tools,folder,y,fit,features=None,feature_names=None):
    """检查所有探索期输入的数值行为；无答案的读数也参与，目标答案不参与。"""
    with np.load(folder/"base.npz") as z:
        ids=z["ids"];raw=z["raw_prediction"]
    use=scene.CLOCK[ids]<scene.START-np.timedelta64(7,'D');ids,raw=ids[use],raw[use]
    limit=1000.*max(1.,float(np.max(np.abs(y[fit]))))
    maximum=float(np.max(np.abs(raw)));order=np.argsort(-np.abs(raw))[:3]
    if features is None:features,feature_names=tools.x[:,:9],tools.names[:9]
    return {"input_rows":len(ids),"prediction_abs_max":maximum,"allowed_abs_max":limit,
            "passed":bool(maximum<=limit),"training_target_range":[float(y[fit].min()),float(y[fit].max())],
            "training_label_hours":sorted(set(map(int,scene.CLOCK[fit].hour))),
            "input_feature_names":feature_names,
            "largest_predictions":[{"time":str(scene.CLOCK[ids[j]]),"input":features[ids[j]].tolist(),
                "prediction":float(raw[j]),"answer_available":bool(np.isfinite(y[ids[j]]))} for j in order]}


def guarded_original(old,site,parent,parent_base,parent_setting,y,fit,windows,service,returned):
    """只给原两案加入历史输入检查，用于分开防失稳和再次提案的收益。"""
    decision=json.loads((old/site/"decision.json").read_bytes())
    audit=json.loads((old/"audit.json").read_bytes());limit=1000.*max(1.,float(np.max(np.abs(y[fit]))))
    safe={row["source"] for row in audit["historical_input_diagnostics"]
          if row["site"]==site and row["historical_prediction_max_abs"]<=limit}
    candidates=[]
    for index,trial in enumerate(decision["trials"],1):
        if trial.get("source_sha256") not in safe:continue
        with np.load(old/site/f"trial-{index}"/"base.npz") as z:
            base=np.zeros(len(y));base[z["ids"]]=z["prediction"]
        setting=setting_of(trial["proposal"]["feedback"])
        candidates.append((base,setting,window_scores(base[:,None],windows,y,setting)[:,0]))
    selected=min(range(len(candidates)),key=lambda i:np.mean(candidates[i][2][:2]**2)) if candidates else None
    parent_errors=window_scores(parent_base[:,None],windows,y,parent_setting)[:,0]
    initial=selected+1 if selected is not None and gate(parent_errors,candidates[selected][2]) else 0
    options=[(parent_base,parent_setting)]+[(b,s) for b,s,_ in candidates]+[(parent_base,s) for _,s,_ in candidates]
    return emit(parent,options,y,windows,service,returned,initial)[0]


def observations(tools,base,windows,y,setting):
    rows={}
    for i,mask in enumerate(windows[:2],1):
        ix=np.flatnonzero(mask);prediction=correct(base[ix,None],y[ix],ix,setting)[:,0]
        residual=prediction-y[ix];fields={}
        for col,field in enumerate(tools.names[:9]):
            values=tools.x[ix,col];cut=float(np.median(values));low=values<=cut
            if low.any() and (~low).any():fields[field]=[round(cut,3),round(float(residual[low].mean()),3),round(float(residual[~low].mean()),3)]
        rows[f"window_{i}"]={"rmse":error(y[ix],prediction),"mean_prediction_minus_answer":float(residual.mean()),
                             "below_above_median_errors":fields}
    return rows


def emit(parent,settings_bases,y,windows,service,returned,initial):
    clock=np.arange(np.flatnonzero(windows[2])[0],len(y));known=np.concatenate((y[clock[clock<service[0]]],returned))
    columns=np.column_stack([correct(base[clock,None],known,clock,setting)[:,0] for base,setting in settings_bases])
    issued,choices=choose_from_feedback(columns,known,clock,service[0],initial,28)
    return issued[clock>=service[0]],choices[clock>=service[0]],columns[clock>=service[0],initial]


def main(argv):
    ap=ArgumentParser(description=__doc__);ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--original",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--probe-history",action="store_true",help="先检查全部历史输入，再重载权重逐行交付")
    ap.add_argument("--replay-first",type=Path,help="沿用指定原运行的首案，仅重新询问第二案")
    ap.add_argument("--reference-code",action="store_true",help="提供当前训练源码、完整原输入及真实小时坐标")
    ap.add_argument("--share-first",action="store_true",help="开发配对：第二案可查看各站已完成首案的合法历史记录")
    args=ap.parse_args(argv);out=args.output;original=args.original
    if args.replay_first and not args.probe_history:raise ValueError("重放首案仅用于历史检查的受控比较")
    if args.reference_code and not args.probe_history:raise ValueError("参考源码模式需要历史检查")
    if args.share_first and (not args.reference_code or not args.replay_first):
        raise ValueError("首案共享开发对照需要相同参考源码接口和已完成的原首案")
    prior=json.loads((original/"scores.json").read_bytes());scene.configure(prior["start"])
    tools=scene.prepare(args.archive,True,True);tools.static=Rule(**prior["initial_spec"]["recipe"])
    features,feature_names=code_inputs(tools) if args.reference_code else (tools.x[:,:9],tools.names[:9])
    shared=[]
    if args.reference_code and args.replay_first:
        earlier=json.loads((args.replay_first/"scores.json").read_bytes())
        if (not earlier.get("reference_code_enabled") or earlier["code_feature_names"]!=feature_names or
                earlier["initial_spec"]!=prior["initial_spec"] or earlier["start"]!=prior["start"] or earlier["counts"]!=prior["counts"]):
            raise ValueError("原首案的输入、起点或时序不同")
        for site in prior["counts"]:
            first=json.loads((args.replay_first/site/"decision.json").read_bytes())["trials"][0]
            if "rejected" in first or "observed_rmse" not in first:continue
            qpath,=list((args.replay_first/"queue/requests/code-edit").glob(site+"-trial-1-*.json"))
            past=json.loads(qpath.read_bytes())["context"]["past"]
            parent_rmse=[past[f"window_{i}"]["parent_rmse"] for i in (1,2)]
            shared.append({"origin_site":site,"proposal":first["proposal"],"source_sha256":first["source_sha256"],
                "origin_observed_rmse":first["observed_rmse"],"origin_parent_rmse":parent_rmse,
                "origin_remaining_errors":first["remaining_errors"],
                "scope":"仅首案在来源站两个历史探索周的结果；可在本站重训，不是本站效果或服务期成绩"})
    parent_setting=setting_of(prior["initial_spec"]["feedback"])
    instruction=("你是现场模型维护驱动。仅凭本请求中的合法历史，提出一个可训练的预测器。返回JSON计划："
        "{hypothesis:中文理由,python_source:完整Python源码字符串,feedback:{gain,hours}}。"
        "代码必须定义build_model()，无参数，返回具有fit(X,y)、predict(X)、get_params的sklearn兼容对象。"
        "fit只会得到历史训练X和目标y；predict每次只得到一行当前读数，返回一维预测。X列顺序见feature_names。"
        "你可以自行构造特征、改变训练目标或组合预测器，不限于历史展示的配方。只可使用列出的导入、numpy纯计算和普通Python。"
        "可定义继承BaseEstimator/RegressorMixin/TransformerMixin的类；属性用公开名称，允许以单下划线结尾。"
        "模型需要保存和重载，自定义变换请使用模块顶层类或函数，不使用lambda。"
        "禁止读写文件、联网、动态导入或执行、访问私有属性和外部状态，不写主程序或打印。"
        "单线程，树最多128棵，每个学习器最多256次迭代；源码不超过20KB；每次训练与逐行预测最多90秒。"
        "必须根据真实试训反馈决定第二案。误差是预测减真实，负值表示低估。feedback只能选allowed_feedback。")
    with LocalStore(out).exclusive():
        if (out/"scores.json").exists() or (out/"sites.json").exists():raise ValueError("不得覆盖原实验")
        inbox=ProposalInbox(out/"queue");datasets={};sources={};model_cache={};published={};decisions={};costs=[]
        atomic_json(out/"sites.json",list(prior["counts"]))
        for site in prior["counts"]:
            truth=scene.readings(args.archive,site).to_numpy();y,fit,windows,service,returned=scene.roles(truth)
            dest=out/site;dest.mkdir()
            ids=(np.flatnonzero((scene.CLOCK>=scene.START-np.timedelta64(21,'D'))&(scene.CLOCK<scene.START))
                 if args.probe_history else np.flatnonzero(np.logical_or.reduce(windows)|(np.arange(len(y))>=service[0])))
            data=dest/"training-input.npz"
            np.savez_compressed(data,train_x=features[fit],train_y=y[fit],predict_x=features[ids],predict_ids=ids)
            parent=tools.train(tools.static,fit,y);joblib.dump((parent,parent_setting),dest/"initial.joblib")
            parent,parent_setting=joblib.load(dest/"initial.joblib");parent_base=tools.issue(parent,np.arange(len(y)))
            parent_errors=window_scores(parent_base[:,None],windows,y,parent_setting)[:,0]
            q=json.loads(next((original/"queue/requests/project").glob(site+"-trial-1-*.json")).read_bytes())["context"]
            context={"task_id":"code-share" if args.share_first else "code-edit" if args.reference_code else "code-probe" if args.probe_history else "code",
                     "site":site,"feature_names":feature_names,"parent_rule":prior["initial_spec"],
                     "past":q["past"],"past_trained_model_comparisons":q["past_trained_model_comparisons"],
                     "allowed_imports":{k:sorted(v) if v else "纯计算函数" for k,v in IMPORTS.items()},
                     "allowed_numpy":sorted(NUMPY),
                     "allowed_feedback":[{"gain":g,"hours":h} for g,h in SETTINGS],"trials":[]}
            if args.probe_history:
                context["observation_contract"]="训练标签仅来自每日0/6/12/18时；预测逐小时交付。历史输入检查覆盖两个探索周全部小时，无答案时刻只检查输入与预测行为。超出训练目标绝对值最大值1000倍的候选被拒绝；第二案请自行决定改法。fit的y是绝对浓度，不是残差；残差目标需在自己的fit中构造。"
            if args.reference_code:
                context["reference_python_source"]=source_for_rule(tools,tools.static)
                context["training_contract"]={"rows":int(fit.sum()),"time_column":"elapsed_hour",
                    "time_meaning":"从输入序列起点经过的真实小时；缺标签不压缩时间，训练按时间递增。predict的时间仅为当前小时。",
                    "reference_meaning":"给出的源码精确实现当前起点，可保留、修改或替换；每项历史配方的days按最后一条训练参考向前计算。",
                    "input_meaning":"原始测点、背景、时钟及已构造乘积均按feature_names顺序；时间列用于选样，当前源码不将其用作回归输入。"}
            candidates=[];cache={};folders={}
            for trial_id in (1,2):
                context["batch_id"]=f"{site}-trial-{trial_id}"
                if trial_id==2 and args.share_first:
                    context["shared_first_trials"]=[entry for entry in shared if entry["origin_site"]!=site]
                body={"model":"gpt-6-luna","messages":[{"role":"system","content":instruction},
                      {"role":"user","content":json.dumps(context,ensure_ascii=False,allow_nan=False)}]}
                if trial_id==1 and args.replay_first:
                    origin=args.replay_first/site/"trial-1/proposal.json";proposal=json.loads(origin.read_bytes())
                    trial={"proposal":proposal,"replayed_from":str(origin),"original_sha256":sha256(origin.read_bytes()).hexdigest()}
                else:
                    response=inbox.send(ChatRequest("local-inbox",body,200));proposal=json.loads(response.body["choices"][0]["message"]["content"])
                    trial={"proposal":proposal}
                folder=dest/f"trial-{trial_id}";folder.mkdir()
                atomic_json(folder/"proposal.json",proposal)
                try:
                    source=proposal["python_source"];setting=setting_of(proposal["feedback"])
                    if not proposal["hypothesis"].strip():raise ValueError("缺少试验理由")
                    identity=sha256(source.encode()).hexdigest()
                    if any(item[0]==identity and item[2]==setting for item in candidates):raise ValueError("重复试验")
                    if identity not in cache:
                        computed,prediction=fit_source(source,data,folder)
                        base=np.zeros(len(y));base[computed]=prediction;cache[identity]=base
                        sources[identity]=source;folders[identity]=folder
                        costs.append({"site":site,"source":identity,"seconds":json.loads((folder/"cost.json").read_bytes())["elapsed_seconds"]})
                    else:base=cache[identity];trial["reused_fit"]=True
                    loss=window_scores(base[:,None],windows,y,setting)[:,0]
                    trial.update(source_sha256=identity,observed_rmse=loss[:2].tolist(),remaining_errors=observations(tools,base,windows,y,setting))
                    if args.probe_history:
                        trial["historical_input_probe"]=history_probe(tools,folders[identity],y,fit,features,feature_names)
                        if not trial["historical_input_probe"]["passed"]:raise ValueError("历史无答案输入上的预测数量级异常；未进入可发布候选")
                    candidates.append((identity,base,setting,loss))
                except (ValueError,TypeError,KeyError,SyntaxError,subprocess.TimeoutExpired) as exc:
                    trial["rejected"]=str(exc)
                context["trials"].append(trial)
            if args.probe_history:
                # 全部提案结束后才打开服务期输入；未来输入上的失败不会进入提案观察。
                service_data=dest/"service-input.npz"
                np.savez_compressed(service_data,predict_x=features[service],predict_ids=service,
                    fallback=parent_base[service],prediction_limit=1000.*max(1.,float(np.max(np.abs(y[fit])))))
                for identity in dict.fromkeys(item[0] for item in candidates):
                    completed,values=fit_source(sources[identity],service_data,folders[identity]/"service",folders[identity]/"model.joblib")
                    cache[identity][completed]=values
            selected=min(range(len(candidates)),key=lambda i:np.mean(candidates[i][3][:2]**2)) if candidates else None
            initial=selected+1 if selected is not None and gate(parent_errors,candidates[selected][3]) else 0
            options=[(parent_base,parent_setting)]+[(base,setting) for _,base,setting,_ in candidates]+[(parent_base,s) for _,_,s,_ in candidates]
            prediction,choices,frozen=emit(parent,options,y,windows,service,returned,initial)
            first=[t for t in context["trials"][:1] if "source_sha256" in t and "rejected" not in t]
            first_only=scene.fixed_stream(parent_base[:,None],y,windows,service,returned,parent_setting,True)[:,0]
            if first:
                identity=first[0]["source_sha256"];s=setting_of(first[0]["proposal"]["feedback"]);base=cache[identity]
                accept=gate(parent_errors,window_scores(base[:,None],windows,y,s)[:,0])
                first_only,_,_=emit(parent,[(parent_base,parent_setting),(base,s),(parent_base,s)],y,windows,service,returned,int(accept))
            with np.load(original/site/"issued.npz") as old:previous={k:old[k] for k in old.files}
            static=scene.fixed_stream(parent_base[:,None],y,windows,service,returned,parent_setting,True)[:,0]
            if not np.allclose(static,previous["static"],rtol=0,atol=1e-10):raise ValueError("与原比较的强起点不一致")
            issued={"project":prediction,"first-only":first_only,"frozen-initial":frozen,"static":static,
                    "previous-project":previous["project"],"paper":previous["paper"],"numeric":previous["numeric"]}
            with np.load(original/site/"fixed-winner.npz") as old:issued["previous-best-fixed"]=old["prediction"]
            if args.replay_first:
                if args.reference_code:
                    with np.load(args.replay_first/site/"issued.npz") as old:issued["prior-code-project"]=old["project"]
                else:issued["guarded-original"]=guarded_original(args.replay_first,site,parent,parent_base,parent_setting,y,fit,windows,service,returned)
            np.savez_compressed(dest/"issued.npz",ids=service,choices=choices,returned=returned,**issued)
            atomic_json(dest/"deployment.json",{"initial":initial,"candidates":[{"source":i,"feedback":s,"errors":loss.tolist()} for i,_,s,loss in candidates],
                        "parent_feedback":parent_setting,"parent_errors":parent_errors.tolist()})
            atomic_json(dest/"decision.json",{"trials":context["trials"],"selected":selected,"adopted":bool(initial)})
            datasets[site]=(truth,y,fit,windows,service,returned,parent_base,parent_errors,data,folders)
            model_cache[site]=cache;published[site]=issued;decisions[site]=context["trials"]
            print(site+" 两次生成代码提案与发布完成",flush=True)
        # 所有现场请求结束后，再把每份实际生成代码固定应用到各站，防止只计局部赢家。
        failures=[]
        for identity,source in sources.items():
            for site,parts in datasets.items():
                truth,y,fit,windows,service,returned,parent_base,parent_errors,data,folders=parts
                try:
                    if identity not in model_cache[site]:
                        folder=out/site/"fixed"/identity
                        ids,prediction=fit_source(source,data,folder);base=np.zeros(len(y));base[ids]=prediction
                        model_cache[site][identity]=base
                        folders[identity]=folder
                        costs.append({"site":site,"source":identity,"seconds":json.loads((folder/"cost.json").read_bytes())["elapsed_seconds"]})
                    base=model_cache[site][identity]
                    if args.probe_history:
                        if not history_probe(tools,folders[identity],y,fit,features,feature_names)["passed"]:raise ValueError("历史输入检查未通过")
                        service_folder=folders[identity]/"service"
                        if not (service_folder/"base.npz").exists():
                            completed,values=fit_source(source,out/site/"service-input.npz",service_folder,folders[identity]/"model.joblib")
                            base[completed]=values
                except (ValueError,TypeError,SyntaxError,subprocess.TimeoutExpired) as exc:
                    failures.append({"site":site,"source":identity,"error":str(exc)});base=None
                for setting in SETTINGS:
                    key=identity[:16]+"-"+name(setting)
                    pred=published[site]["static"] if base is None else scene.fixed_stream(base[:,None],y,windows,service,returned,setting,True)[:,0]
                    published[site]["direct-"+key]=pred
                    accept=base is not None and gate(parent_errors,window_scores(base[:,None],windows,y,setting)[:,0])
                    published[site]["fixed-"+key]=pred if accept else published[site]["static"]
        scores={}
        for site,issued in published.items():
            truth,y,fit,windows,service,*_=datasets[site];valid=np.isfinite(truth[service])
            for key,pred in issued.items():scores.setdefault(key,{})[site]=error(truth[service][valid],pred[valid])
            np.savez_compressed(out/site/"all-issued.npz",**issued)
        means={k:float(np.mean(list(v.values()))) for k,v in scores.items()}
        controls=[k for k in means if k.startswith(("direct-","fixed-")) or k in ("static","numeric","previous-best-fixed")]
        best=min(controls,key=means.get)
        result={"scope":"已见2025开发，生成代码的完整交付实测；每站两案，不认定次数最优", "start":prior["start"],
                "historical_probe_enabled":args.probe_history,"first_proposal_origin":str(args.replay_first) if args.replay_first else None,
                "reference_code_enabled":args.reference_code,"code_feature_names":feature_names,
                "shared_first_enabled":args.share_first,
                "station_rmse":scores,"macro_rmse":means,"best_fixed":best,"source_ids":list(sources),"fixed_failures":failures,
                "fits":costs,"original":str(original),"initial_spec":prior["initial_spec"],"counts":prior["counts"]}
        atomic_json(out/"scores.json",result)
        print(json.dumps({"生成代码完整误差":means["project"],"原现场":means["previous-project"],"论文适配":means["paper"],
                          "最强固定":means[best],"仅首案":means["first-only"],"有效源码":len(sources)},ensure_ascii=False))
