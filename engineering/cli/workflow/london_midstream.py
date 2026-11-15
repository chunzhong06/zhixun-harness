"""冻结前半段强交付，在中途用已到期反馈重新提案、实训、确认后交接。"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_code_driver import fit_source
from harness.integrations.generated_regressor import IMPORTS, NUMPY
from harness.integrations.candidate_edit import named_scores, resolve_edit
from harness.integrations.proposal_inbox import ProposalInbox
from harness.integrations.pytrio_planner import ChatRequest
from harness.integrations.recipe_regressor import code_inputs, source_for_rule
from harness.models.sparse_reference import SETTINGS, correct, setting_of
from harness.models.station_residual import Rule, error, gate
from harness.runtime.storage import LocalStore, atomic_json


def read(path): return json.loads(path.read_bytes())


def identity(plan):
    payload = {k: plan[k] for k in ("components", "filters")}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize(raw, arm, reference, names, *, edit_contract=False):
    """保留旧子集协议；新版两组共享父版编辑权限和相同执行校验。"""
    if raw.get("action") == "stop":
        if not str(raw.get("reason", "")).strip(): raise ValueError("停止需要理由")
        return None
    if raw.get("action") != "experiment" or not str(raw.get("hypothesis", "")).strip():
        raise ValueError("需提出experiment及中文假设，或明确stop")
    if edit_contract:
        raw, _ = resolve_edit(raw, reference)
        # 两组使用相同的可编辑能力；研究策略差别不通过削弱工具实现。
        arm = "project"
    filters = raw.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 2: raise ValueError("最多两个训练筛选条件")
    for f in filters:
        if set(f) != {"feature", "operator", "threshold"} or f["feature"] not in names or f["operator"] not in ("<=", ">"):
            raise ValueError("筛选应指定已给特征、<=或>及有限阈值")
        if not np.isfinite(float(f["threshold"])): raise ValueError("筛选阈值不有限")
    if arm == "paper":
        components = deepcopy(reference["components"])
        feedbacks = raw.get("feedbacks", [c["feedback"] for c in components])
        if len(feedbacks) != len(components): raise ValueError("论文反馈数与原组件数不同")
        for c, feedback in zip(components, feedbacks, strict=True): c["feedback"] = feedback
    else:
        components = deepcopy(raw["components"])
        if not isinstance(components, list) or not 1 <= len(components) <= 2: raise ValueError("每案一或两个训练组件")
    for c in components:
        if set(c) != {"python_source", "weight", "feedback"}: raise ValueError("组件字段不符")
        if not isinstance(c["python_source"], str) or not np.isfinite(float(c["weight"])) or c["weight"] <= 0:
            raise ValueError("组件源码及正权重不完整")
        setting_of(c["feedback"])
    if not np.isclose(sum(float(c["weight"]) for c in components), 1., rtol=0, atol=1e-10):
        raise ValueError("组件权重之和应为1")
    return {"components": components, "filters": filters, "hypothesis": raw["hypothesis"]}


def selected_rows(plan, fit, features, names):
    selected = fit.copy()
    for f in plan["filters"]:
        values = features[:, names.index(f["feature"])]
        selected &= values <= float(f["threshold"]) if f["operator"] == "<=" else values > float(f["threshold"])
    if selected.sum() < 12: raise ValueError(f"筛选前{fit.sum()}条，筛选后{selected.sum()}条，至少12条")
    return selected


def summarize(prediction, known, clock, windows, features, names):
    result = []
    for mask in windows:
        use = mask[clock]; ix = clock[use]; residual = prediction[use] - known[use]
        if not len(ix):
            result.append({"rows":0,"rmse":None,"prediction_minus_answer":None,"median_low_high_error":{}})
            continue
        fields = {}
        for j, field in enumerate(names[:9]):
            values = features[ix,j]; split = float(np.median(values)); lo = values <= split
            if lo.any() and (~lo).any(): fields[field] = [split, float(residual[lo].mean()), float(residual[~lo].mean())]
        result.append({"rows": len(ix), "rmse": error(known[use], prediction[use]),
                       "prediction_minus_answer": float(residual.mean()), "median_low_high_error": fields})
    return result


def train_plan(plan, folder, features, names, fit, labels, history, known, windows):
    """这里只接触维护时点前输入；未来服务另用重载接口，不能影响提案观察。"""
    folder.mkdir(parents=True, exist_ok=True); atomic_json(folder / "plan.json", plan)
    selected = selected_rows(plan, fit, features, names)
    data = folder / "training-input.npz"
    np.savez_compressed(data, train_x=features[selected], train_y=labels[selected],
                        train_ids=np.flatnonzero(selected), predict_x=features[history], predict_ids=history)
    maximum = 1000. * max(1., float(np.max(np.abs(labels[selected]))))
    components = []; total = np.zeros(len(history)); component_predictions = []
    for i, component in enumerate(plan["components"]):
        ids, values = fit_source(component["python_source"], data, folder / f"component-{i}")
        if not np.array_equal(ids, history): raise ValueError("训练后历史预测身份不同")
        with np.load(folder / f"component-{i}/base.npz") as z:
            if np.max(np.abs(z["raw_prediction"])) > maximum: raise ValueError("历史全部小时预测超过训练量级上限")
        corrected = correct(values[:,None], known, history, setting_of(component["feedback"]))[:,0]
        total += component["weight"] * corrected
        component_predictions.append(corrected)
        components.append({"source_sha256":sha256(component["python_source"].encode()).hexdigest(),
                           "weights_sha256":sha256((folder/f"component-{i}/model.joblib").read_bytes()).hexdigest()})
    loss = [error(known[mask[history]], total[mask[history]]) for mask in windows]
    record = {"identity": identity(plan), "training_rows": int(selected.sum()), "last_training_hour": int(np.flatnonzero(selected)[-1]),
              "history_rmse": loss, "exploration": summarize(total, known, history, windows[:2], features, names),
              "components": components, "prediction_limit": maximum}
    # 复用已真实训练的组件预测作去组件消融；不额外训练，也不披露确认周。
    ablations = []
    if len(components) > 1:
        for i, component in enumerate(plan["components"]):
            remaining = 1. - component["weight"]
            ablated = (total - component["weight"] * component_predictions[i]) / remaining
            values = [error(known[m[history]], ablated[m[history]]) for m in windows[:2]]
            ablations.append({"removed_component": i, "remaining_weights_renormalized": True,
                              "scores": named_scores(values, windows[:2], history,
                                  prediction_object="ensemble_without_component_" + str(i) + "_after_feedback")})
    record["component_ablations"] = ablations
    atomic_json(folder / "training.json", record)
    np.savez_compressed(folder / "historical-issued.npz", ids=history, prediction=total)
    return record


def serve_plan(plan, folder, features, history, future, known, returned, fallback):
    """所有提案和确认结束后重载权重；新权重用自己的历史误差建立反馈状态。"""
    record = read(folder / "training.json"); total = np.zeros(len(future))
    data = folder / "future-input.npz"
    np.savez_compressed(data, predict_x=features[future], predict_ids=future, fallback=fallback,
                        prediction_limit=record["prediction_limit"])
    clock = np.r_[history, future]; reference = np.r_[known, returned]
    for i, component in enumerate(plan["components"]):
        unit = folder / f"component-{i}"; weight_path = unit / "model.joblib"
        before = sha256(weight_path.read_bytes()).hexdigest()
        ids, values = fit_source(component["python_source"], data, unit / "service", weight_path)
        if not np.array_equal(ids, future) or before != sha256(weight_path.read_bytes()).hexdigest():
            raise ValueError("重载预测更换了权重或时钟")
        with np.load(unit / "base.npz") as z: base = np.r_[z["prediction"], values]
        prediction = correct(base[:,None], reference, clock, setting_of(component["feedback"]))[:,0]
        total += component["weight"] * prediction[len(history):]
    np.savez_compressed(folder / "future-issued.npz", ids=future, prediction=total)
    return total


def propose(arm, site, dest, reference, context, parent_errors, features, names, fit, labels, history, known, windows, inbox, edit_contract=False,
            *, train_callback=None, initial_candidates=None, follow_best=False):
    common = ("你是已运行28天后的现场维护驱动。仅用本请求的数据，先诊断当前预测哪里出错，再选择停止或真实试验。"
        "返回JSON。停止为{action:'stop',reason:中文理由}；实验需action:'experiment',hypothesis:中文机制假设。"
        "每站最多3次请求，可在任一轮停止；不是必须用完。执行结果将作为下一轮观察。"
        "最后确认周不会反馈给你。误差是预测减答案；训练标签仅0/6/12/18时，实际逐小时预测。"
        "所有输入及时间列含义见context，禁止访问外部文件、未来答案、其他站或其他组。")
    if arm == "project" or edit_contract:
        specific = ("实验还需components:[{python_source:完整源码,weight:正权重,feedback:{gain,hours}}],filters:[]。"
            "组件一或两个，权重和为1；你可修改所给原训练实现、训练窗口、目标、输入或学习器，也可保留。"
            "每份源码定义无参数build_model，返回sklearn兼容fit/predict/get_params对象；fit的y为绝对浓度，predict每次只有当前一行。"
            "可选filters最多两个{feature,operator:'<='或'>',threshold}，先筛训练行。"
            "仅用列出的纯计算导入与numpy操作。模块顶层定义类/函数，不用lambda、私有属性、动态执行、文件或网络。"
            "单线程，每组件最多128棵树/256次迭代，20KB源码，训练推理90秒。")
    else:
        specific = ("遵循Self-Healing ML的训练子集假设、真实重训与回测方式：保留给定完整原学习器、特征和配比，"
            "实验返回filters:[{feature,operator:'<='或'>',threshold}]，最多两个条件或空列表；"
            "可给feedbacks:[{gain,hours},{gain,hours}]调整两个组件既有反馈，省略则保持。"
            "不要写源码或改变学习器。用实际诊断提出筛选，而非按最终未来分数挑选。")
    if edit_contract:
        specific = specific.replace("实验还需components:[{python_source:完整源码,weight:正权重,feedback:{gain,hours}}],filters:[]。", "实验采用下面的编辑契约。")
        specific += ("本次使用父版绑定的编辑契约。每案需parent_identity与reference_identity完全相同。"
            "优先返回component_edits:[{index:从0开始的编号,changes:{python_source或weight或feedback:新值}}]；"
            "未提到的组件和字段自动保留，禁止在此格式同时给components。"
            "如确需整套替换，返回replace_all:true和components完整列表。每次以reference_plan为父版，先前提案不自动变成父版。"
            "训练输入fit(X,y)中的y是绝对浓度；若学残差，fit中自行减去背景，predict中加回。"
            "scores的每项是一个时间窗的整套预测误差，绝不是两组件各自误差。去组件消融已另行标明对象。")
        if arm == "paper":
            specific += ("本组执行MLE-STAR部件消融与定向改进的本地适配：先依据实际去组件消融和错例定位值得修改的部分，"
                         "在hypothesis中说明定位依据；围绕该部分提出可执行改动，收到实际结果后修订或停止。"
                         "同样拥有完整源码编辑权，本地预算内完成定位、试改、选优。不是作者整套搜索复现。")
        if follow_best:
            specific=specific.replace('每次以reference_plan为父版，先前提案不自动变成父版。',
                '每次请求显式给出当前探索最优工作版，它可能来自共享候选或先前试改；必须使用本请求的父版身份，不能沿用旧请求身份。')
    context = deepcopy(context); context.update(task_id="mid-"+arm, site=site, trials=[])
    trials = []; candidates = (list(initial_candidates) if initial_candidates is not None else
        [{"folder":"refit", "plan":reference, "training":read(dest/"refit/training.json")}])
    for turn in range(1,4):
        if follow_best:
            working=min(candidates,key=lambda c:np.mean(np.square(c['training']['history_rmse'][:2])))
            reference=deepcopy(working['plan']);training=working['training']
            context.update(reference_plan=reference,reference_identity=identity(reference),
                           reference_origin=working['folder'])
            context['refit_observation']={'scores':named_scores(training['history_rmse'][:2],windows[:2],history,
                prediction_object='current_working_candidate_after_feedback'),'errors':deepcopy(training['exploration']),
                'component_ablations':deepcopy(training['component_ablations'])}
        context["batch_id"] = f"{site}-trial-{turn}"
        body={"model":"gpt-6-luna","messages":[{"role":"system","content":common+specific},
              {"role":"user","content":json.dumps(context,ensure_ascii=False,allow_nan=False)}]}
        folder = dest/arm/f"trial-{turn}";folder.mkdir(parents=True)
        try:
            response = inbox.send(ChatRequest("local-inbox",body,200))
            raw = json.loads(response.body["choices"][0]["message"]["content"])
        except (TimeoutError,json.JSONDecodeError,KeyError,TypeError) as exc:
            trial={"request_failed":True,"rejected":str(exc)};trials.append(trial);context["trials"]=trials
            atomic_json(folder/"request-error.json",trial);continue
        atomic_json(folder/"proposal.json",raw); trial={"proposal":raw}
        try:
            plan=normalize(raw,arm,reference,names,edit_contract=edit_contract)
            if plan is None:trial["stopped"]=True;trials.append(trial);break
            if any(identity(c["plan"])==identity(plan) for c in candidates):raise ValueError("与已完成试训相同，不重复训练")
            training=(train_callback or train_plan)(plan,folder,features,names,fit,labels,history,known,windows)
            candidates.append({"folder":str(folder.relative_to(dest)),"plan":plan,"training":training})
            trial.update(identity=identity(plan),remaining_errors=training["exploration"])
            if edit_contract:
                _, trial["actual_change"] = resolve_edit(raw, reference)
                trial["scores"] = named_scores(training["history_rmse"][:2], windows[:2], history,
                                                prediction_object="complete_ensemble_after_feedback")
                trial["component_ablations"] = training["component_ablations"]
                for i, error_summary in enumerate(trial["remaining_errors"]):
                    error_summary.update(window_id=f"exploration-{i+1}", prediction_object="complete_ensemble_after_feedback")
            else:
                trial["observed_rmse"] = training["history_rmse"][:2]
        except (KeyError,ValueError,TypeError,SyntaxError,subprocess.TimeoutExpired) as exc: trial["rejected"]=str(exc)
        trials.append(trial);context["trials"]=trials
    best=min(candidates,key=lambda c:np.mean(np.square(c["training"]["history_rmse"][:2])))
    # 两组都可以选择相同的无新改法补训。确认仅发生在全部生成停止之后。
    parent=parent_errors
    adopted=gate(np.asarray(parent),np.asarray(best["training"]["history_rmse"]))
    decision={"trials":trials,"nominated":best["folder"],"adopted":adopted,
              "parent_errors":parent,"nominated_errors":best["training"]["history_rmse"],
              "candidates":[{"folder":c["folder"],"identity":identity(c["plan"])} for c in candidates]}
    atomic_json(dest/(arm+"-decision.json"),decision)
    return candidates,decision


def main(argv):
    ap=ArgumentParser(description=__doc__);ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--baseline",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--edit-contract", action="store_true", help="明确评分窗口、局部编辑和同工具MLE-STAR机制适配")
    ap.add_argument("--prior-run", type=Path, help="复用同条件已核验固定交付，保留此前更强对手")
    args=ap.parse_args(argv);baseline=read(args.baseline/"scores.json")
    if not read(args.baseline/"audit.json")["controls_and_scores_verified"]:raise ValueError("强基准未核验")
    if args.prior_run:
        prior = read(args.prior_run/"scores.json")
        prior_audit = read(args.prior_run/"audit.json")
        if not prior_audit.get("all_fixed_and_scores_verified") or prior["baseline"] != str(args.baseline) or prior["start"] != baseline["start"]:
            raise ValueError("已有强固定的起点或时间范围不同")
    scene.configure(baseline["start"]);tools=scene.prepare(args.archive,True,True);features,names=code_inputs(tools)
    reference={"hypothesis":"当前已冻结最强组合，原办法原样补训", "filters":[], "components":[
        {"python_source":source_for_rule(tools,Rule("median","network",21,1.)),"weight":.75,"feedback":{"gain":.5,"hours":6.}},
        {"python_source":Path("results/experiments/local-defaults/whole-maintenance/london-reference-code/CE2/trial-1/source.py").read_text(encoding="utf-8"),
         "weight":.25,"feedback":{"gain":.75,"hours":18.}}]}
    with LocalStore(args.output).exclusive():
        if (args.output/"sites.json").exists():raise ValueError("不可覆盖中途研究")
        atomic_json(args.output/"sites.json",list(baseline["counts"]));atomic_json(args.output/"reference-plan.json",reference)
        inbox=ProposalInbox(args.output/"queue");cache={};registry={};issued_all={};counts={}
        for site in baseline["counts"]:
            dest=args.output/site;dest.mkdir();truth=scene.readings(args.archive,site).to_numpy()
            old_y,old_fit,old_windows,service,returned=scene.roles(truth);start=int(service[0]);at=start+28*24
            labels=np.r_[old_y[:start],returned]
            fit=(np.arange(len(labels))<at-21*24)&np.isfinite(labels)
            windows=[(np.arange(len(labels))>=at+a*24)&(np.arange(len(labels))<at+(a+7)*24)&np.isfinite(labels) for a in (-21,-14,-7)]
            history=np.arange(at-21*24-6,at);known=labels[history];future=np.arange(at,len(labels))
            with np.load(args.baseline/site/"best-fixed.npz") as z:
                if not np.array_equal(z["ids"],service):raise ValueError("前半段原交付时钟不同")
                original=z["prediction"].copy()
            parent_errors=[error(labels[m],original[np.flatnonzero(m)-start]) for m in windows]
            refit=train_plan(reference,dest/"refit",features,names,fit,labels,history,known,windows)
            observed=np.flatnonzero((np.arange(len(labels))>=start)&(np.arange(len(labels))<at-7*24)&np.isfinite(labels))
            summaries=[]
            for hour in observed:
                summaries.append({"hour":int(hour),"raw_inputs":features[hour,:9].round(5).tolist(),
                                  "answer":float(labels[hour]),"issued_prediction":float(original[hour-start])})
            context={"feature_names":names,"raw_input_names":names[:9],"reference_plan":reference,
                     "parent_history_rmse":parent_errors[:2],"refit_observation":{"rmse":refit["history_rmse"][:2],"errors":refit["exploration"]},
                     "legal_service_observations":summaries,"trials":[],
                     "training_contract":{"last_training_hour":int(np.flatnonzero(fit)[-1]),"rows":int(fit.sum()),
                        "maintenance_hour":at,"exploration_hours":[[at-21*24,at-14*24],[at-14*24,at-7*24]],
                        "time_column":"elapsed_hour，真实连续小时，缺答案不压缩；训练按小时递增，内部窗口从最后训练小时回数",
                        "confirmation":"最后7日仅用于提名后确认，不向驱动提供答案或分数","future":"第28至56日尚未读取给任何驱动"},
                     "allowed_imports":{k:sorted(v) if v else "纯计算" for k,v in IMPORTS.items()},"allowed_numpy":sorted(NUMPY),
                     "allowed_feedback":[{"gain":g,"hours":h} for g,h in SETTINGS]}
            if args.edit_contract:
                context.pop("parent_history_rmse")
                context["reference_identity"] = identity(reference)
                context["parent_scores"] = named_scores(parent_errors[:2], windows[:2], history,
                                                          prediction_object="deployed_parent_ensemble_after_feedback")
                context["refit_observation"] = {
                    "scores": named_scores(refit["history_rmse"][:2], windows[:2], history,
                                            prediction_object="complete_refit_ensemble_after_feedback"),
                    "errors": refit["exploration"], "component_ablations": refit["component_ablations"]}
                for i, error_summary in enumerate(context["refit_observation"]["errors"]):
                    error_summary.update(window_id=f"exploration-{i+1}", prediction_object="complete_refit_ensemble_after_feedback")
            # 不把确认成绩装进驱动请求；另传给采用逻辑，保持两组同一条件。
            with ThreadPoolExecutor(2) as pool:
                tasks={arm:pool.submit(propose,arm,site,dest,reference,context,parent_errors,features,names,fit,labels,history,known,windows,inbox,args.edit_contract)
                       for arm in ("project","paper")}
                arm_results={}
                for arm,task in tasks.items():arm_results[arm]=task.result()
            result={"baseline":original.copy()};predictions={}
            for arm,(candidates,decision) in arm_results.items():
                nominee=next(c for c in candidates if c["folder"]==decision["nominated"])
                for candidate in candidates:
                    key=identity(candidate["plan"])
                    if key not in predictions:
                        predictions[key]=serve_plan(candidate["plan"],dest/candidate["folder"],features,history,future,known,labels[future],original[28*24:])
                    registry.setdefault(key,{"plan":candidate["plan"],"origin_site":site,"origin_arm":arm,"origin_folder":candidate["folder"]})
                    cache[(site,key)]={**candidate,"future":predictions[key]}
                result[arm]=original.copy()
                if decision["adopted"]:result[arm][28*24:]=predictions[identity(nominee["plan"])]
            base_key=identity(reference);refit_values=predictions[base_key]
            result["refit-direct"]=np.r_[original[:28*24],refit_values]
            result["refit-gate"]=result["refit-direct"] if gate(np.asarray(parent_errors),np.asarray(refit["history_rmse"])) else original.copy()
            np.savez_compressed(dest/"issued.npz",ids=service,returned=returned,**result)
            atomic_json(dest/"timing.json",{"start_hour":start,"maintenance_hour":at,"training_last_hour":int(np.flatnonzero(fit)[-1]),
                        "history_hours":history.tolist(),"windows":[np.flatnonzero(m).tolist() for m in windows],"parent_errors":parent_errors})
            counts[site]={"issued":len(service),"scored":int(np.isfinite(truth[service]).sum()),"reference":int(np.isfinite(returned).sum())}
            issued_all[site]=result
            print(site+" 中途两组提案及完整交接完成",flush=True)
        atomic_json(args.output/"driver-complete.json",{"done":True});atomic_json(args.output/"plan-bank.json",registry)
        # 生成停止、实际交付封存后，再把每份新办法固定跨站重训作强对照，不回到驱动。
        for site in baseline["counts"]:
            dest=args.output/site;truth=scene.readings(args.archive,site).to_numpy();old_y,_,_,service,returned=scene.roles(truth)
            start=int(service[0]);at=start+28*24;labels=np.r_[old_y[:start],returned]
            fit=(np.arange(len(labels))<at-21*24)&np.isfinite(labels)
            windows=[(np.arange(len(labels))>=at+a*24)&(np.arange(len(labels))<at+(a+7)*24)&np.isfinite(labels) for a in (-21,-14,-7)]
            history=np.arange(at-21*24-6,at);future=np.arange(at,len(labels));known=labels[history]
            original=issued_all[site]["baseline"];parent_errors=read(dest/"timing.json")["parent_errors"];bank={};failures={}
            for key,entry in registry.items():
                try:
                    if (site,key) in cache:item=cache[(site,key)]
                    else:
                        folder=dest/"fixed"/key;training=train_plan(entry["plan"],folder,features,names,fit,labels,history,known,windows)
                        value=serve_plan(entry["plan"],folder,features,history,future,known,labels[future],original[28*24:])
                        item={"training":training,"future":value,"folder":str(folder.relative_to(dest))}
                    direct=np.r_[original[:28*24],item["future"]]
                    bank["direct-"+key]=direct
                    bank["gated-"+key]=direct if gate(np.asarray(parent_errors),np.asarray(item["training"]["history_rmse"])) else original.copy()
                except (KeyError,ValueError,TypeError,SyntaxError,subprocess.TimeoutExpired) as exc:
                    failures[key]=str(exc);bank["direct-"+key]=original.copy();bank["gated-"+key]=original.copy()
            atomic_json(dest/"fixed-failures.json",failures);np.savez_compressed(dest/"fixed-issued.npz",ids=service,**bank)
            issued_all[site].update(bank)
            if args.prior_run:
                with np.load(args.prior_run/site/"fixed-issued.npz") as saved:
                    if not np.array_equal(saved["ids"], service): raise ValueError("已有强固定的服务身份不同")
                    for key in saved.files:
                        if key == "ids": continue
                        if not np.array_equal(saved[key][:28*24], original[:28*24]): raise ValueError("已有固定的前半段不同")
                        issued_all[site]["prior-"+key] = saved[key].copy()
        scores={}
        for site in baseline["counts"]:
            truth=scene.readings(args.archive,site).to_numpy();_,_,_,service,_=scene.roles(truth);valid=np.isfinite(truth[service])
            for key,value in issued_all[site].items():scores.setdefault(key,{})[site]=error(truth[service][valid],value[valid])
        means={k:float(np.mean(list(v.values()))) for k,v in scores.items()}
        fixed=[k for k in means if k.startswith(("direct-","gated-","prior-")) or k in ("baseline","refit-direct","refit-gate")]
        best=min(fixed,key=means.get)
        result={"scope":"已见2025开发；前28日重用已冻结强固定交付，中途双方各最多3次提案；完整56日评分",
                "start":baseline["start"],"baseline":str(args.baseline),"counts":counts,"macro_rmse":means,"station_rmse":scores,
                "best_fixed":best,"fixed_plan_count":len(registry),"field_requests":len(list((args.output/"queue/requests").rglob('*.json'))),
                "maintenance_day":28,"reference_identity":identity(reference),"confirmation_visible_to_driver":False}
        result.update(edit_contract=args.edit_contract, prior_run=str(args.prior_run) if args.prior_run else None,
                      paper_strategy="MLE-STAR部件消融与定向改进本地适配" if args.edit_contract else "Self-Healing ML本地适配")
        atomic_json(args.output/"scores.json",result)
        print(json.dumps({"现场新增提案":means["project"],"同机会论文适配":means["paper"],"原强固定":means["baseline"],
                          "原办法直接补训":means["refit-direct"],"原办法确认补训":means["refit-gate"],"最强固定":means[best],
                          "实际现场请求":result["field_requests"],"固定候选方案":len(registry)},ensure_ascii=False))
