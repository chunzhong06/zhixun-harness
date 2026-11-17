"""把原问题、合法观察、尝试与后续复查相连；规划目标不改变正式采用门槛。

Iris启发证据状态，Mixture-of-Depths启发有容量的深入处理。这里以明确的
误差目标和有限检查次数实现，不使用未来答案或额外神经路由器。
"""
from copy import deepcopy
import math

from harness.core.contracts import Sample, digest
from harness.evaluation.metrics import compare_evaluations, prediction_score
from harness.learning.models.corrections import inside


def _losses(task, evaluation):
    score = evaluation["score"]
    result = {"overall": {"label": "整体预测误差", "loss": 1-score if task.task_type == "classification" else score,
                           "count": len(evaluation.get("sample_versions", []))}}
    for group in evaluation.get("groups", []):
        result["group:"+group["group"]] = {"label": f"资料组{group['group']}的预测误差",
            "loss": 1-group["score"] if task.task_type == "classification" else group["score"], "count": group["count"]}
    for row in (evaluation.get("error_profile") or {}).get("classes", []):
        if row["count"]:
            result[f"class:{row['label']}"] = {"label": f"类别{row['label']}的漏判",
                "loss": row["missed"]/row["count"], "count": row["count"]}
    return result


def _problem_losses(engine,evaluation):
    """问题观察优先看本批资料；选取指标仍按独立的窗口权重比较完整候选。"""
    windows=evaluation.get("observation_windows",{})
    window="current" if "current" in windows else "returned" if "returned" in windows else "all"
    view=windows.get(window,evaluation)
    result=_losses(engine.task,view)
    ids={v.rsplit("@",1)[0] for v in view.get("sample_versions",[])}
    for record in engine.state.get("problems",{}).values():
        if not record["scope"].startswith("region:"):
            continue
        rows=tuple(Sample.parse(engine.state["samples"][sid]) for sid in sorted(ids)
                   if sid in engine.state["samples"] and inside(engine.state["samples"][sid]["values"],record["predicates"]))
        if rows:
            score=prediction_score(engine.task,rows,{s.sample_id:evaluation["predictions"][s.sample_id] for s in rows})
            result[record["scope"]]={"label":record["label"],"loss":1-score if engine.task.task_type=="classification" else score,
                                    "count":len(rows),"member_ids":[r.sample_id for r in rows]}
    return result,window


def _register_regions(engine,batch):
    """相同条件共享问题身份；条件变动产生新身份，保留每次发现的来源。"""
    problems=engine.state.setdefault("problems",{})
    for checks in engine.state["inspections"].values():
        for check in checks:
            for region in check.get("regions",[]):
                predicates=sorted(region["predicates"],key=lambda p:(p["index"],p["side"],p["cut"]))
                signature=digest([(p["index"],p["side"],p["cut"]+0.) for p in predicates])
                scope="region:"+signature
                key="problem-"+digest([engine.task.signature,scope])[:20]
                if key not in problems:
                    problems[key]={"id":key,"scope":scope,"label":"冻结输入条件内的预测误差",
                        "discovered_batch":batch.batch_id,"discovered_ordinal":batch.ordinal,
                        "discovered_model":region["model_id"],"predicates":deepcopy(predicates),
                        "region_ids":[],"observations":[]}
                if region["id"] not in problems[key]["region_ids"]:
                    problems[key]["region_ids"].append(region["id"])


def refresh_problems(engine, batch, baseline, trials):
    """每批冻结在用模型的原误差，完整探索候选更新进展；跨批保留同一问题身份。"""
    if not engine.config.problem_tracking or baseline is None:
        return {"enabled": False, "items": []}
    _register_regions(engine,batch)
    evaluations = [baseline]
    for trial in trials:
        if trial.get("status") != "completed" or not trial.get("final_resource", True) or trial.get("resource_fraction", 1.) != 1.:
            continue
        op = engine.state["operations"].get(f"{batch.batch_id}:explore:{trial.get('model_id')}", {})
        value = op.get("value")
        if op.get("status") == "completed" and value and compare_evaluations(engine.task, baseline, value)["status"] == "matched":
            evaluations.append(value)
    original,window = _problem_losses(engine,baseline)
    evaluated={e["model_id"]:_problem_losses(engine,e)[0] for e in evaluations}
    state = engine.state.setdefault("problems", {})
    identifiers = set(original) | {r["scope"] for r in state.values()}
    items = []
    for scope in sorted(identifiers):
        candidates=[e for e in evaluations if scope in evaluated[e["model_id"]]]
        chosen=min(candidates,key=lambda e:(evaluated[e["model_id"]][scope]["loss"],e["model_id"]!=baseline["model_id"],e["model_id"])) if candidates else baseline
        first, latest = original.get(scope), evaluated[chosen["model_id"]].get(scope)
        key = "problem-" + digest([engine.task.signature,scope])[:20]
        record = state.get(key)
        if record is None and (first is None or first["loss"] <= 0 or first["count"] < engine.config.problem_min_support):
            continue
        if record is None:
            record = {"id": key, "scope": scope, "label": first["label"],
                      "discovered_batch": batch.batch_id,"discovered_ordinal":batch.ordinal,
                      "discovered_model": baseline["model_id"], "observations": []}
            state[key] = record
        before = first["loss"] if first else None
        after = latest["loss"] if latest else None
        sufficient = latest is not None and latest["count"] >= engine.config.problem_min_support
        target = before*(1-engine.config.problem_reduction_target) if before is not None else None
        if "original_target" not in record and target is not None:
            record.update(original_target=target,original_loss=before)
        status = ("unobserved" if latest is None else "insufficient_support" if not sufficient or target is None else
                  "target_met_on_exploration" if after <= target + 1e-12 else "open")
        evidence_key = f"{batch.batch_id}:explore:{chosen['model_id']}"
        observation = {"batch_id": batch.batch_id, "model_id": chosen["model_id"], "baseline_model": baseline["model_id"],
            "before": before, "after": after, "target": target, "support": latest["count"] if latest else 0,
            "status": status, "evaluation_key": evidence_key, "sample_revision_digest": digest(chosen.get("sample_versions", [])),
            "observation_window":window,"original_target":record.get("original_target"),
            "member_ids":latest.get("member_ids",[]) if latest else [],
            "candidate_observations":[{"model_id":e["model_id"],"loss":evaluated[e["model_id"]][scope]["loss"],
                "support":evaluated[e["model_id"]][scope]["count"],"evaluation_key":f"{batch.batch_id}:explore:{e['model_id']}"} for e in candidates],
            "attempted_actions": sorted({t["action"] for t in trials})}
        oid = digest(observation)
        if not any(o["id"] == oid for o in record["observations"]):
            record["observations"].append({"id": oid, **observation})
            record["observations"] = record["observations"][-8:]
        record["latest"] = observation
        items.append({"id": key, "scope": scope, "label": record["label"], **observation,
                      "service":deepcopy(record.get("service")),"latest_feedback":deepcopy(record.get("latest_feedback"))})
    return {"enabled": True, "items": items, "target_kind": "planning_reduction_not_adoption_threshold"}


def problem_router(context, state, config, batch_id):
    """未解决且有支持的问题竞争深查名额；每个问题和本批总深度均有上限。"""
    facts = next((f["value"] for f in context["facts"]["evidence"] if f["kind"] == "problems"), {"items": []})
    events = [e for e in state["events"] if e["kind"] == "problem_check" and e["batch_id"] == batch_id]
    observed_checks = {}
    for event in events:
        for pid in event["problem_ids"]:
            observed_checks.setdefault(pid,set()).add(event["check_key"])
    depths = {pid:len(keys) for pid,keys in observed_checks.items()}
    eligible_states = {"open", "target_met_on_exploration"} if config.problem_depth == "fixed" else {"open"}
    ranked = sorted([i for i in facts["items"] if i["status"] in eligible_states and depths.get(i["id"],0) < config.problem_max_depth],
                    key=lambda i: (-(i["after"] or 0),-i["support"],i["id"]))
    # 同一批上覆盖完全相同的区域只占一个深查名额；保留各自定义和历史。
    unique,seen=[],set()
    for item in ranked:
        members=tuple(sorted(item.get("member_ids",[])))
        if members and members in seen: continue
        if members: seen.add(members)
        unique.append(item)
    ranked=unique
    if config.problem_depth == "rotation" and ranked:
        offset = len(events) % len(ranked)
        ranked = ranked[offset:]+ranked[:offset]
    if config.problem_depth == "fixed":
        ranked = sorted(ranked,key=lambda i:i["id"])
    selected = ranked[:config.problem_capacity]
    remaining = max(0,config.problem_check_budget-len({e["check_key"] for e in events}))
    if selected:
        remaining = min(remaining,min(config.problem_max_depth-depths.get(i["id"],0) for i in selected))
    return {"enabled": config.problem_tracking, "mode": config.problem_depth,
        "selected_ids": [i["id"] for i in selected], "selected": selected, "depths": depths,
        "remaining_checks": remaining, "allow_inspection": bool(selected and remaining),
        "performed_tools": {i["id"]: [e["tool"] for e in events if i["id"] in e["problem_ids"]] for i in selected}}


def propose_problem_check(context, options):
    """规则对照先试改，再按原问题选择尚能区分两种后续动作的只读检查。"""
    m = context.get("maintenance", {})
    router = m.get("problem_router", {})
    if not router.get("allow_inspection") or not context.get("trials") or m.get("stage") != "decision":
        return None
    candidates = {r["key"]:r for r in options.get("candidates", []) if not r["attempted"] and r.get("eligible",True)}
    rows = [candidates[k] for k in options.get("ranked_keys", []) if k in candidates]
    if len(rows) < 2 or context.get("remaining_demo_units",0) <= 7:
        return None
    problem = router["selected"][0]
    performed = router["performed_tools"][problem["id"]]
    tool=yes=no=None
    for candidate_tool in ("error_regions","feature_scale"):
        if candidate_tool not in m["available_checks"] or candidate_tool in performed: continue
        positive=("correction","predictor") if candidate_tool=="error_regions" else ("processor",)
        negative=("history","weight","retrain") if candidate_tool=="error_regions" else ("predictor","history","retrain")
        yes=next((r for kind in positive for r in rows if r["kind"]==kind),None)
        no=next((r for kind in negative for r in rows if r["kind"]==kind),None)
        if yes and no:
            tool=candidate_tool
            break
    if tool is None:
        return None
    fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "problems")
    return {"operation":"inspect", "reason": f"{problem['label']}尚未达到本轮目标，检查结果用于区分接下来的改法",
        "evidence_refs":[fact["id"]], "model_id":problem["model_id"],
        "checks":[{"tool":tool,"problem_id":problem["id"],"question":f"问题{problem['id']}："+
                   ("误差是否集中于可识别输入条件，值得尝试局部或灵活预测？" if tool=="error_regions" else "特征尺度差异是否值得先调整处理方式？"),
                   "threshold":0. if tool=="error_regions" else 100.,
                   "if_true":deepcopy(yes["action"]),"if_false":deepcopy(no["action"])}]}


def record_problem_adoption(engine,batch,decision,selected_id):
    """在实际服务版本确定后登记；候选采用后输出失败也按最终接续版本记录。"""
    if not engine.config.problem_tracking:
        return
    for record in engine.state.get("problems",{}).values():
        observation=record.get("latest",{})
        if observation.get("batch_id")!=batch.batch_id:
            continue
        entry={"batch_id":batch.batch_id,"ordinal":batch.ordinal,"current_model":decision["current_model"],
            "selected_model":selected_id,"exploration_model":observation["model_id"],
            "adopted":selected_id!=decision["current_model"],"output_fallback":bool(decision.get("output_fallback")),
            "reason":decision["reason"],"target":record.get("original_target")}
        entry["id"]=digest(entry)
        history=record.setdefault("adoptions",[])
        if not any(e["id"]==entry["id"] for e in history): history.append(entry)
        record["adoptions"]=history[-8:]
        record["service"]={"model_id":selected_id,"since_ordinal":batch.ordinal,
            "status":"awaiting_feedback","adopted":entry["adopted"],"adoption_id":entry["id"]}


def update_problem_feedback(engine):
    """按原发出批次/版本复查固定问题目标，迟到标签不会改写当时的预测。

    局部类别的完整覆盖需要整批答案齐备；未返回行的真实类别尚不可知。
    记录反映实际后续表现，不把不同批次变化归因为某一次改法的因果效果。
    """
    if not engine.config.problem_tracking:
        return
    state=engine.state
    labels={}
    for item in state["feedback"].values():
        if item["source"]=="service_feedback" and (item["sample_id"] not in labels or item["label_revision"]>labels[item["sample_id"]]["label_revision"]):
            labels[item["sample_id"]]=item
    for record in state.get("problems",{}).values():
        if record.get("original_target") is None:
            continue
        scope=record["scope"]
        groups={}
        for sid,pred in state["predictions"].items():
            if pred.get("ordinal",-1)<record["discovered_ordinal"]:
                continue
            if scope.startswith("group:") and pred["group"]!=scope.removeprefix("group:"): continue
            if scope.startswith("region:") and not inside(pred["input_values"],record["predicates"]): continue
            groups.setdefault((pred["ordinal"],pred["batch_id"],pred["model_id"]),[]).append(sid)
        history=record.setdefault("service_observations",[])
        for (ordinal,bid,mid),expected in sorted(groups.items()):
            available=[sid for sid in expected if sid in labels]
            if not available:
                continue
            members=[sid for sid in available if not scope.startswith("class:") or str(labels[sid]["label"])==scope.removeprefix("class:")]
            rows=tuple(Sample(sid,labels[sid]["label_revision"],tuple(state["predictions"][sid]["input_values"]),
                labels[sid]["label"],"explore",state["predictions"][sid]["group"],ordinal) for sid in members)
            score=prediction_score(engine.task,rows,{sid:state["predictions"][sid]["prediction"] for sid in members}) if rows else None
            loss=1-score if score is not None and engine.task.task_type=="classification" else score
            enough=len(members)>=engine.config.problem_min_support
            complete=len(available)==len(expected)
            status=("insufficient_support" if not enough else "partial_feedback" if not complete else
                    "target_met_on_service" if loss<=record["original_target"]+1e-12 else "open")
            entry={"prediction_batch":bid,"ordinal":ordinal,"model_id":mid,"loss":loss,"support":len(members),
                "expected_predictions":len(expected),"returned_answers":len(available),"status":status,
                "original_target":record["original_target"],"label_versions":[f"{sid}@{labels[sid]['label_revision']}" for sid in sorted(available)],
                "unit_count":len({state["predictions"][sid].get("unit_id") for sid in members if state["predictions"][sid].get("unit_id")}),
                "unknown_unit_rows":sum(state["predictions"][sid].get("unit_id") is None for sid in members)}
            # 同一发出批次的修订替换该观察；重算后再识别复发，避免旧标签结论残留。
            history[:]=[e for e in history if (e["prediction_batch"],e["model_id"])!=(bid,mid)]
            history.append(entry)
        history.sort(key=lambda e:(e["ordinal"],e["model_id"]))
        met=False
        for entry in history:
            if entry["status"]=="target_met_on_service": met=True
            elif entry["status"] in {"open","recurred"}: entry["status"]="recurred" if met else "open"
        # 保留每个服务批次的最新标签版本，迟到修订需要重算此前达标/复发关系。
        record["service_observations"]=history
        if history:
            latest=history[-1]
            record["latest_feedback"]=deepcopy(latest)
            service=record.get("service",{})
            if latest["model_id"]==service.get("model_id") and latest["ordinal"]>=service["since_ordinal"]:
                service.update(status=latest["status"],feedback=deepcopy(latest))
