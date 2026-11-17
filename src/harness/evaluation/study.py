"""只读已发出的预测，分开维护表现、未来留出、共同恢复目标和费用。"""
from collections import Counter
from copy import deepcopy
import math

from harness.core.contracts import digest
from harness.evaluation.final_answers import service_scope
from harness.evaluation.metrics import macro_f1
from harness.evaluation.costs import operation_costs
from harness.evaluation.problem_results import evaluate_problems


def validate_definition(task, batches, definition):
    """阶段须按输入顺序完整且互斥；恢复阈值是预声明的研究标准。"""
    required = {"schema_version", "baseline_id", "initial_batch", "maintenance_batches", "holdout_batches", "recovery"}
    if task.task_type != "classification" or set(definition) != required or definition["schema_version"] != 1:
        raise ValueError("当前研究评价需要明确的分类任务阶段定义")
    ids = [b.batch_id for b in batches]
    if (not isinstance(definition["initial_batch"],str) or not definition["baseline_id"]
            or not isinstance(definition["maintenance_batches"],list) or not definition["maintenance_batches"]
            or not isinstance(definition["holdout_batches"],list) or not definition["holdout_batches"]
            or [definition["initial_batch"], *definition["maintenance_batches"], *definition["holdout_batches"]] != ids
            or len(set(ids)) != len(ids)):
        raise ValueError("阶段必须依次覆盖全部批次且不能重复")
    recovery = definition["recovery"]
    if (not isinstance(recovery,dict) or set(recovery) != {"drop", "tolerance", "consecutive"}
            or any(type(recovery[k]) not in {int,float} or not math.isfinite(recovery[k]) for k in ("drop","tolerance"))
            or not 0 <= recovery["tolerance"] < recovery["drop"] <= 1
            or type(recovery["consecutive"]) is not int or recovery["consecutive"] < 1):
        raise ValueError("恢复定义需要合理下降量、容忍量与连续批数")


def _batch_scores(task, scope, records):
    expected = Counter(row["batch_id"] for row in scope.values())
    result = {}
    for bid, count in expected.items():
        rows = [r for r in records if r["batch_id"] == bid]
        complete = len(rows) == count
        correct = sum(r["prediction"] == r["label"] for r in rows)
        recall = {}
        for label in task.classes:
            members = [r for r in rows if r["label"] == label]
            recall[str(label)] = {"support": len(members), "recall": (
                sum(r["prediction"] == label for r in members) / len(members) if members and complete else None)}
        result[bid] = {"complete": complete, "expected": count, "observed": len(rows),
            "correct": correct, "accuracy": correct / count if complete and count else None, "class_recall": recall}
        if task.metric == "macro_f1":
            result[bid]["macro_f1"] = macro_f1((r["label"], r["prediction"]) for r in rows) if complete and rows else None
    return result


def representative_process(state):
    """取最早有实际尝试的问题批次；展示当批动作，不把同时出现解释为因果归属。"""
    problems = sorted(state.get("problems", {}).values(), key=lambda p: (p["discovered_ordinal"], p["id"]))
    for p in problems:
        bid = p["discovered_batch"]
        batch = state.get("completed", {}).get(bid, {})
        if not batch.get("trials"):
            continue
        checks = [x for x in state.get("inspections", {}).get(bid, [])
                  if x.get("problem_id") == p["id"] or x.get("scope") == p["scope"]]
        return {"status": "available", "selection": "first_registered_problem_with_batch_trials",
            "problem": {k:deepcopy(p.get(k)) for k in ("id","label","scope","original_loss","original_target","observations")},
            "batch_id": bid, "checks": deepcopy(checks), "batch_attempts": deepcopy(batch["trials"]),
            "adoption": deepcopy(batch["decision"]), "served_model": batch["model_id"],
            "later_feedback": deepcopy(p.get("service_observations", [])),
            "attribution": "当批实际尝试与采用；动作是否针对问题由计划和检查引用核对"}
    return {"status": "no_eligible_problem", "selection": "first_registered_problem_with_batch_trials"}


def evaluate_study(task, batches, definition, final_reports, states):
    """评价侧调用；不写运行状态，不预测或训练，不把评价答案交给计划器。

    输入应来自score_final的原服务报告。另核对快照、输入、样本及标签版本，
    防止混合不同运行或不同答案。缺输出时保留缺项，不靠删除批次得到较高均值。
    """
    validate_definition(task, batches, definition)
    scope = service_scope(task, batches)
    from dataclasses import asdict
    input_digest = digest([asdict(b) for b in batches])
    cases, label_identity = {}, None
    for key, state in states.items():
        report = final_reports.get(key)
        if not state or not report:
            cases[key] = {"status": "missing_final_report", "stages": {}, "batches": {},
                          "cost": operation_costs(list(state.get("operations", {}).values())) if state else None}
            continue
        if report["run_snapshot_digest"] != digest(state) or report["input_digest"] != input_digest:
            raise ValueError("研究报告与实际运行或输入不一致")
        records = report["records"]
        ids = [r["sample_id"] for r in records]
        if len(set(ids)) != len(ids) or not set(ids) <= set(scope):
            raise ValueError("研究评价样本重复或超出范围")
        for row in records:
            sid = row["sample_id"]
            original = state["predictions"][sid]
            if (row["batch_id"] != scope[sid]["batch_id"] or row["prediction"] != original["prediction"]
                    or row["model_id"] != original["model_id"]):
                raise ValueError("研究评价改变了原发出的预测或版本")
        current_labels = {r["sample_id"]: (r["label"], r["label_revision"]) for r in records}
        if label_identity is None:
            label_identity = current_labels.copy()
        else:
            if any(label_identity[sid] != value for sid,value in current_labels.items() if sid in label_identity):
                raise ValueError("比较组使用了不同的最终答案修订")
            label_identity.update(current_labels)
        scores = _batch_scores(task, scope, records)
        stages = {}
        phases = {"initial": [definition["initial_batch"]], "maintenance": definition["maintenance_batches"],
                  "holdout": definition["holdout_batches"]}
        for phase, ids in phases.items():
            values = [scores.get(bid, {}) for bid in ids]
            complete = all(v.get("complete") and v.get("accuracy") is not None for v in values)
            observed = sum(v.get("observed", 0) for v in values)
            stages[phase] = {"complete": complete, "batch_ids": ids, "observed": observed,
                "mean_batch_accuracy": math.fsum(v["accuracy"] for v in values)/len(values) if complete else None,
                "pooled_accuracy": sum(v["correct"] for v in values)/observed if complete and observed else None,
                "worst_batch": min(ids, key=lambda bid:scores[bid][task.metric]) if complete else None}
            if task.metric == "macro_f1":
                stages[phase]["mean_batch_macro_f1"] = math.fsum(v["macro_f1"] for v in values)/len(values) if complete else None
                # 汇总F1按所有有效预测重算，逐批均值另列以免稀少类别被重复加权。
                stages[phase]["pooled_macro_f1"] = macro_f1((r["label"], r["prediction"]) for r in records
                                                            if r["batch_id"] in ids) if complete else None
        last = state.get("completed",{}).get(definition["maintenance_batches"][-1],{}).get("model_id")
        holdout = [state.get("completed",{}).get(bid,{}).get("model_id") for bid in definition["holdout_batches"]]
        cases[key] = {"status": report["status"], "evidence_kind":state["evidence_kind"],
            "model_experiment":state["model_experiment"], "batches":scores, "stages":stages,
            "holdout_version_continuity": all(mid==last for mid in holdout) if last and all(holdout) else None,
            "cost":operation_costs(list(state["operations"].values())), "representative_process":representative_process(state),
            "frozen_problem_results":evaluate_problems(state,scope,records)}
        first_model=state.get("models",{}).get(definition["initial_batch"]+"/initialize")
        cases[key]["initial_model_identity"] = digest({k:first_model.get(k) for k in
            ("pipeline","training_versions","artifact")}) if first_model else None
    reference = cases.get(definition["baseline_id"],{}).get("batches",{})
    initial = reference.get(definition["initial_batch"],{}).get(task.metric)
    rule = definition["recovery"]
    target = max(0.,initial-rule["tolerance"]) if initial is not None else None
    onset = None
    baseline_complete = initial is not None and all(reference.get(bid,{}).get(task.metric) is not None
                                                   for bid in definition["maintenance_batches"])
    if baseline_complete:
        onset = next((bid for bid in definition["maintenance_batches"]
                      if reference[bid][task.metric] < max(0.,initial-rule["drop"])),None)
    for case in cases.values():
        result = {"status": "baseline_unavailable" if not baseline_complete else "not_triggered",
                  "onset_batch":onset,f"target_{task.metric}":target,"batches_to_recovery":None}
        if onset is not None:
            ids = definition["maintenance_batches"]
            start, streak, found = ids.index(onset), 0, None
            for index,bid in enumerate(ids[start:], start):
                value = case["batches"].get(bid,{}).get(task.metric)
                streak = streak+1 if value is not None and value >= target else 0
                if streak >= rule["consecutive"]:
                    found = index-rule["consecutive"]+1-start
                    break
            complete = case["stages"].get("maintenance",{}).get("complete",False)
            result.update(status="incomplete" if not complete else "recovered" if found is not None else "not_recovered_by_end",
                          batches_to_recovery=found if complete else None)
        case["recovery"] = result
    initial_ids=[case.get("initial_model_identity") for case in cases.values()]
    initial_match = len(set(initial_ids))==1 if initial_ids and all(initial_ids) else None
    return {"schema_version":1,"definition":deepcopy(definition),"input_digest":input_digest,
            "answer_source":"independent_final_answers","cases":cases,
            "initial_model_identity_match":initial_match,
            "primary_measure":f"maintenance.mean_batch_{task.metric}","winner":None}


def render_study(report):
    """以固定字段生成人可读结果；没有观察到的内容直接保留对应状态。"""
    def number(value, percent=False):
        return "未形成完整结果" if value is None else f"{100*value:.2f}%" if percent else f"{value:.3g}"
    labels={"static":"保持初始模型","periodic":"每批近期重训","history":"合理历史维护",
        "mature":"同工具成熟选择","problem":"问题驱动方案","adaptive_reward":"按问题检查＋收益选择",
        "fixed_opportunity":"固定检查＋机会补偿"}
    recovery_names={"baseline_unavailable":"共同参照未齐","not_triggered":"未触发预设下降条件",
        "incomplete":"反馈未齐","recovered":"已观察到恢复","not_recovered_by_end":"观察结束仍未恢复"}
    real=any(case.get("model_experiment") for case in report["cases"].values())
    metric = report["primary_measure"].removeprefix("maintenance.mean_batch_")
    metric_name = "Macro-F1" if metric == "macro_f1" else "准确率"
    lines=["# 维护结果", "", "真实执行记录的评价。" if real else "人工模型与预写预测的工程演练，用于核对流程和计算。", "",
        "维护段与未来留出分别评价；费用使用实际操作记录。", "",
        f"| 方案 | 维护段逐批平均{metric_name} | 未来留出汇总{metric_name} | 恢复状态 | 已测操作耗时（秒） |",
        "| --- | --- | --- | --- | --- |"]
    for key,case in report["cases"].items():
        stages=case["stages"];recovery=case["recovery"]
        lines.append(f"| {labels.get(key,key)} | {number(stages.get('maintenance',{}).get('mean_batch_'+metric),True)} | "
            f"{number(stages.get('holdout',{}).get('pooled_'+metric),True)} | {recovery_names[recovery['status']]} | "
            f"{number((case.get('cost') or {}).get('measured_elapsed_seconds'))} |")
    lines += ["", "操作耗时汇总不包含未测步骤；工作进程CPU、峰值内存、失败和未知用量详见study-evaluation.json。", ""]
    chosen=report["cases"].get("problem",{})
    process=chosen.get("representative_process",{})
    if process.get("status") == "available":
        issue=process["problem"]
        lines += ["## 代表维护过程", "", f"取最早符合条件的问题批次：{process['batch_id']}。",
            f"原问题：{issue.get('label')}；范围：{issue.get('scope')}。", "",
            f"发现时的范围内损失：{number(issue.get('original_loss'))}；固定目标：{number(issue.get('original_target'))}。", "",
            "本批实际尝试："]
        recipes={"recent":"近期资料","uniform":"普通历史回放","coverage":"类别覆盖回放"}
        predictors={"linear":"线性模型","forest":"随机森林","catboost":"提升树"}
        for trial in process["batch_attempts"]:
            action=trial.get("action_plan",{})
            pipe=action.get("pipeline") or {}
            recipe=action.get('recipe',trial.get('action'))
            predictor=pipe.get('predictor','继承当前配置')
            lines.append(f"- 资料方式：{recipes.get(recipe,recipe)}；预测器：{predictors.get(predictor,predictor)}；"
                f"探索准确率：{number(trial.get('exploration_score'),True)}；执行状态：{trial.get('status')}。")
        decision=process["adoption"]
        lines += ["", f"采用决定：{decision['reason']}。实际服务版本：{process['served_model']}。", "",
                  "后续表现："]
        for observation in process["later_feedback"]:
            lines.append(f"- {observation['prediction_batch']}：{observation['status']}；范围内损失={observation.get('loss')}，观察数={observation.get('support')}。")
        if not process["later_feedback"]:
            lines.append("- 尚未收到该问题范围的后续反馈。")
        final_problem=chosen.get("frozen_problem_results",{}).get(issue["id"],{})
        if final_problem:
            lines += ["", "最终答案核对（各列使用同一实际服务版本）：",
                "", "| 批次 | 原问题范围准确率 | 范围外准确率 | 整体准确率 |", "| --- | --- | --- | --- |"]
            for observation in final_problem["observations"]:
                lines.append(f"| {observation['batch_id']} | {number(observation['inside']['accuracy'],True)} | "
                    f"{number(observation['outside']['accuracy'],True)} | {number(observation['overall']['accuracy'],True)} |")
        lines += ["", "检查引用、原始目标及候选得失保留在JSON中；本批动作的对应关系按实际记录解释。"]
    else:
        lines += ["尚无符合预设条件的代表问题，保留全部运行记录。"]
    return "\n".join(lines)+"\n"
