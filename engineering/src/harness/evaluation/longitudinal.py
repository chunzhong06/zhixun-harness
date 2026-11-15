"""从已发生的服务预测与已返回答案生成时间/分组评价，不用探索成绩代替服务效果。"""
from __future__ import annotations

import math
from collections import defaultdict

from harness.contracts import TaskSpec,digest
from harness.evaluation.costs import operation_costs
from harness.evaluation.metrics import macro_f1


def _score(task, records):
    if not records:
        return None
    if task.metric == "macro_f1":
        return macro_f1((r["label"], r["prediction"]) for r in records)
    errors=[abs(r["prediction"]-r["label"]) for r in records]
    if task.metric=="accuracy":
        return sum(v==0 for v in errors)/len(errors)
    mean=math.fsum(e*e if task.metric=="rmse" else e for e in errors)/len(errors)
    return math.sqrt(mean) if task.metric=="rmse" else mean


def evaluate_history(state, *, phase_by_batch=None, recovery=None, reference_report=None, final_report=None):
    """缺阶段定义时以批次显示，恢复目标须事前声明，缺答案保留待返回数量。

    固定参照通过独立评价器输入，保持自己的样本范围和费用；服务分组
    仍按实际生产输入汇总，两者不按同名分组自动混合。
    """
    task=TaskSpec.parse(state["task"])
    if reference_report and (reference_report["run_fingerprint"]!=state["fingerprint"]
                             or reference_report["run_snapshot_digest"]!=digest(state)):
        raise ValueError("固定参照报告属于其他运行")
    answers={}
    for item in state["feedback"].values():
        if item["source"]=="service_feedback" and (item["sample_id"] not in answers or
                item["label_revision"]>answers[item["sample_id"]]["label_revision"]):
            answers[item["sample_id"]]=item
    if final_report is not None:
        if final_report["run_snapshot_digest"] != digest(state):
            raise ValueError("终局答案报告与运行快照不一致")
        answers = {r["sample_id"]: r for r in final_report["records"]}
    phases=defaultdict(list)
    groups=defaultdict(list)
    curve=[]
    for bid,batch in state["completed"].items():
        predictions=[(sid,p) for sid,p in state["predictions"].items() if p["batch_id"]==bid]
        records=[]
        local_groups=defaultdict(list)
        for sid,pred in predictions:
            if sid not in answers:
                continue
            item={**answers[sid],"group":pred.get("group",state["samples"].get(sid,{}).get("group","unspecified"))}
            records.append(item)
            local_groups[item["group"]].append(item)
        phase=(phase_by_batch or {}).get(bid,bid)
        phases[phase].extend(records)
        value=_score(task,records)
        entry={"batch_id":bid,"phase":phase,"score":value,"labeled":len(records),
               "predicted":len(predictions),"pending_labels":len(predictions)-len(records),
               "complete_feedback":len(records)==len(predictions) and bool(records),
               "groups":{g:{"score":_score(task,r),"count":len(r)} for g,r in local_groups.items()}}
        curve.append(entry)
        for group,rows in local_groups.items():
            groups[group].append({"batch_id":bid,"score":_score(task,rows),"count":len(rows)})
    group_history={}
    for group,values in groups.items():
        latest=task.utility(values[-1]["score"])
        best=max(task.utility(v["score"]) for v in values)
        group_history[group]={"curve":values,"observed_utility_drop":best-latest,
            "comparison":"observed_service_group","fixed_reference_forgetting":None}
    phase_scores={p:{"score":_score(task,r),"count":len(r),
                    "pending_labels":sum(x["pending_labels"] for x in curve if x["phase"]==p)} for p,r in phases.items()}
    measured={p:r for p,r in phase_scores.items() if r["score"] is not None}
    worst=min(measured,key=lambda p:task.utility(measured[p]["score"])) if measured else None
    recovery_results=[]
    for request in recovery or []:
        onset=request["onset_batch"]
        target=float(request["target_utility"])
        consecutive=request.get("consecutive",1)
        if not math.isfinite(target) or type(consecutive) is not int or consecutive<1:
            raise ValueError("恢复定义需要有限目标与正整数连续批数")
        indices=[i for i,x in enumerate(curve) if x["batch_id"]==onset]
        if not indices:
            recovery_results.append({**request,"status":"onset_unavailable","batches_to_recovery":None})
            continue
        start=indices[0]
        streak=0
        found=None
        for i in range(start,len(curve)):
            row=curve[i]
            streak=streak+1 if row["complete_feedback"] and task.utility(row["score"])>=target else 0
            if streak>=consecutive:
                found=i-consecutive+1-start
                break
        recovery_results.append({**request,"status":"recovered" if found is not None else "not_observed",
                                 "batches_to_recovery":found})
    return {"objective":task.objective,"evidence_kind":state["evidence_kind"],
            "answer_source":"final_evaluation" if final_report is not None else "runtime_feedback",
            "phase_definition":"supplied" if phase_by_batch else "batch_as_display_unit",
            "curve":curve,"phases":phase_scores,"worst_observed_phase":worst,
            "group_history":group_history,"recovery":recovery_results,
            "recovery_definition":"supplied" if recovery else "not_supplied",
            "cost":operation_costs(list(state["operations"].values())),
            "fixed_reference":reference_report or {"status":"not_configured"}}
