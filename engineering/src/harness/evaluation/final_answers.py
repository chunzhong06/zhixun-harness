"""将独立答案对应到当时实际发出的预测；不导入训练工具或修改运行状态。"""
from dataclasses import asdict

from harness.contracts import Sample, digest
from harness.evaluation.metrics import prediction_score


def service_scope(task, batches):
    """冻结完整服务范围和输入版本，重发资料不会再次成为待评分预测。"""
    scope = {}
    for batch in batches:
        if batch.task_id != task.task_id:
            raise ValueError("最终评价批次属于其他任务")
        for sample in batch.samples:
            if sample.role != "serve" or sample.available_at != batch.ordinal:
                continue
            if sample.sample_id in scope:
                raise ValueError("同一服务身份被重复发放")
            scope[sample.sample_id] = {"batch_id": batch.batch_id, "prediction_ordinal": batch.ordinal,
                "prediction_revision": sample.revision, "sample": asdict(sample)}
    return scope


def prepare_answers(task, batches, specification):
    """评价侧绑定任务、完整输入及答案修订；旧列表需调用者显式提供任务和批次。

    同一标签修订重复时拒绝；不同修订取最高版本，完整修订历史保留在摘要中。
    允许答案缺项以报告覆盖缺口，额外身份、错批次和无效目标则拒绝。
    """
    scope = service_scope(task, batches)
    input_digest = digest([asdict(b) for b in batches])
    legacy = isinstance(specification, list)
    if legacy:
        rows = specification
    else:
        if not isinstance(specification, dict) or specification.get("schema_version") != 1:
            raise ValueError("最终答案需要版本1清单或显式绑定的旧列表")
        if specification.get("task_signature") != task.signature or specification.get("input_digest") != input_digest:
            raise ValueError("最终答案与任务或完整输入不一致")
        rows = specification.get("answers")
    if not isinstance(rows, list):
        raise ValueError("最终答案记录须为列表")
    latest, versions = {}, {}
    for row in rows:
        if not isinstance(row, dict) or row.get("sample_id") not in scope:
            raise ValueError("最终答案包含未知服务身份")
        sid = row["sample_id"]
        expected = scope[sid]
        for field in ("batch_id", "prediction_ordinal", "prediction_revision"):
            if field in row and (type(row[field]) is not type(expected[field]) or row[field] != expected[field]):
                raise ValueError("最终答案的预测批次或输入版本不一致")
        revision = row.get("label_revision", 1)
        if type(revision) is not int or revision < 1 or not task.accepts_target(row.get("label")):
            raise ValueError("最终答案的标签修订或数值无效")
        if revision in versions.setdefault(sid, {}):
            raise ValueError("同一样本的标签修订重复")
        versions[sid][revision] = row["label"]
        if sid not in latest or revision > latest[sid]["label_revision"]:
            latest[sid] = {"sample_id": sid, "label": row["label"], "label_revision": revision,
                          **{k: expected[k] for k in ("batch_id", "prediction_ordinal", "prediction_revision")}}
    return {"task_signature": task.signature, "input_digest": input_digest, "scope": scope,
            "batch_digests": {b.batch_id: digest(asdict(b)) for b in batches}, "answers": latest,
            "source_digest": digest(specification), "legacy_explicit_binding": legacy,
            "revision_policy": "highest_declared_label_revision", "declared_versions": versions}


def score_final(task, state, prepared):
    """仅评价保存的服务输出；缺预测、答案或完成批次时整段分数留空。

    completed中的输入摘要绑定实际运行与本次答案清单。返回的评价记录只写到
    评价目录，不能作为维护反馈传给计划器；历史模型不在此重新预测。
    """
    if prepared["task_signature"] != task.signature or digest(state.get("task")) != task.signature:
        raise ValueError("最终评价与运行任务不一致")
    scope, answers = prepared["scope"], prepared["answers"]
    predictions = state.get("predictions", {})
    if set(predictions) - set(scope):
        raise ValueError("运行含本评价范围之外的预测")
    for bid, run in state.get("completed", {}).items():
        if prepared["batch_digests"].get(bid) != run.get("input_digest"):
            raise ValueError("实际运行批次内容与评价声明不一致")
    records, rows, values = [], [], {}
    for sid, pred in predictions.items():
        expected = scope[sid]
        if pred.get("batch_id") != expected["batch_id"] or pred.get("model_id") not in state.get("models", {}) or not task.accepts_target(pred.get("prediction")):
            raise ValueError("服务预测的批次、模型或数值无效")
        if pred["batch_id"] not in state.get("completed", {}):
            raise ValueError("预测所属批次尚未提交")
        committed=state["completed"][pred["batch_id"]]
        if (committed.get("predictions",{}).get(sid)!=pred["prediction"]
                or committed.get("model_id")!=pred["model_id"]
                or pred.get("sample_revision",expected["prediction_revision"])!=expected["prediction_revision"]):
            raise ValueError("服务预测与原提交回执或输入修订不一致")
        if sid not in answers:
            continue
        answer = answers[sid]
        raw = {**expected["sample"], "label": answer["label"], "role": "explore"}
        rows.append(Sample.parse(raw))
        values[sid] = pred["prediction"]
        records.append({**answer, "prediction": pred["prediction"], "model_id": pred["model_id"],
                        "group": expected["sample"]["group"], "source": "final_evaluation"})
    complete = (bool(scope) and set(predictions) == set(scope) == set(answers)
                and set(state.get("completed", {})) == set(prepared["batch_digests"]))
    return {"status": "completed" if complete else "incomplete", "scope": "all_declared_service_predictions",
            "evidence_kind": state.get("evidence_kind"), "model_experiment": state.get("model_experiment",False),
            "score": prediction_score(task, tuple(rows), values) if complete else None,
            "expected_samples": len(scope), "predicted_samples": len(predictions), "labeled_samples": len(answers),
            "missing_predictions": sorted(set(scope)-set(predictions)), "missing_answers": sorted(set(scope)-set(answers)),
            "source_digest": prepared["source_digest"], "revision_policy": prepared["revision_policy"],
            "legacy_explicit_binding": prepared["legacy_explicit_binding"],
            "declared_versions": prepared["declared_versions"], "input_digest": prepared["input_digest"],
            "run_snapshot_digest": digest(state), "objective": task.objective, "records": records}
