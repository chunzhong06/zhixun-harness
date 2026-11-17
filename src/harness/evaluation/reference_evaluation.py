"""评价侧固定参照：重载已发布模型，比较同一批旧场景，不回写训练或采用状态。"""
from copy import deepcopy
from dataclasses import asdict
import math
from time import perf_counter

from harness.core.contracts import digest
from harness.engine.legacy.execution import JobUnconfirmed
from harness.evaluation.longitudinal import _score
from harness.evaluation.costs import operation_costs
from harness.adapters.storage.local import LocalStore, atomic_json


def validate_reference(task, specification, state=None):
    """参照集由评价者事先提供；身份、输入和答案一起固定，不能混入运行资料。

    anchor_batch声明开始观察旧能力的批次。若提供独立单位，则同时检查它与
    运行资料的单位隔离；缺单位时只声称已核对样本身份，不推断设备独立性。
    """
    if set(specification)!={"schema_version","task_id","sets"} or specification["schema_version"]!=1:
        raise ValueError("固定参照清单格式无效")
    if specification["task_id"]!=task.task_id or not specification["sets"]:
        raise ValueError("固定参照任务不匹配或集合为空")
    sets,ids=set(),set()
    for group in specification["sets"]:
        if set(group)!={"set_id","anchor_batch","source","samples"} or not group["samples"]:
            raise ValueError("每个参照需要名称、起点批次、来源和样本")
        if any(not isinstance(group[k],str) or not group[k].strip() for k in ("set_id","anchor_batch","source")):
            raise ValueError("固定参照声明不能为空")
        if group["set_id"] in sets:
            raise ValueError("参照集合身份重复")
        sets.add(group["set_id"])
        for row in group["samples"]:
            if set(row)-{"sample_id","values","label","group","unit_id"} or not {"sample_id","values","label","group"}<=set(row):
                raise ValueError("参照样本字段无效")
            if any(not isinstance(row[k],str) or not row[k] for k in ("sample_id","group")) or row["sample_id"] in ids:
                raise ValueError("参照样本身份无效或重复")
            ids.add(row["sample_id"])
            if len(row["values"])!=len(task.features) or any(type(v) not in (int,float) or not math.isfinite(v) for v in row["values"]):
                raise ValueError("参照特征无效")
            if not task.accepts_target(row["label"]):
                raise ValueError("参照目标无效")
            if "unit_id" in row and (not isinstance(row["unit_id"],str) or not row["unit_id"]):
                raise ValueError("参照独立单位无效")
    if state is not None:
        if state["task"]!=asdict(task) and digest(state["task"])!=digest(asdict(task)):
            raise ValueError("运行与固定参照任务不一致")
        used=set(state["samples"])
        used.update(s for model in state["models"].values() for s in model["training_ids"])
        if ids & used:
            raise ValueError("固定参照与运行资料的样本身份重叠")
        units={s.get("unit_id") for s in state["samples"].values()}-{None}
        if any(row.get("unit_id") in units for group in specification["sets"] for row in group["samples"]):
            raise ValueError("固定参照与运行资料的独立单位重叠")
    return digest(specification)


def reference_tools(engine, folder):
    """沿用已发布模型仓库，给评价创建独立预测记录与限时监督。"""
    from harness.learning.models.training_tools import TrainingModelTools
    from harness.engine.legacy.process_tools import ProcessTrainingTools
    from harness.engine.legacy.execution import LocalJobRunner
    from harness.adapters.executors.fixture_backend import FixtureModelTools
    tools=engine.tools
    if isinstance(tools,ProcessTrainingTools):
        runner=LocalJobRunner(folder/'jobs',tools.artifacts,tools.runner.limits)
        return ProcessTrainingTools(engine.task,tools.backend,tools.artifacts,runner)
    if isinstance(tools,TrainingModelTools):
        return TrainingModelTools(engine.task,tools.backend,tools.artifacts)
    if isinstance(tools,FixtureModelTools):
        return FixtureModelTools(tools.fixture)
    raise ValueError("该后端需要提供独立的固定参照预测适配器")


def _measures(task, rows, predictions):
    if set(predictions)!={r["sample_id"] for r in rows} or any(not task.accepts_target(v) for v in predictions.values()):
        raise ValueError("固定参照预测缺失或取值无效")
    records=[{**r,"prediction":predictions[r["sample_id"]]} for r in rows]
    return {"score":_score(task,records),"groups":{g:{"score":_score(task,[r for r in records if r["group"]==g]),
            "count":sum(r["group"]==g for r in records)} for g in sorted({r["group"] for r in rows})}}


def evaluate_references(task, state, specification, tools, folder, *, allow_real_execution=False, after_prediction=None):
    """在已完成运行快照上评价，按模型/参照缓存结果；失败项与费用一并保留。

    输入与标签留在本评价器，predict只接收身份和特征；不调用build。started
    任务须由recover_predict确认，不能确认则等待。after_prediction仅用于
    工程注入：模拟预测完成但本评价器尚未保存回执的中断。
    """
    if state["evidence_kind"]!=tools.evidence_kind:
        raise ValueError("固定参照预测器的执行来源不匹配")
    if tools.evidence_kind=="real_model" and not (allow_real_execution and getattr(tools,"capabilities",{}).get("supervised_execution")):
        raise ValueError("真实固定参照评价需要执行许可和受控预测器")
    spec_id=validate_reference(task,specification,state)
    identity=digest(["fixed-reference-v1",digest(state),spec_id,tools.identity])
    store=LocalStore(folder)
    with store.exclusive():
        saved=store.load() or {"identity":identity,"operations":{}}
        if saved["identity"]!=identity:
            raise ValueError("固定参照、运行快照或预测器改变，请使用新评价目录")
        store.save(saved)
        versions=[]
        for batch in state["completed"].values():
            for mid in (batch["decision"]["current_model"],batch["model_id"]):
                if mid not in versions:versions.append(mid)
        for group in specification["sets"]:
            if group["anchor_batch"] not in state["completed"]:continue
            rows=group["samples"]
            inputs=tuple({"sample_id":r["sample_id"],"values":deepcopy(r["values"])} for r in rows)
            for mid in versions:
                key=digest([group["set_id"],mid])
                op=saved["operations"].get(key)
                artifact=state["models"][mid]["artifact"]
                if op and op["status"] in {"completed","failed"}:continue
                if op:
                    receipt=tools.recover_predict(artifact,inputs) if hasattr(tools,"recover_predict") else None
                    if not receipt or receipt["status"] not in {"completed","failed"}:
                        raise JobUnconfirmed("固定参照预测尚未确认")
                    op.update(execution=deepcopy(receipt),elapsed_seconds=receipt.get("wall_seconds"),recovered=True)
                    if receipt["status"]=="failed":
                        op.update(status="failed",error=receipt.get("failure","prediction_failed"))
                        store.save(saved);continue
                    predictions=receipt["artifact"]
                else:
                    op={"name":"reference_predict","status":"started","model_id":mid,"set_id":group["set_id"],"demo_units":0}
                    saved["operations"][key]=op;store.save(saved)
                    start=perf_counter()
                    try:
                        predictions=tools.predict(artifact,inputs)
                    except JobUnconfirmed:
                        raise
                    except Exception as exc:
                        op.update(status="failed",error=type(exc).__name__,elapsed_seconds=perf_counter()-start,
                                  execution=deepcopy(getattr(tools,"last_execution",None)))
                        store.save(saved);continue
                    op.update(elapsed_seconds=perf_counter()-start,execution=deepcopy(getattr(tools,"last_execution",None)))
                    if after_prediction:after_prediction(key)
                try:
                    measures=_measures(task,rows,predictions)
                    op.update(status="completed",predictions=predictions,**measures)
                except (ValueError,TypeError) as exc:
                    op.update(status="failed",error=type(exc).__name__)
                store.save(saved)
        result=summarize_references(task,state,specification,saved)
        atomic_json(store.folder/'reference-report.json',result)
        _write_explanation(store.folder,result)
        return result


def summarize_references(task, state, specification, saved):
    """同时保留起点到当前变化和历史最佳到当前的下降，指标方向由任务统一转换。"""
    results=[];batches=list(state["completed"])
    for group in specification["sets"]:
        anchor=group["anchor_batch"]
        if anchor not in batches:
            results.append({"set_id":group["set_id"],"status":"anchor_unavailable","curve":[]});continue
        curve=[];first=best=None;complete=True
        for bid in batches[batches.index(anchor):]:
            batch=state["completed"][bid];mid=batch["model_id"]
            current=saved["operations"][digest([group["set_id"],mid])]
            previous=saved["operations"][digest([group["set_id"],batch["decision"]["current_model"]])]
            good=current["status"]=="completed"
            complete=complete and good
            value=task.utility(current["score"]) if good else None
            if bid==anchor:first=value
            if good:best=value if best is None else max(best,value)
            curve.append({"batch_id":bid,"model_id":mid,"previous_model":batch["decision"]["current_model"],
                "status":current["status"],"score":current.get("score"),"groups":current.get("groups"),
                "update_utility_delta":value-task.utility(previous["score"]) if good and previous["status"]=="completed" else None,
                "forgetting_from_anchor":first-value if first is not None and good else None,
                "drop_from_best":best-value if complete else None})
        results.append({"set_id":group["set_id"],"source":group["source"],"anchor_batch":anchor,
            "sample_count":len(group["samples"]),"status":"completed" if complete else "incomplete","curve":curve,
            "unit_isolation":"checked" if all(r.get("unit_id") for r in group["samples"]) else "sample_ids_only"})
    return {"schema_version":1,"identity":saved["identity"],"reference_digest":digest(specification),
        "run_fingerprint":state["fingerprint"],"run_snapshot_digest":digest(state),
        "evidence_kind":state["evidence_kind"],"objective":task.objective,
        "scope":"post_run_evaluation_only","sets":results,"cost":operation_costs(list(saved["operations"].values())),
        "failed_predictions":sum(op["status"]=="failed" for op in saved["operations"].values()),
        "prediction_operations":len(saved["operations"]),"model_training":False}


def _write_explanation(folder, report):
    """从同一评价回执生成阅读入口；原始预测、缺项和费用保留在JSON。"""
    source='预写预测的工程核对' if report['evidence_kind']=='interface_fixture' else '已保存模型的实际预测评价'
    lines=['# 固定旧场景上的能力变化','',source+'。每一组始终使用同一批参照输入和答案，评价结果独立于改法选择。','',
           '更新变化大于0表示相对上一个服务版本改善；起点下降大于0表示比指定起点变差，负值表示改善。','']
    def value(number):return '未取得' if number is None else f'{number:.6g}'
    for item in report['sets']:
        lines.extend([f"## {item['set_id']}",'',f"评价状态：{item['status']}。",'',
            '| 批次 | 服务模型 | 指标值 | 本次更新变化 | 相对起点下降 | 相对历史最佳下降 |',
            '| --- | --- | --- | --- | --- | --- |'])
        for row in item['curve']:
            lines.append(f"| {row['batch_id']} | {row['model_id']} | {value(row['score'])} | {value(row['update_utility_delta'])} | "
                         f"{value(row['forgetting_from_anchor'])} | {value(row['drop_from_best'])} |")
        lines.append('')
    lines.extend(['评价预测费用单列在reference-report.json的cost；模型维护费用保留在运行记录。',''])
    (folder/'历史参照说明.md').write_text('\n'.join(lines),encoding='utf-8')
