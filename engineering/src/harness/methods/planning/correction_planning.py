"""将冻结模型及错误区域转成当前可执行的修正目录，不读取采用资料。"""
from copy import deepcopy

from harness.methods.diagnosis.error_regions import frozen_regions
from harness.models.corrections import inside


def regions_for(state, batch_id):
    records = {r["id"]: r for r in frozen_regions(state, batch_id)}
    for check in state.get("inspections", {}).get(batch_id, []):
        for region in check.get("regions", []):
            records[region["id"]] = region
    return records


def correction_catalog(engine, batch, recent, history, trials):
    """只有本轮可见、已完整构建的一层基础模型可被引用；缺少新fit资料时不提案。"""
    if not engine.config.local_corrections or not getattr(engine.tools, "capabilities", {}).get("local_corrections"):
        return {"enabled": False, "bases": [], "regions": []}
    regions = regions_for(engine.state, batch.batch_id)
    identifiers = [engine.state["current"]] + [t.get("model_id") for t in trials
        if t.get("status") == "completed" and t.get("final_resource", True) and t.get("resource_fraction", 1.) == 1.]
    bases = []
    for mid in dict.fromkeys(identifiers):
        model = engine.state["models"].get(mid)
        if not model or model.get("correction"):
            continue
        used = set(model["training_ids"])
        # 当前自动目录使用recent配方，其适用条件按实际近期池检查；历史行不能虚增覆盖。
        fresh = {s.sample_id: s for s in recent if s.sample_id not in used}
        if len(fresh) < 2:
            continue
        supported = [rid for rid, r in regions.items() if sum(inside(s.values, r["predicates"]) for s in fresh.values()) >= 2]
        bases.append({"model_id": mid, "fresh_fit_rows": len(fresh), "region_ids": supported})
    return {"enabled": True, "bases": bases,
            "regions": [{"id": rid, "predicates": deepcopy(r["predicates"])} for rid, r in regions.items()]}


def resolve_base(engine, batch, action):
    """执行时核对冻结来源；完整模型包含基础对象，运行不形成外部权重依赖链。"""
    spec = action.correction
    if not spec:
        return None
    if not engine.config.local_corrections:
        raise ValueError("局部修正未开放")
    spec.validate_task(engine.task)
    model = engine._bundle(spec.base_model_id)
    if model.correction:
        raise ValueError("当前只开放一层修正")
    region = regions_for(engine.state, batch.batch_id).get(spec.region_id) if spec.region_id else None
    if spec.region_id and region is None:
        raise ValueError("修正区域尚未冻结")
    from dataclasses import asdict
    return {"bundle": asdict(model), "predicates": deepcopy(region["predicates"]) if region else []}
