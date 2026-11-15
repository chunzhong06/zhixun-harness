"""把局部维护绑定到明确父版，保留未指定组件，并返回实际改动范围。"""
from copy import deepcopy
from hashlib import sha256
import json


def plan_identity(plan):
    payload = {key: plan[key] for key in ("components", "filters")}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_edit(raw, reference):
    """局部编辑只改指名字段；整套替换须显式声明，过期父版直接拒绝。"""
    if raw.get("parent_identity") != plan_identity(reference):
        raise ValueError("父版身份不符，不能把本次编辑应用到另一套模型")
    modes = [key for key in ("component_edits", "components") if key in raw]
    if len(modes) != 1:
        raise ValueError("须选择component_edits局部编辑或components显式整套替换")
    if modes == ["components"]:
        if raw.get("replace_all") is not True:
            raise ValueError("整套替换须明确replace_all:true；局部修改使用component_edits")
        components = deepcopy(raw["components"])
    else:
        components = deepcopy(reference["components"])
        edits = raw["component_edits"]
        if not isinstance(edits, list) or not edits:
            raise ValueError("局部编辑需要非空component_edits")
        visited = set()
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"index", "changes"}:
                raise ValueError("每项编辑需要index和changes")
            index, changes = edit["index"], edit["changes"]
            if type(index) is not int or not 0 <= index < len(components) or index in visited:
                raise ValueError("组件编号越界或重复")
            if not isinstance(changes, dict) or not changes or not set(changes) <= {"python_source", "weight", "feedback"}:
                raise ValueError("只允许修改源码、权重或反馈设置")
            visited.add(index)
            components[index].update(deepcopy(changes))
    plan = {"components": components, "filters": deepcopy(raw.get("filters", reference["filters"])),
            "hypothesis": raw["hypothesis"]}
    delta = []
    for index in range(max(len(reference["components"]), len(components))):
        before = reference["components"][index] if index < len(reference["components"]) else None
        after = components[index] if index < len(components) else None
        fields = [key for key in ("python_source", "weight", "feedback")
                  if before is None or after is None or before[key] != after[key]]
        if fields:
            delta.append({"index": index, "operation": "added" if before is None else "removed" if after is None else "updated",
                          "changed_fields": fields})
    return plan, {"parent_identity": plan_identity(reference), "candidate_identity": plan_identity(plan),
                  "component_changes": delta, "filters_changed": plan["filters"] != reference["filters"]}


def named_scores(values, windows, clock, *, prediction_object):
    """仅包装调用者明确给出的窗口，不自行读取或扩展确认范围。"""
    if len(values) != len(windows):
        raise ValueError("成绩与窗口数不同")
    return [{"window_id": f"exploration-{i+1}", "prediction_object": prediction_object,
             "start_hour": int(clock[mask[clock]][0]), "end_hour_inclusive": int(clock[mask[clock]][-1]),
             "scored_rows": int(mask[clock].sum()), "metric": "RMSE", "direction": "minimize", "value": float(value)}
            for i, (value, mask) in enumerate(zip(values, windows, strict=True))]
