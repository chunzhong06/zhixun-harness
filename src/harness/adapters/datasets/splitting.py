"""只在已释放资料内固定用途与折；设备身份、时间区间由任务适配器声明。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from harness.core.contracts import digest
from harness.adapters.datasets.data_conditions import independent_units, assign_group_folds, training_condition


@dataclass(frozen=True)
class SplitPolicy:
    mode: str = "group"
    explore_fraction: float = .2
    adopt_fraction: float = .2
    seed: int = 17
    gap: float = 0.

    def __post_init__(self):
        if self.mode not in {"group", "time"}:
            raise ValueError("划分方式须为group或time")
        if any(not math.isfinite(x) or not 0 < x < 1 for x in (self.explore_fraction,self.adopt_fraction)) or self.explore_fraction + self.adopt_fraction >= 1:
            raise ValueError("拟合、探索及采用均须保留正份额")
        if not math.isfinite(self.gap) or self.gap < 0:
            raise ValueError("时间间隔须为非负数")


def _interval(sample, metadata):
    declared=({"start":sample.time_start,"end":sample.time_end,"unit":sample.unit_id}
              if sample.time_start is not None else None)
    item = metadata.get(sample.sample_id,declared)
    if declared and item!=declared or item and sample.unit_id is not None and item.get("unit")!=sample.unit_id:
        raise ValueError("时间元数据与样本声明不一致")
    if item is None or set(item) != {"start", "end", "unit"}:
        raise ValueError("时间划分需要每个样本的独立单位与观察区间")
    if not item["unit"] or any(type(item[k]) not in {int,float} or not math.isfinite(item[k]) for k in ("start","end")) or item["end"] < item["start"]:
        raise ValueError("样本时间区间无效")
    return item


def split_released(rows, ordinal, policy=SplitPolicy(), *, metadata=None, task=None):
    """输入是尚未分用途的fit资料；原有探索/采用或服务资料不在此重分。

    group以完整独立单位划分，time按已观察区间排序并清除跨边界重叠。
    样本不足返回insufficient及原资料，调用方保持版本或积累新资料。
    清单只含身份、用途和排除原因，便于恢复核对。
    """
    if any(s.role != "fit" or s.label is None or s.available_at > ordinal for s in rows):
        raise ValueError("只允许划分已释放、尚未分用途的带标签资料")
    if len({s.sample_id for s in rows}) != len(rows):
        raise ValueError("划分输入包含重复样本")
    metadata = metadata or {}
    if policy.mode == "group":
        try:
            units = sorted(independent_units(rows), key=lambda x: digest([policy.seed,x]))
        except ValueError as exc:
            return tuple(rows),{"status":"insufficient","reason":str(exc),"assignments":{}}
        key = lambda s: s.unit_id
    else:
        # 同起点资料不跨用途；重叠窗口在划分后从较早用途清除。
        units = sorted({_interval(s,metadata)["start"] for s in rows})
        key = lambda s: _interval(s,metadata)["start"]
    if len(units) < 3:
        return tuple(rows), {"status":"insufficient","reason":"需要至少三个独立单位或时间点","assignments":{}}
    n_adopt = max(1,int(len(units)*policy.adopt_fraction))
    n_explore = max(1,int(len(units)*policy.explore_fraction))
    labels = (["fit"]*(len(units)-n_adopt-n_explore) + ["explore"]*n_explore + ["adopt"]*n_adopt)
    role_for = dict(zip(units,labels))
    assigned = [replace(s,role=role_for[key(s)]) for s in rows]
    excluded = []
    if policy.mode == "time":
        ranks = {"fit":0,"explore":1,"adopt":2}
        kept = []
        for sample in assigned:
            own = _interval(sample,metadata)
            later = [_interval(other,metadata) for other in assigned if ranks[other.role] > ranks[sample.role]
                     and _interval(other,metadata)["unit"] == own["unit"]]
            if later and own["end"] + policy.gap >= min(v["start"] for v in later):
                excluded.append(sample.sample_id)
            else:
                kept.append(sample)
        assigned = kept
    status = "ready" if {s.role for s in assigned} == {"fit","explore","adopt"} else "insufficient"
    condition=training_condition(task,tuple(s for s in assigned if s.role=="fit"),
        required_classes={s.label for s in rows} if task.task_type=="classification" else None) if task else None
    if condition and not condition["ready"]:
        status="insufficient"
    manifest = {"status":status,"policy":asdict(policy),"ordinal":ordinal,
                "source_digest":digest([asdict(s) for s in rows]),
                "assignments":{f"{s.sample_id}@{s.revision}":s.role for s in assigned},
                "excluded_overlap":excluded,"fit_condition":condition}
    manifest["identity"] = digest(manifest)
    return tuple(assigned) if status == "ready" else tuple(rows), manifest


def validation_folds(rows, *, mode="group", folds=3, metadata=None, gap=0., task=None, action=None):
    """小样本配方比较只在fit用途内进行；每个验证单位只出现一次。

    time使用逐步扩大的历史前缀，排除同单位的窗口重叠。采用资料由调用方
    单独保留；action传入本次已解析配方，以该预测器检查折内可训练条件。
    这里返回明确样本清单，不访问采用标签或自动训练。
    """
    if folds < 2 or any(s.role != "fit" for s in rows):
        raise ValueError("交叉验证需要至少两折且仅接受fit资料")
    metadata = metadata or {}
    units = independent_units(rows) if mode == "group" else sorted({_interval(s,metadata)["start"] for s in rows})
    if mode not in {"group","time"} or len(units) < folds + (mode == "time"):
        raise ValueError("独立单位不足或折方式无效")
    chunks = [units[i*len(units)//(folds+(mode=="time")):(i+1)*len(units)//(folds+(mode=="time"))]
              for i in range(folds+(mode=="time"))]
    if mode=="group":
        chunks=assign_group_folds(rows,folds,classification=bool(task and task.task_type=="classification"))
    result = []
    key = (lambda s:s.unit_id) if mode == "group" else (lambda s:_interval(s,metadata)["start"])
    for index in range(folds):
        valid_units = set(chunks[index if mode == "group" else index+1])
        train_units = set(units)-valid_units if mode == "group" else set(x for chunk in chunks[:index+1] for x in chunk)
        valid = tuple(s for s in rows if key(s) in valid_units)
        train = tuple(s for s in rows if key(s) in train_units)
        if mode == "time":
            train = tuple(s for s in train if all(_interval(s,metadata)["unit"] != _interval(v,metadata)["unit"]
                or _interval(s,metadata)["end"] + gap < _interval(v,metadata)["start"] for v in valid))
        if not train or not valid:
            raise ValueError("清除重叠后无法形成完整训练/验证折")
        if task:
            condition=training_condition(task,train,action,required_classes={s.label for s in rows} if task.task_type=="classification" else None)
            if not condition["ready"]:
                raise ValueError("该独立划分无法形成可训练折："+condition["reason"])
        result.append((train,valid))
    return result
