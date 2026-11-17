"""拟合资料的必要条件与独立单位；只检查清单，不调用训练器。"""
from collections import Counter


def independent_units(rows):
    """单位必须明确声明，报告分组和类别名称不自动取得独立性含义。"""
    if not rows or any(s.unit_id is None for s in rows):
        raise ValueError("独立设备/轨迹身份未完整声明")
    return sorted({s.unit_id for s in rows})


def training_condition(task, rows, action=None, *, required_classes=None):
    """检查当前开放预测器的拟合前提；覆盖要求由比较所用完整拟合池给出。

    逻辑回归和CatBoost分类需要至少两个观察类别；随机森林允许单类拟合。分档/CV另要求
    保留完整拟合池中已观察的类别，不从探索或采用答案补齐训练类别。
    """
    classes=sorted({s.label for s in rows}) if task.task_type=="classification" else []
    reason=None
    if action and action.pipeline:
        action.pipeline.validate_task(task)
    if not rows:
        reason="empty_fit"
    elif task.task_type=="classification":
        predictor=action.pipeline.predictor if action and action.pipeline else "linear"
        if len(classes)<(2 if predictor in {"linear", "catboost"} else 1):
            reason="insufficient_observed_classes"
        elif required_classes is not None and not set(required_classes)<=set(classes):
            reason="missing_fit_classes"
    if reason is None and action and action.pipeline and (
            action.pipeline.features=="nonconstant" or action.pipeline.predictor=="catboost"):
        if not any(len(set(values))>1 for values in zip(*(s.values for s in rows))):
            reason="no_nonconstant_features"
    if (reason is None and task.task_type=="regression" and action and action.pipeline
            and action.pipeline.predictor=="catboost" and len({s.label for s in rows})<2):
        reason="constant_regression_target"
    return {"ready":reason is None,"reason":reason,"rows":len(rows),"observed_classes":classes,
            "unit_count":len({s.unit_id for s in rows if s.unit_id is not None}),
            "missing_unit_rows":sum(s.unit_id is None for s in rows)}


def assign_group_folds(rows, folds, *, classification=False):
    """按完整单位分折；分类时贪心平衡类别数量，随后由调用者检查每折可训练性。

    使用GroupKFold/StratifiedGroupKFold的职责分工。分配为确定性启发式，
    无可用划分时交由调用者保留当前模型，不把单位拆开来满足类别数量。
    """
    units=independent_units(rows)
    by_unit={unit:[s for s in rows if s.unit_id==unit] for unit in units}
    labels=sorted({s.label for s in rows}) if classification else [None]
    counts={u:Counter(s.label if classification else None for s in group) for u,group in by_unit.items()}
    totals=Counter(s.label if classification else None for s in rows)
    ordered=sorted(units,key=lambda u:(-max(counts[u].values()),-len(by_unit[u]),u))
    allocations=[[] for _ in range(folds)]
    assigned=[Counter() for _ in range(folds)]
    for index,unit in enumerate(ordered):
        candidates=[index] if index<folds else range(folds)
        def imbalance(target):
            trial=[count+counts[unit] if i==target else count for i,count in enumerate(assigned)]
            loss=sum(sum((count[label]/totals[label]-1/folds)**2 for count in trial) for label in labels)
            return loss,len(allocations[target]),target
        target=min(candidates,key=imbalance)
        allocations[target].append(unit)
        assigned[target].update(counts[unit])
    return allocations
