"""按当前允许的用途生成可复现的准确样本清单。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import random

from harness.core.contracts import ActionPlan, BatchContext, EngineConfig, Sample, TaskSpec, digest


def ingest(task: TaskSpec, batch: BatchContext, existing: dict) -> tuple[dict, dict]:
    """检查整批身份与修订后返回新快照；出错时不修改existing。"""
    if batch.task_id != task.task_id or batch.ordinal < 0 or not batch.batch_id:
        raise ValueError("任务或批次身份不匹配")
    updated = dict(existing)
    seen = set()
    counts = {"new": 0, "replayed": 0, "revised": 0}
    for sample in batch.samples:
        sample.validate(task, batch.ordinal)
        if sample.sample_id in seen:
            raise ValueError("同批出现重复样本身份")
        seen.add(sample.sample_id)
        old = existing.get(sample.sample_id)
        new = asdict(sample)
        if old is None:
            counts["new"] += 1
        elif sample.revision == old["revision"]:
            if digest(new) != digest(old):
                raise ValueError("样本内容改变但修订号没有增加")
            counts["replayed"] += 1
        else:
            if sample.revision < old["revision"] or sample.available_at < old["available_at"]:
                raise ValueError("样本修订或可见时刻倒退")
            # 已公开的探索反馈或历史训练记录不能转成新的采用证据。
            if sample.role == "adopt" and old["role"] != "adopt":
                raise ValueError("旧资料不能重新标记为新的采用资料")
            counts["revised"] += 1
        updated[sample.sample_id] = new
    return updated, counts


def fitting_pools(records: dict, ordinal: int) -> tuple[tuple[Sample, ...], tuple[Sample, ...]]:
    rows = tuple(Sample.parse(v) for _, v in sorted(records.items())
                 if v["role"] == "fit" and v["available_at"] <= ordinal)
    return (tuple(s for s in rows if s.available_at == ordinal),
            tuple(s for s in rows if s.available_at < ordinal))


def _preserve_classes(pool, chosen, rng, covered=()):
    """在已定容量内补回漏选类别，其余随机选择尽量保留。

    只替换仍有其他代表的类别；容量小于类别数时保留实际缺项，由回执说明。
    covered是本轮已选近期标签，历史补选不必重复为它们保留位置。
    """
    selected = list(chosen)
    counts = Counter([*covered, *(s.label for s in selected)])
    by_class = defaultdict(list)
    for row in pool:
        by_class[row.label].append(row)
    missing = sorted(set(by_class) - set(counts))
    rng.shuffle(missing)
    for label in missing:
        donors = [i for i, row in enumerate(selected) if counts[row.label] > 1]
        if not donors:
            break
        index = rng.choice(donors)
        counts[selected[index].label] -= 1
        selected[index] = rng.choice(by_class[label])
        counts[label] += 1
    return tuple(selected)


def select_rows(task_id: str, batch_id: str, action: ActionPlan,
                recent: tuple[Sample, ...], history: tuple[Sample, ...],
                config: EngineConfig, *, task: TaskSpec | None = None) -> tuple[Sample, ...]:
    """从已分好用途的池中选择准确训练清单，不执行标签推断。

    ratio表示历史条数/近期条数；coverage配方使用报告组，class_coverage使用真实类别。
    后者只补抽样漏类，不将各类改为等量；不查看评价标签。回归沿用原规则。
    本函数按行限量；独立单位划分由上游负责，完整轨迹训练需任务适配。
    """
    # 近期清单仅由批次决定，不随配方或调参次数改变。
    base_seed = int(digest([task_id, batch_id, config.seed])[:16], 16)
    rng = random.Random(base_seed)
    if config.fit_sampling == "class_coverage" and task is None:
        raise ValueError("类别覆盖抽样需要明确任务类型")
    cover = config.fit_sampling == "class_coverage" and task.task_type == "classification"
    recent = tuple(sorted(recent, key=lambda x: x.sample_id))
    recent_pool = recent
    if len(recent) > config.recent_cap:
        recent = tuple(rng.sample(recent, config.recent_cap))
    if cover:
        recent = _preserve_classes(recent_pool, recent, rng)
    if not recent or action.recipe == "recent":
        return recent
    count = min(int(len(recent) * action.ratio), config.history_cap, len(history))
    # 同一资料配方的组件对照共用样本；处理器/模型变化不改变抽样种子。
    rng = random.Random(base_seed + int(digest(f"{action.recipe}-{action.ratio:g}")[:12], 16))
    history = tuple(sorted(history, key=lambda x: x.sample_id))
    if action.recipe == "uniform":
        chosen = rng.sample(history, count)
    else:
        groups = defaultdict(list)
        for s in history:
            groups[s.group].append(s)
        keys = sorted(groups)
        rng.shuffle(keys)
        for values in groups.values():
            rng.shuffle(values)
        chosen = []
        while len(chosen) < count:
            for key in keys:
                if groups[key] and len(chosen) < count:
                    chosen.append(groups[key].pop())
    if cover:
        chosen = _preserve_classes(history, chosen, rng, (s.label for s in recent))
    return tuple(sorted((*recent, *chosen), key=lambda x: x.sample_id))


def selection_report(task, action, recent, history, selected, config):
    """保存实际可用池与入选资料的区别；报告组、标签和独立单位分别描述。"""
    eligible = (*recent, *(history if action.recipe != "recent" else ()))
    classes = sorted({s.label for s in eligible}) if task.task_type == "classification" else []
    chosen_classes = sorted({s.label for s in selected}) if task.task_type == "classification" else []
    recent_ids = {s.sample_id for s in recent}
    units, picked = Counter(s.unit_id for s in eligible if s.unit_id is not None), Counter(s.unit_id for s in selected if s.unit_id is not None)
    groups = {s.group for s in eligible}
    chosen_groups = {s.group for s in selected}
    missing_groups = sorted(groups-chosen_groups)
    return {"policy": config.fit_sampling, "unit": "row", "recent_available": len(recent),
            "history_available": len(history) if action.recipe != "recent" else 0,
            "recent_selected": sum(s.sample_id in recent_ids for s in selected),
            "history_selected": sum(s.sample_id not in recent_ids for s in selected),
            "recent_cap": config.recent_cap, "history_cap": config.history_cap,
            "observed_classes": classes, "selected_classes": chosen_classes,
            "missing_observed_classes": sorted(set(classes)-set(chosen_classes)),
            "observed_groups": len(groups), "selected_groups": len(chosen_groups),
            "missing_groups": missing_groups[:16], "omitted_missing_groups": max(0,len(missing_groups)-16),
            "known_units_selected": len(picked), "partial_units": sum(n<units[u] for u,n in picked.items()),
            "rows_without_unit": sum(s.unit_id is None for s in selected),
            "selected_versions": [f"{s.sample_id}@{s.revision}" for s in selected]}
