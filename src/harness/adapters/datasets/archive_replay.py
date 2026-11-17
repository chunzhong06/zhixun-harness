"""将已完成轨迹档案按固定方案分次发放；原生设备时间与人工批次分别保留。"""
from collections import defaultdict
from dataclasses import replace

from harness.core.contracts import BatchContext, digest


def replay_archives(task, batches, protocol, *, rounds=3):
    """从设备已隔离的静态开发输入构造受控多轮维护，不模拟未知失效时间。

    fit/explore/adopt来自开始前已完成的历史轨迹，标签此时已合法可知；服务
    前缀的答案始终留在评价侧。每台设备内部按原时间有序发放，单前缀测试
    设备按固定身份分批到达。发放次序是研究协议，不表示真实生产漂移。
    """
    if task.task_type != "regression" or type(rounds) is not int or rounds < 2:
        raise ValueError("档案重放需要数值寿命任务及至少两轮")
    if len(batches) != 2 or any(s.role=="serve" for s in batches[0].samples) or any(s.role!="serve" for s in batches[1].samples):
        raise ValueError("档案重放入口需要开发批与独立服务批")
    units, roles = defaultdict(list), {}
    for batch in batches:
        for row in batch.samples:
            if not row.unit_id or row.time_end is None:
                raise ValueError("档案重放需要明确设备和已见前缀时间")
            if row.unit_id in roles and roles[row.unit_id] != row.role:
                raise ValueError("同一设备跨越拟合、探索、采用或服务用途")
            roles[row.unit_id] = row.role
            units[row.unit_id].append(row)
    assigned = [[] for _ in range(rounds)]
    for unit, rows in sorted(units.items()):
        rows = sorted(rows,key=lambda r:(r.time_end,r.sample_id))
        if len({r.sample_id for r in rows}) != len(rows):
            raise ValueError("档案出现重复样本")
        for index,row in enumerate(rows):
            slot = int(digest(unit)[:12],16)%rounds if len(rows)==1 else index*rounds//len(rows)
            assigned[slot].append(replace(row,available_at=slot+1))
    required = {"fit", "explore", "adopt"}
    if any(not required <= {s.role for s in rows} for rows in assigned):
        raise ValueError("当前档案量不能支持各轮独立用途，减少轮数或增加设备")
    output = tuple(BatchContext(task.task_id,f"archive-release-{i+1}",i+1,tuple(sorted(rows,key=lambda r:r.sample_id)))
                   for i,rows in enumerate(assigned))
    return task, output, {**protocol, "protocol": protocol["protocol"]+"-archive-replay-v1",
        "replay": {"rounds":rounds, "release_time":"synthetic_order_not_production_time",
            "native_time":"sample.time_start/time_end within declared unit",
            "development_labels":"completed_source_trajectories_known_before_replay",
            "service_labels":"evaluation_only",
            "unit_roles":"fixed_disjoint_roles; repeated windows within units are correlated",
            "purpose":"controlled_multiple_updates_and_heldout_service_evaluation"},
        # 旧静态答案的发放序号由导出器按新服务事件重新绑定。
        "answers":[{k:v for k,v in row.items() if k not in {"prediction_ordinal","batch_id","prediction_revision","reveal_ordinal"}}
                   for row in protocol["answers"]]}
