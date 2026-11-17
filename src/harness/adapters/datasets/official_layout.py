"""官方公开回放的固定划分与包身份；平台输入另外适配。"""
from collections import defaultdict
from dataclasses import replace
from harness.core.contracts import digest
from harness.adapters.datasets.official_bearing import read_official_bearing
from harness.adapters.datasets.official_engine import read_official_engine
from harness.adapters.datasets.official_rul import ARCHIVE_SHA
from harness.learning.models.official_gas_cpu import PACKAGE_SHA
from harness.paths import PACKAGE_ROOT,workspace_root
ROOT=workspace_root()
PROTOCOL=PACKAGE_ROOT/'resources/official-rul-device-maintenance.json'
PACKAGES = {
    "bearing": ("cpu-b-rul-20261110-v1.zip", ARCHIVE_SHA["bearing"], read_official_bearing),
    "engine": ("cpu-c-guide-20261110-v1.zip", ARCHIVE_SHA["engine"], read_official_engine),
}
def partition(package, protocol):
    """按先定设备身份分配用途，划分不读取预测或寿命答案。训练时可取完整训练轨迹。"""
    by_unit = defaultdict(list)
    for row in package.train:
        by_unit[row.unit_id].append(row)
    fit = []
    for group in by_unit.values():
        ordered = sorted(group, key=lambda r: r.time_end)
        count = min(protocol["training_observations_per_device"], len(ordered))
        indices = [i * (len(ordered) - 1) // (count - 1) for i in range(count)] if count > 1 else [0]
        fit.extend(ordered[i] for i in indices)
    strata = defaultdict(set)
    for row in package.validation:
        stratum = row.unit_id.split(":")[0] if package.task.target_unit == "cycles" else "bearing"
        strata[stratum].add(row.unit_id)
    assignment = {}
    for stratum, units in strata.items():
        if len(units) < 3:
            raise ValueError(f"{stratum} 缺少分开探索、确认和服务所需的设备")
        for index, uid in enumerate(sorted(units, key=lambda u: digest([protocol["seed"], u]))):
            assignment[uid] = ("explore", "adopt", "serve")[index % 3]
    if set(assignment) & set(by_unit):
        raise ValueError("验证设备越入训练集")
    rows = {"fit": tuple(fit), "explore": [], "adopt": [], "serve": []}
    for row in package.validation:
        role = assignment[row.unit_id]
        if role == "serve" or (row.time_end - 1) % protocol["development_observation_stride"] == 0:
            rows[role].append(replace(row, role=role))
    return {k: tuple(v) for k, v in rows.items()}, assignment
