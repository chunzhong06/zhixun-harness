"""发动机首次维护的等量选样对照，使用官方训练划分和已揭晓的探索观测。"""
from collections import defaultdict
from hashlib import sha256

from harness.contracts import ActionPlan, digest
from harness.models.official_rul_cpu import RulPilotTools


BALANCED = "device-balanced"
TARGETED = "cycle-window"
# 从首次公开探索记录冻结的运行周期区间，单位是已运行循环数。
FIRST, LAST = 81, 161


def fit_lists(fit):
    """每台八取七，固定首末。替换仅在同一设备和同一工况内部进行。"""
    devices = defaultdict(list)
    for row in fit:
        if row.role != "fit" or not row.unit_id or row.time_end is None:
            raise ValueError("发动机选样需要带设备和观测时刻的训练资料")
        devices[row.unit_id].append(row)
    if not devices or len({r.sample_id for r in fit}) != len(fit):
        raise ValueError("冻结的训练编号为空或有重复")
    lists = {BALANCED: [], TARGETED: []}
    for members in devices.values():
        ordered = sorted(members, key=lambda r: (r.time_end, r.sample_id))
        if len(ordered) != 8 or len({r.time_end for r in ordered}) != 8:
            raise ValueError("本次原始清单要求每台恰有八个不同的观测时刻")
        interior = ordered[1:-1]
        omitted = min(interior, key=lambda r: sha256(("17|" + r.sample_id).encode()).digest())
        replacement = omitted
        if FIRST <= omitted.time_end <= LAST:
            replacement = next((r for r in interior if r.time_end > LAST and r.group == omitted.group), omitted)
        lists[BALANCED].extend(r for r in ordered if r.sample_id != omitted.sample_id)
        lists[TARGETED].extend(r for r in ordered if r.sample_id != replacement.sample_id)
    return lists


def ids_sha(ids):
    return sha256("\n".join(sorted(ids)).encode()).hexdigest()


class EngineFitTools(RulPilotTools):
    """两张清单共用原版、训练器和探索集。构建和恢复均由同一个请求函数选定清单。"""

    def __init__(self, archive, package, fit, discovery, original, runner):
        super().__init__("engine", archive, package, fit, discovery, original, runner)
        self.choices = {
            name: RulPilotTools("engine", archive, package, rows, discovery, original, runner)
            for name, rows in fit_lists(fit).items()
        }
        self.list_hashes = {name: ids_sha(tool.fit) for name, tool in self.choices.items()}
        self.identity = digest([self.identity, self.list_hashes, "engine-fit-v1"])

    def select_fit_rows(self, name, eligible):
        if name not in self.choices:
            raise ValueError("未登记的发动机训练清单")
        registered = self.choices[name].fit
        rows = tuple(r for r in eligible if r.sample_id in registered)
        if len(rows) != len(registered) or {r.sample_id: r for r in rows} != registered:
            raise ValueError("本批可用训练资料与登记的清单不符")
        return rows

    def _fit_request(self, task, action, rows, parent):
        if action.fit_manifest not in self.choices:
            raise ValueError("发动机续训缺少登记的清单")
        name = action.fit_manifest
        request = self.choices[name]._fit_request(task, action, rows, parent)
        return {**request, "fit_manifest": name, "fit_ids_sha256": self.list_hashes[name]}

    def _diagnostic_inputs(self, batch_id):
        return {**super()._diagnostic_inputs(batch_id), "stage": batch_id}

    def fit_manifest_context(self, batch, baseline):
        # 只读取本批可用的探索预测和标签；确认设备和服务设备不参与选案。
        by_unit = defaultdict(list)
        if baseline:
            for row in batch.samples:
                if (row.role == "explore" and row.available_at == batch.ordinal and row.label is not None
                        and FIRST <= row.time_end <= LAST):
                    by_unit[row.unit_id].append(baseline["predictions"][row.sample_id] > row.label)
        eligible = [signs for signs in by_unit.values() if len(signs) >= 4]
        return {"names": list(self.choices), "window": [FIRST, LAST], "eligible_devices": len(eligible),
                "persistently_high": sum(4 * sum(signs) >= 3 * len(signs) for signs in eligible)}


class EngineFitPlanner:
    """固定策略总选均衡清单；按反馈选样时，以探索设备的偏差决定是否调换名额。"""

    def __init__(self, policy):
        if policy not in ("fixed", "observed"):
            raise ValueError("未知的发动机选样策略")
        self.policy = policy
        self.identity = "engine-fit-v1-" + policy

    def propose(self, context):
        if context["trials"] or 1 not in context.get("continuation_epochs", ()):
            return {"stop": True, "actions": []}
        fact = next(f for f in context["facts"]["evidence"] if f["kind"] == "performance")
        info = context["fit_manifests"]
        n, high = info["eligible_devices"], info["persistently_high"]
        use_window = self.policy == "observed" and n >= 2 and high >= 2 and 2 * high > n
        name = TARGETED if use_window else BALANCED
        reason = ("按固定策略每台均衡取七条。" if self.policy == "fixed" else
                  f"在已运行 {FIRST} 至 {LAST} 个循环的观测中，{n} 台可核对设备有 {high} 台持续高估。"
                  + ("把同工况的一部分训练名额移入该时段。" if use_window else "沿用每台均衡取七条的清单。"))
        action = ActionPlan("recent", epochs=1, fit_manifest=name, reason=reason, evidence_refs=(fact["id"],))
        return {"stop": False, "actions": [action.to_record()]}
