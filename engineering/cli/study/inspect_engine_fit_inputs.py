"""用已保存的原版预测重走两种选样策略，在官方训练器第一次更新前读取输入。"""
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.run_rul_maintenance import engine_windows, run
from harness.contracts import ActionPlan, Sample, TaskSpec, digest
from harness.models.engine_manifests import BALANCED, TARGETED, FIRST, LAST, ids_sha
from harness.models.official_rul_cpu import execute_official_job
from harness.runtime.engine import UnconfirmedOperation
from harness.runtime.execution import JobUnconfirmed, LocalJobRunner
from harness.runtime.storage import atomic_json

AUDIT = ROOT / "results/inspections/cross-task/维护提案数据核查.json"
OUTPUT = ROOT / "results/inspections/cross-task/发动机训练输入核验.json"


class InputsRead(JobUnconfirmed):
    """让只读工作流在候选训练入口返回；临时工作区随后整体释放。"""


def coverage_by_family(batch, tools):
    """检视训练名额是否落在发现问题的发动机子集。每台设备单独计票，只读探索标签。"""
    groups = defaultdict(list)
    for row in batch.samples:
        if row.role == "explore" and row.available_at == batch.ordinal and FIRST <= row.time_end <= LAST:
            groups[row.unit_id].append(tools.original[row.sample_id] - row.label)
    fixed = tools.choices[BALANCED].fit
    targeted = tools.choices[TARGETED].fit
    swaps = Counter(targeted[sid].unit_id.split(":")[0] for sid in targeted.keys() - fixed.keys())
    summaries = {}
    for family in sorted({u.split(":")[0] for u in groups} | set(swaps)):
        devices = [e for u, e in groups.items() if u.split(":")[0] == family and len(e) >= 4]
        high = sum(4 * sum(v > 0 for v in errors) >= 3 * len(errors) for errors in devices)
        summaries[family] = {"eligible_devices": len(devices), "persistently_high": high,
                             "extra_target_rows": swaps[family]}
    return summaries


class InputReader(LocalJobRunner):
    def run(self, request, rows, backend_config):
        if request["kind"] != "continue" or rows:
            raise ValueError("只读核验只接受一次正式续训请求")
        self.request = request
        self.folder.mkdir(parents=True, exist_ok=True)
        self.loaded = execute_official_job(self.folder, {"request": request, "backend": backend_config},
                                           self.artifacts, inspect_resume=True)
        raise InputsRead()

    def recover(self, request):
        if request != self.request:
            raise ValueError("恢复时重建的请求与首次执行不同")
        return None

    def inspect(self, main, batch, tools, fit):
        try:
            main.run((batch,))
        except UnconfirmedOperation as exc:
            if not isinstance(exc.__cause__, InputsRead):
                raise
        else:
            failures = [(op["name"], op.get("error")) for op in main.state["operations"].values()
                        if op["status"] == "failed"]
            raise ValueError(f"没有读到续训输入，工作流报告 {failures}")
        request = self.request
        operation = next(op for op in main.state["operations"].values()
                         if op["status"] == "started" and op["name"].startswith("build:"))
        action = ActionPlan.parse(operation["effective_action"])
        selected = tools.select_fit_rows(action.fit_manifest, fit)
        ids = set(request["fit_ids"])
        if ids != {r.sample_id for r in selected} or ids_sha(ids) != request["fit_ids_sha256"]:
            raise ValueError("执行样本与计划器选择的清单不一致")
        tools.recover_build(main.task, batch.batch_id, action, selected, parent=tools.initial())
        # 独立按编号筛选官方原始字节，与工作者在临时目录中真正写出的训练输入核对。
        raw = sha256()
        with ZipFile(tools.worker["archive"]) as archive:
            cfg = json.loads(archive.read("baselines/configs/C.json"))
            for line in archive.open("public/train.jsonl"):
                if json.loads(line)["id"] in ids:
                    raw.update(line)
        if raw.hexdigest() != self.loaded["train_sha256"]:
            raise ValueError("官方训练器实际读取的行与清单不符")
        result = {"action": action.to_record(), "request_digest": digest(request),
                "ids_sha256": ids_sha(ids), "rows": len(ids), "batch_size": cfg["batch_size"],
                "loaded": self.loaded, "recovery_request_matches": True}
        if action.fit_manifest == TARGETED:
            result["exploration_and_allocation_by_family"] = coverage_by_family(batch, tools)
        return result


def inspect_pair():
    audit = json.loads(AUDIT.read_bytes())
    previous = audit["engine_candidate_pair"]
    with TemporaryDirectory(dir=ROOT / "scratch", prefix="engine-fit-inputs-") as tmp:
        arms = {policy: run("engine", Path(tmp) / policy, fit_policy=policy, input_reader=InputReader)
                for policy in ("fixed", "observed")}
    fixed, observed = arms.values()
    if (fixed["ids_sha256"] != previous["fixed_digest_sorted_lf"]
            or observed["ids_sha256"] != previous["alternative_digest_sorted_lf"]
            or fixed["rows"] != observed["rows"] or fixed["batch_size"] != observed["batch_size"]
            or fixed["request_digest"] == observed["request_digest"]):
        raise ValueError("两种选法未复现原先冻结的等量对照")
    shared = {k: v for k, v in fixed["loaded"].items() if k != "train_sha256"}
    if any(observed["loaded"][k] != v for k, v in shared.items()):
        raise ValueError("两组的父模型、优化器或探索输入不一致")
    # 等两份新请求已经生成，再用旧候选的确认预测核验分段计分，结果不返回计划器。
    record = audit["inputs"]["engine_state"]
    raw_state = (ROOT / record["path"]).read_bytes()
    if sha256(raw_state).hexdigest() != record["sha256"]:
        raise ValueError("旧维护回执已变化")
    state = json.loads(raw_state)
    confirm = tuple(Sample.parse(r) for r in state["samples"].values() if r["role"] == "adopt")
    ops = state["operations"]
    prefix = "maintenance-1:adopt:maintenance-1/"
    old, candidate = (ops[prefix + key]["value"] for key in ("initialize", "recent-0-epochs1"))
    if old["predictions"].keys() != candidate["predictions"].keys() or set(old["predictions"]) != {r.sample_id for r in confirm}:
        raise ValueError("旧版本的确认预测不在同一组设备记录上")
    windows = engine_windows(TaskSpec.parse(state["task"]), confirm, old["predictions"], candidate["predictions"])
    historical = {k: {arm: v[arm] for arm in ("original", "candidate")} for k, v in windows.items()}
    count = fixed["rows"]
    size = fixed["batch_size"]
    result = {"scope": "official_worker_inputs_before_training", "training_runs": 0,
              "audit_sha256": sha256(AUDIT.read_bytes()).hexdigest(), "rows_per_arm": count,
              "planned_updates_per_arm": (count + size - 1) // size,
              "paired_replacements": len(previous["paired_replacements"]), "common": shared,
              "historical_3968_row_candidate_confirmation": historical,
              "arms": {name: {**{k: v for k, v in arm.items() if k != "loaded"},
                              "training_file_sha256": arm["loaded"]["train_sha256"]} for name, arm in arms.items()}}
    atomic_json(OUTPUT, result)
    return OUTPUT


if __name__ == "__main__":
    print(inspect_pair())
