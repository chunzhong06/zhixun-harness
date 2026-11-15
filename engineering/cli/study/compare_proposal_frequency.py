"""重放同一提案者的首轮真实回答，独立走完后九轮固定更新，再按实际交接配对计分。"""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from math import fsum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from compare_public_selection import OUT as EXTERNAL, external_batches
from score_local_defaults import BASE, cost, metric, official, read
from verify_unseen_scores import remaining_cycles, remaining_minutes
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from harness.contracts import digest
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.storage import LocalStore, atomic_json

OUTPUT = BASE / "proposal-frequency" / "streams"
TASKS = {"gas": "gas", "bearing": "bearing", "engine": "engine", "pronostia": "bearing", "phm08": "engine"}


def reference_folders(cohort, arm):
    task = TASKS[cohort]
    if cohort == task:
        paper = "paper-gas-round-budget" if task == "gas" else "paper-" + task
        return BASE / "fixed" / task, BASE / paper / task
    parent = EXTERNAL / "short-window" if cohort == "pronostia" else EXTERNAL
    return parent / task / "fixed", parent / task / arm


def first_context(state):
    return next(event["context"] for event in state["events"] if event["key"] == "round-1:plan_context:0")


def comparable_context(context):
    """仅实测用时和峰值内存在两次执行间自然变化，保留资源余量、操作计数和所有任务观察。"""
    snapshot = deepcopy(context)
    resources, = [f for f in snapshot["facts"]["evidence"] if f["kind"] == "resources"]
    resources.pop("id")
    for period in ("current_batch", "all_visible_batches"):
        for name in ("measured_elapsed_seconds", "worker_cpu_seconds", "observed_peak_worker_rss_bytes"):
            resources["value"][period].pop(name)
    return snapshot


class FirstProposal:
    def __init__(self, fixed, source):
        self.fixed = fixed
        context = first_context(read(source / "state.json"))
        entry = read(source / "driver-exchanges.json")["exchanges"][0]
        if (entry["batch_id"] != "round-1" or entry["status"] != "completed"
                or entry["context_digest"] != digest(context)
                or entry["evidence_kind"] != "local_proposal_response"):
            raise ValueError("历史首轮回答或其发生时的观察无法验证")
        self.context_digest = digest(comparable_context(context))
        self.plan = entry["plan"]
        self.identity = digest(["first-local-reply-then-fixed-v1", fixed.identity, entry["request_id"], self.plan])

    def propose(self, context):
        # 气体固定策略首轮建立方向顺序，随后使用自己的已采用历史。
        fallback = self.fixed.propose(context)
        if context["batch_id"] != "round-1":
            return fallback
        # 运行中的元组和已持久化的 JSON 列表在同一规范编码下比较。
        if digest(comparable_context(context)) != self.context_digest:
            raise ValueError("首轮可见数据、预算或合法动作发生变化")
        return deepcopy(self.plan)


def execute(cohort, arm):
    task = TASKS[cohort]
    fixed, source = reference_folders(cohort, arm)
    output = OUTPUT / cohort / arm
    driver = {"mode": "rule"}
    with LocalStore(output / "study-lock").exclusive():
        harness, batches = assemble_public(task, ROOT / "data/official", output, "fixed", driver,
                                          select_training_rows=True)
        harness.planner = FirstProposal(harness.planner, source)
        if cohort != task:
            batches = external_batches(task, batches, match_window=(cohort == "pronostia"))
            harness.tools.values = {r.sample_id: r.values for b in batches for r in b.samples if r.role != "fit"}
        fixed_state = read(fixed / "state.json")
        if any(fixed_state["completed"][b.batch_id]["input_digest"] != digest(asdict(b)) for b in batches):
            raise ValueError("输入与已完成的强固定对照不同")
        report = run_prepared(task, output, "first-proposal", 10, driver, harness, batches)
        prior = read(source / "run.json")["rounds"][0]
        first = report["rounds"][0]
        if (first["planned_actions"] != prior["planned_actions"] or first["outcome"] != prior["outcome"]
                or read(output / first["submitted"])["payload"]["predictions"]
                != read(source / prior["submitted"])["payload"]["predictions"]):
            raise ValueError("首轮回答、采用结果或服务预测没有复现")
    summarize(cohort, arm)


def summarize(cohort, arm):
    fixed, source = reference_folders(cohort, arm)
    one = OUTPUT / cohort / arm
    exchanges = read(source / "driver-exchanges.json")["exchanges"]
    results, streams, controls = measure_streams(cohort, {"zero":fixed, "one":one, "ten":source},
                                                {"zero":[], "one":exchanges[:1], "ten":exchanges})
    for name, stream in streams.items():
        results[name]["different_from_zero"] = sum(value != streams["zero"][key] for key, value in stream.items())
    atomic_json(one / "scores.json", {"cohort": cohort, "proposer": arm, "input_and_budget": controls,
                                     "new_model_calls_in_replay": 0, "arms": results})
    print(cohort, arm, {name: round(record["score"], 5) for name, record in results.items()})


def measure_streams(cohort, paths, replies):
    """用相同的公开答案复算每个真实交接流，核对父权重、反馈时序、服务输出与训练费用。"""
    task = TASKS[cohort]
    package, checksum, _ = official(task)
    if cohort == "phm08":
        answers = remaining_cycles()
    elif cohort == "pronostia":
        answers = remaining_minutes("full-test-manifest.json")
    else:
        answers = {s.sample_id: s.label for s in package.validation}
    results, streams = {}, {}
    controls = None
    for name, path in paths.items():
        state, report = read(path / "state.json"), read(path / "run.json")
        control = digest([state["config"], state["samples"],
                          [state["completed"][r["batch"]]["input_digest"] for r in report["rounds"]]])
        if controls is None:
            controls = control
        if controls != control or not read(path / report["swanlab"])["verified"] or len(report["rounds"]) != 10:
            raise ValueError("三路数据、时序、预算或交接完整性不同")
        rows = state["samples"]
        store = ModelArtifacts(path / "models")
        start, = [m for m in state["models"].values() if m["parent_id"] is None]
        expected_start = package.checkpoint_sha256.removeprefix("sha256:") if task == "gas" else package.checkpoint_sha
        if store.read(start["artifact"])[0]["sha256"] != expected_start:
            raise ValueError("初始模型不一致")
        stream, round_scores = {}, []
        for ordinal, r in enumerate(report["rounds"]):
            payload = read(path / r["submitted"])["payload"]
            prediction = payload["predictions"]
            expected = {i for i, row in rows.items() if row["role"] == "serve" and row["available_at"] == ordinal}
            if (set(prediction) != expected or stream.keys() & expected or payload["model_id"] != r["selected"]
                    or any(rows[i]["label"] is not None for i in expected) or len(r["trials"]) > 1):
                raise ValueError("实际服务分片、采用版或单轮预算不符")
            worker = state["operations"][f"{r['batch']}:serve:{r['selected']}"]["execution"]["request_id"]
            result = read(path / "jobs" / worker / "state.json")["value"]
            if prediction != (result["predictions"] if "gas_repairs" in result else result):
                raise ValueError("交接内容与工作进程发出的预测不符")
            store.read(r["model_artifact"])
            stream.update(prediction)
            round_scores.append(metric(task, answers, prediction))
        streams[name] = stream
        groups = defaultdict(dict)
        if task != "gas":
            for key, value in stream.items():
                groups[rows[key]["unit_id"]][key] = value
        devices = {unit: metric(task, answers, p) for unit, p in groups.items()}
        used = replies[name]
        fits = {k for k, row in rows.items() if row["role"] == "fit"}
        results[name] = {"score": metric(task, answers, stream), "round_scores": round_scores,
                         "device_scores": devices, "worst_round": (min if task == "gas" else max)(round_scores),
                         "worst_device": max(devices.values(), default=None), "rows": len(stream),
                         "model_replies_used": len(used), "source_reply_seconds": fsum(v["elapsed_seconds"] for v in used),
                         "issued_sha256": digest(stream), **cost(path, checksum, fits)}
    return results, streams, controls


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=TASKS)
    parser.add_argument("arm", choices=("mature", "simple"))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.cohort == TASKS[args.cohort] and args.arm != "mature":
        parser.error("三个公开包的同协议十轮参考只有论文适配")
    (summarize if args.score_only else execute)(args.cohort, args.arm)
