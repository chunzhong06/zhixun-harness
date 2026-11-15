"""把已完成工程运行送入真正的SwanLab离线SDK，并重新打开文件逐条核查。"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.abc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.runtime.storage import atomic_json


def history_for(job):
    training = json.loads((job / "training.json").read_bytes())
    epochs = [json.loads(line) for line in (job / "history.jsonl").read_bytes().splitlines()]
    if len(epochs) != training["final_epochs"] - training["initial_epochs"]:
        raise ValueError("随包训练历史缺少轮次")
    return {"parent": training["parent_checkpoint_sha256"], "candidate": training["checkpoint_sha256"], "epochs": epochs}


def child(run_folder: Path, output: Path, *, frozen_review=False) -> None:
    """子进程只读取工程事件；在导入SDK前关闭网络和模型库入口。"""
    counts = {"network_attempts": 0, "model_import_attempts": 0}

    def deny_network(*args, **kwargs):
        counts["network_attempts"] += 1
        raise RuntimeError("离线记录核查不允许网络连接")

    class ModelGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"sklearn", "torch", "tensorflow", "transformers", "openai", "pytrio"}:
                counts["model_import_attempts"] += 1
                raise RuntimeError("离线记录核查不导入模型库")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    sys.meta_path.insert(0, ModelGuard())
    from harness.integrations.tracking import SwanLabRecorder
    from harness.integrations.offline_tracking import verify_offline_records
    state = json.loads((run_folder / "state.json").read_text(encoding="utf-8"))
    expected_kind = "real_model" if state["model_experiment"] else "interface_fixture"
    if state["evidence_kind"] != expected_kind:
        raise ValueError("运行的实验标记与证据来源冲突")
    jobs = run_folder / "jobs"
    series_scope = None
    if run_folder.name == "workflow":
        if frozen_review:
            jobs = run_folder.parent / "jobs"
        root = run_folder.parent.parent
        definition = root / "protocol.json"
        if state["model_experiment"] and definition.is_file():
            scope = json.loads(definition.read_bytes())
            if scope.get("definition", {}).get("name") == "rul-disjoint-device-stream-v1":
                if run_folder.parent.name not in scope["definition"]["arms"]:
                    raise ValueError("寿命续训记录未对应冻结的比较组")
                jobs, series_scope = root / "jobs", scope
    histories = {}
    for key, op in state["operations"].items():
        request_id = op.get("execution", {}).get("request_id")
        if not request_id or not op["name"].startswith("build:") or op["status"] != "completed":
            continue
        job = jobs / request_id
        if (job / "training.json").is_file():
            histories[key] = history_for(job)
        elif series_scope is not None:
            raise ValueError("连续寿命维护缺少该候选的真实续训回执")
    audit_path = run_folder / "对话接口记录.json"
    # 官方气体将运行状态与驱动并列存放；寿命和逐批演练在独立 workflow 子目录保存状态。
    driver_root = run_folder.parent if run_folder.name == "workflow" else run_folder
    official_audit = driver_root / "driver-exchanges.json"
    if official_audit.is_file():
        if audit_path.is_file():
            raise ValueError("运行同时含两套驱动记录，请核对其归属")
        audit_path = official_audit
    audit_data = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit = audit_data.get("exchanges", [])
    calls = [c for op in state["operations"].values() for c in op.get("service_calls", [])]
    fields = ("request_id", "status", "usage", "evidence_kind")
    def signatures(receipts):
        return Counter(json.dumps([r[k] for k in fields], sort_keys=True) for r in receipts)
    if signatures(calls) != signatures(audit):
        raise ValueError("工作流收取的驱动回执与导出的审计记录不匹配")
    task = state["task"]["task_id"]
    project = {"official-engine-C": "official-engine-cpu", "official-bearing-B": "official-bearing-cpu"}.get(task, "official-gas-cpu")
    comparison = None
    extra_count = 0
    if frozen_review and series_scope is not None:
        from harness.contracts import digest
        summary = json.loads((jobs.parent / "comparison.json").read_bytes())
        arm = run_folder.parent.name
        result = summary["arms"][arm]
        if (summary["protocol_digest"] != digest(series_scope) or result["workflow"] != state["fingerprint"]
                or result["candidate_epochs"] != len(histories)
                or any(r["decision"] != state["completed"][r["batch"]]["decision"] for r in result["rounds"])):
            raise ValueError("连续寿命对照成绩、续训记录或逐轮决定与本次运行不符")
        comparison = {"metrics": {"service/selected_rmse": result["overall"]["rmse"],
                                   "service/worst_round_rmse": result["worst_round_rmse"],
                                   "comparison/logical_worker_cpu_seconds": result["logical_worker_cpu_seconds"]},
                      "detail": {"arm": arm, "scope": summary["protocol_digest"], "rounds": result["rounds"]}}
        extra_count = 1
    elif frozen_review and state["model_experiment"] and task in {"official-engine-C", "official-bearing-B"}:
        result = json.loads((jobs.parent / "result.json").read_bytes())
        batch = state["completed"]["maintenance-1"]
        if result["decision"] != batch["decision"] or result["plan"] != batch["trials"]:
            raise ValueError("事后比较属于不同的维护决定")
        report = result["service_comparison"]
        history = history_for(jobs / result["candidate"]["key"])
        comparison = {"metrics": {"service/original_rmse": report["original"]["rmse"],
                                   "service/candidate_rmse": report["candidate"]["rmse"],
                                   "service/selected_rmse": result["harness_service"]["rmse"],
                                   "comparison/all_worker_cpu_seconds": result["shared_jobs_cpu_seconds"]},
                      "detail": {"policies": result["policies"], "service_rows": result["service_rows"],
                                 "logical_fit_counts": result["logical_fit_counts"], "parent_sha256": history["parent"],
                                 "candidate_sha256": history["candidate"], "failed_jobs": result.get("prior_failed_jobs", [])}}
        if not batch["trials"]:
            comparison["control_history"] = history
        elif history not in histories.values():
            raise ValueError("实际候选的续训回执未对应到工作流")
        extra_count = 1 + len(comparison.get("control_history", {}).get("epochs", []))
    recorder = SwanLabRecorder(output, enable_sdk=True, project=project if state["model_experiment"] else "harness-engineering")
    options = {"training_histories": histories, "comparison": comparison}
    receipt = recorder.sync(state, audit, **options)
    if receipt["cursor"] != len(state["events"]) + extra_count + sum(len(item["epochs"]) for item in histories.values()):
        raise RuntimeError(f"离线SDK没有完整结束：{receipt['segments'][-1]}")
    report = verify_offline_records(output)
    # 重复导出验证记录补交的幂等性，不产生第二个相同分段。
    if recorder.sync(state, audit, **options) != receipt:
        raise AssertionError("重复导出改变了已完成分段")
    report.update(counts, repeat_sync_unchanged=True, linked_driver_calls=len(calls))
    atomic_json(output / "verification.json", report)
    if any(counts.values()):
        raise RuntimeError("记录核查出现预期外网络或模型访问")
    from harness.contracts import digest
    from harness.runtime.storage import LocalStore
    atomic_json((output if frozen_review else run_folder)/"recording-status.json",{"mode":"offline","offline_verified":True,
        "cloud_verified":False,"run_fingerprint":state["fingerprint"],"event_digest":digest(state["events"]),
        "verification_path":str((output/"verification.json").resolve()),"sdk_version":report["sdk_version"]})
    if not frozen_review:
        LocalStore(run_folder).export(state)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-review", action="store_true", help="只读已冻结的运行，在输出目录生成独立对照记录")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.child:
        atomic_json(output / "verification.json", {"verified": False, "status": "checking"})
        try:
            child(run, output, frozen_review=args.frozen_review)
        except BaseException as exc:
            # SDK也管理进程异常输出，另存核查错误以免被其终端处理隐藏。
            import traceback
            atomic_json(output / "verification-failure.json", {
                "error": type(exc).__name__, "traceback": traceback.format_exc()})
            atomic_json(output / "verification.json", {"verified": False, "status": "failed", "error": type(exc).__name__})
            raise
        (output / "verification-failure.json").unlink(missing_ok=True)
        return
    # SDK读取配置时使用此任务独有目录；子进程不继承账户密钥和用户配置。
    with tempfile.TemporaryDirectory(prefix="sdk-check-", dir=output) as scratch:
        env = {k: v for k, v in os.environ.items()
               if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "PATHEXT"}}
        env.update(TEMP=scratch, TMP=scratch, SWANLAB_ROOT=scratch, SWANLAB_CONFIG_DIR=scratch,
                   SWANLAB_MODE="offline", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
            "--run", str(run), "--output", str(output), "--child", *(["--frozen-review"] if args.frozen_review else [])], env=env, cwd=scratch,
            timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    print(f"SwanLab离线文件核验完成：{report['events']}条事件，{report['scalar_values']}个指标值；网络访问=0。")


if __name__ == "__main__":
    main()
