"""单进程持久记录：状态原子替换，已完成外部步骤可复用。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import tempfile
import time


def atomic_json(path: Path, value: object) -> None:
    """同目录写临时文件后替换单个快照；不同时提交外部模型服务的副作用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # Windows读句柄短暂占用目标时允许有限等待，只重试同一快照的替换。
        # 不重新执行训练或重新序列化；持续权限错误仍向上传递并清理临时文件。
        deadline = time.monotonic() + .5
        while True:
            try:
                os.replace(temp, path)
                break
            except PermissionError as exc:
                if getattr(exc,"winerror",None) not in {5,32,33} or time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class LocalStore:
    """state.json是恢复依据；汇总和流程说明可从它重新导出。"""
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "state.json"

    def load(self) -> dict | None:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def save(self, state: dict) -> None:
        atomic_json(self.path, state)

    @contextmanager
    def exclusive(self):
        """对遵循此协议的写入者加非阻塞锁，竞争时抛出OSError。

        锁由操作系统在进程退出时释放；调用者须持锁完成加载、修改和保存。
        """
        with (self.folder / ".lock").open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def export(self, state: dict) -> None:
        """导出便于查阅的派生文件；中途失败后可重跑，恢复始终读取state。"""
        recording = self.folder / "recording-status.json"
        recording_status = json.loads(recording.read_text(encoding="utf-8")) if recording.exists() else None
        from harness.contracts import digest
        if recording_status and (recording_status.get("run_fingerprint") != state["fingerprint"] or
                                 recording_status.get("event_digest") != digest(state["events"])):
            recording_status = {"status":"stale_receipt","offline_verified":False,"cloud_verified":False}
        summary = {
            "evidence_kind": state["evidence_kind"], "model_experiment": state["model_experiment"],
            "policy": state["config"]["policy"], "current_model": state["current"],
            "task": state["task"],
            "batches": state["completed"], "ledger": state["ledger"],
            "service_predictions": state["predictions"], "feedback": state["feedback"],
            "inspections": state.get("inspections", {}), "methods": state.get("methods", {}),
            "maintenance_exits": state.get("maintenance_exits", {}),
            "maintenance_entries":state.get("maintenance_entries",{}),
            "runtime":state.get("runtime"),"cross_validation":state.get("cross_validation",{}),
            "recording_status":recording_status,
            "swanlab_connected": False,
        }
        if state["task"]["task_id"] == "official-gas-A":
            from harness.evaluation.gas_service import service_repair_summary
            summary["gas_service_repairs"] = service_repair_summary(state)
        atomic_json(self.folder / "summary.json", summary)
        from harness.evaluation.longitudinal import evaluate_history
        atomic_json(self.folder / "longitudinal.json",evaluate_history(state))
        (self.folder / "events.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in state["events"]),
            encoding="utf-8",
        )
        from harness.evaluation.workflow_report import render_workflow
        (self.folder / "流程说明.md").write_text(render_workflow(state), encoding="utf-8")
