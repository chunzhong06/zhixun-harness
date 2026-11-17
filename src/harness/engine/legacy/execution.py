"""单个本地训练工作进程的限时、资源记录与产物核对。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from harness.core.contracts import digest
from harness.adapters.storage.local import LocalStore, atomic_json


@dataclass(frozen=True)
class ExecutionLimits:
    """工程默认值可配置，不代表官方配额；资源上限由监督进程周期检查。"""
    wall_seconds: float = 30.
    cpu_seconds: float | None = None
    memory_bytes: int | None = None
    poll_seconds: float = .05

    def __post_init__(self):
        for value in (self.wall_seconds, self.poll_seconds, self.cpu_seconds):
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value <= 0):
                raise ValueError("执行时限和采样间隔须为正数")
        if self.memory_bytes is not None and (type(self.memory_bytes) is not int or self.memory_bytes < 1):
            raise ValueError("内存上限无效")


class JobFailure(RuntimeError):
    """任务已确认结束；完整原因和过程由receipt持有。"""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt.get("failure", "worker_failed"))


class JobUnconfirmed(RuntimeError):
    """已登记启动但尚不能确认进程身份；不自行再次启动。"""


class LocalJobRunner:
    """持久任务与工作进程一一对应，恢复仅核对原任务，不自动重训。

    当前工作进程运行登记的本地训练后端；不接收任意shell命令。psutil
    延迟导入，普通内存样例仍可只用标准库。RSS为采样观察值，CPU与
    墙钟分别记录；短暂尖峰可能发生在采样间隔内。
    """
    def __init__(self, folder: Path, artifacts, limits=None, *, python_executable=None):
        self.folder, self.artifacts = Path(folder).resolve(), artifacts
        self.limits = limits or ExecutionLimits()
        self.python = str(Path(python_executable or sys.executable).resolve())
        self.specification = {"implementation": "local-worker-v1", "limits": asdict(self.limits)}
        self.runtime_timeout = None
        self.on_progress = None

    def set_runtime_timeout(self, seconds):
        self.runtime_timeout = seconds

    def cancel(self, request):
        """提交持久取消信号；监督循环按原进程身份执行，不争抢正在持有的运行锁。"""
        folder = self._job(request)
        receipt = self._load(folder / "state.json")
        if receipt is None or receipt["status"] in {"completed", "failed"}:
            return receipt
        atomic_json(folder / "cancel.json", {"request_id":digest(request),"requested_at":time.time()})
        return {"status":"cancel_requested","request_id":digest(request)}

    def _job(self, request):
        return self.folder / digest(request)

    @staticmethod
    def _load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @staticmethod
    def _process(receipt):
        import psutil
        try:
            process = psutil.Process(receipt["pid"])
            if abs(process.create_time() - receipt["process_created_at"]) > .001 or not process.is_running():
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except (psutil.NoSuchProcess, KeyError):
            return None

    @staticmethod
    def _stop(process):
        """只结束已核对身份的工作进程及当前可见子进程，并等待退出。"""
        import psutil
        try:
            children = process.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in [*reversed(children), process]:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([*children, process], timeout=5)
        if alive:
            raise JobUnconfirmed("工作进程尚未确认退出")

    def _finish(self, folder, receipt, request):
        result = self._load(folder / "result.json")
        if result and result.get("nonce") != receipt["nonce"]:
            raise ValueError("工作结果不属于本次任务")
        receipt["progress"] = self._load(folder / "progress.json") or []
        end = (result or {}).get("finished_at") or receipt.get("stopped_at")
        if end is None and receipt["progress"] and receipt["progress"][-1]["phase"] == "published":
            end = receipt["progress"][-1].get("observed_at")
        receipt["wall_seconds"] = max(0., (end or time.time()) - receipt["started_at"])
        if result:
            receipt["cpu_seconds"] = max(receipt.get("cpu_seconds", 0.), result.get("cpu_seconds", 0.))
            receipt["observed_peak_rss_bytes"] = max(receipt.get("observed_peak_rss_bytes", 0), result.get("rss_bytes", 0))
        if receipt.get("failure"):
            receipt["status"] = "failed"
        elif result and result["status"] == "failed":
            receipt.update(status="failed", failure=result["error_type"], error_message=result["error_message"])
        elif request.get("kind") == "predict":
            if result and "value" in result:
                receipt.update(status="completed",value=result["value"])
            else:
                receipt.update(status="failed",failure="worker_exited_without_predictions")
        else:
            reference = self.artifacts.find(request)
            if reference:
                receipt.update(status="completed", artifact=reference,
                               completion_basis="worker_result" if result else "verified_published_artifact")
            else:
                receipt.update(status="failed", failure="worker_exited_without_artifact")
        atomic_json(folder / "state.json", receipt)
        return receipt

    def _wait(self, folder, receipt, request, popen=None):
        import psutil
        while True:
            process = self._process(receipt)
            if process is None:
                break
            try:
                cpu = process.cpu_times()
                receipt["cpu_seconds"] = max(receipt.get("cpu_seconds", 0.), cpu.user + cpu.system)
                receipt["observed_peak_rss_bytes"] = max(receipt.get("observed_peak_rss_bytes", 0), process.memory_info().rss)
            except psutil.NoSuchProcess:
                continue
            elapsed = time.time() - receipt["started_at"]
            progress = self._load(folder / "progress.json") or []
            cursor = receipt.get("progress_cursor", 0)
            for index, update in enumerate(progress[cursor:], cursor):
                if self.on_progress:
                    self.on_progress({"request_id": receipt["request_id"], "index": index, **update})
                receipt["progress_cursor"] = index + 1
            failure = receipt.get("failure")
            cancellation = self._load(folder / "cancel.json")
            if cancellation and cancellation.get("request_id") == digest(request):
                failure = "cancelled"
            if elapsed > receipt["limits"]["wall_seconds"]:
                failure = "wall_limit"
            elif self.limits.cpu_seconds is not None and receipt["cpu_seconds"] > self.limits.cpu_seconds:
                failure = "cpu_limit"
            elif self.limits.memory_bytes is not None and receipt["observed_peak_rss_bytes"] > self.limits.memory_bytes:
                failure = "memory_limit"
            if failure:
                receipt["failure"] = failure
                receipt["stopped_at"] = time.time()
                # 先保存停止决定；若监督进程此刻退出，恢复仍按该决定处理。
                atomic_json(folder / "state.json", receipt)
                self._stop(process)
                break
            time.sleep(self.limits.poll_seconds)
        if popen is not None:
            popen.wait(timeout=5)
        return self._finish(folder, receipt, request)

    def recover(self, request):
        """只接续已有进程或核对发布文件；没有可确认身份时返回None。"""
        folder = self._job(request)
        with LocalStore(folder).exclusive():
            receipt = self._load(folder / "state.json")
            if receipt is None:
                return None
            if receipt["request_id"] != digest(request):
                raise ValueError("任务身份不一致")
            if receipt["status"] in {"completed", "failed"}:
                if receipt["status"] == "completed" and request.get("kind") != "predict":
                    self.artifacts.read(receipt["artifact"])
                return receipt
            worker = self._load(folder / "worker.json")
            if "pid" not in receipt and worker and worker.get("nonce") == receipt["nonce"]:
                receipt.update(pid=worker["pid"], process_created_at=worker["process_created_at"])
            if "pid" not in receipt:
                return None
            return self._wait(folder, receipt, request)

    def run(self, request, rows, backend_config):
        import psutil
        folder = self._job(request)
        # 重入只核对已经登记的任务；异常中断不会隐式增加一次训练。
        if (folder / "state.json").exists():
            receipt = self.recover(request)
            if receipt is None:
                raise JobUnconfirmed("任务启动结果尚未确认")
            if receipt["status"] == "failed":
                raise JobFailure(receipt)
            return receipt
        with LocalStore(folder).exclusive():
            if (folder / "state.json").exists():
                raise JobUnconfirmed("另一个调用者已登记此任务")
            receipt = {"request_id": digest(request), "nonce": uuid4().hex, "status": "launching",
                       "started_at": time.time(), "cpu_seconds": 0., "observed_peak_rss_bytes": 0,
                       "limits": asdict(self.limits), "evidence_kind": request["evidence_kind"]}
            if self.runtime_timeout is not None:
                receipt["limits"]["wall_seconds"] = min(self.limits.wall_seconds, self.runtime_timeout)
            atomic_json(folder / "input.json", {"request": request, "rows": rows, "backend": backend_config,
                "artifact_root": str(self.artifacts.folder), "nonce": receipt["nonce"]})
            atomic_json(folder / "state.json", receipt)
            # 不向训练子进程传递模型服务或账户密钥；Windows窗口保持隐藏。
            keep = {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "TEMP", "TMP", "PATHEXT"}
            env = {k: v for k, v in os.environ.items() if k.upper() in keep}
            env.update(PYTHONPATH=str(__import__("harness.paths",fromlist=["PACKAGE_ROOT"]).PACKAGE_ROOT.parent), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            process = None
            with (folder / "worker.log").open("wb") as log:
                try:
                    process = subprocess.Popen([self.python, "-B", "-m", "harness.engine.legacy.training_worker", str(folder)],
                        cwd=folder, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    receipt.update(status="running", pid=process.pid,
                                   process_created_at=psutil.Process(process.pid).create_time())
                    atomic_json(folder / "state.json", receipt)
                    receipt = self._wait(folder, receipt, request, process)
                except BaseException:
                    if process is not None:
                        owned = self._process(receipt)
                        if owned:
                            self._stop(owned)
                        process.wait(timeout=5)
                    raise
            if receipt["status"] == "failed":
                raise JobFailure(receipt)
            return receipt
