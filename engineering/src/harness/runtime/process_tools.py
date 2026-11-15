"""把既有训练/保存接口接到工作进程，预测仍按已发布模型版本读取。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

from harness.contracts import digest
from harness.runtime.execution import JobFailure
from harness.models.training_tools import SklearnBackend, TrainingModelTools
from harness.testing.fixtures import FixtureTrainingBackend


class ProcessTrainingTools(TrainingModelTools):
    def __init__(self, task, backend, artifacts, runner, *, fixture_behavior=None):
        super().__init__(task, backend, artifacts)
        self.runner, self.last_execution = runner, None
        behavior = fixture_behavior or {}
        if set(behavior) - {"sleep_seconds", "fail_before_build", "exit_after_publish", "progress_steps"}:
            raise ValueError("未知工程故障设置")
        delay = behavior.get("sleep_seconds", 0)
        if type(delay) not in {int, float} or not math.isfinite(delay) or not 0 <= delay <= 60:
            raise ValueError("工程延迟无效")
        if any(type(behavior[k]) is not bool for k in ("fail_before_build", "exit_after_publish") if k in behavior):
            raise ValueError("工程故障开关无效")
        if type(behavior.get("progress_steps", 0)) is not int or not 0 <= behavior.get("progress_steps", 0) <= 20:
            raise ValueError("工程进度步数无效")
        if isinstance(backend, FixtureTrainingBackend):
            self.worker = {"kind": "fixture", "fixture": deepcopy(backend.tools.fixture), "fixture_behavior": behavior}
        elif isinstance(backend, SklearnBackend):
            if behavior:
                raise ValueError("真实后端不接受故障样例配置")
            self.worker = {"kind": "sklearn", "recipe": asdict(backend.recipe),
                           "execution_enabled": backend.execution_enabled}
        else:
            raise ValueError("该后端尚未登记工作进程适配")
        self.identity = "process-training-v1:" + digest([self.identity, runner.specification, self.worker])

    @property
    def capabilities(self):
        return {"stage_progress": True, "training_metrics": self.worker["kind"] == "fixture",
                "local_corrections": hasattr(self.backend, "fit_correction"),
                "supervised_execution":True,
                "cancel": True, "resume_training": False, "recover_published_artifact": True}

    def set_runtime_timeout(self, seconds):
        self.runner.set_runtime_timeout(seconds)

    def set_progress_callback(self, callback):
        self.runner.on_progress = callback

    def predict(self, artifact, inputs):
        """加载和预测也由工作进程监督；同模型同输入复用已确认结果。"""
        self.last_execution = None
        request = {"kind":"predict","task":asdict(self.task),"artifact":artifact,"inputs":list(inputs),
                   "backend_specification":self._specification,"execution":self.runner.specification,
                   "evidence_kind":self.evidence_kind}
        # 预测没有训练进度，不沿用上一个构建操作的回调身份。
        callback,self.runner.on_progress = self.runner.on_progress,None
        try:
            self.last_execution = self.runner.run(request,[],self.worker)
        except JobFailure as exc:
            self.last_execution = exc.receipt
            raise
        finally:
            self.runner.on_progress = callback
        return self.last_execution["value"]

    def recover_predict(self, artifact, inputs):
        request = {"kind":"predict","task":asdict(self.task),"artifact":artifact,"inputs":list(inputs),
                   "backend_specification":self._specification,"execution":self.runner.specification,
                   "evidence_kind":self.evidence_kind}
        receipt=self.runner.recover(request)
        if receipt and receipt["status"] == "completed":
            # 控制器的通用恢复契约以artifact字段携带原操作返回值。
            return {**receipt,"artifact":receipt["value"]}
        return receipt

    def prepare(self, *args, **kwargs):
        request = super().prepare(*args, **kwargs)
        request["execution"] = {**self.runner.specification, "worker_config_digest": digest(self.worker)}
        return request

    def build(self, task, batch_id, action, rows, *, initialize=False, base=None):
        self.last_execution = None
        request = self.prepare(task, batch_id, action, rows, initialize=initialize, base=base)
        try:
            self.last_execution = self.runner.run(request, [asdict(row) for row in rows], self.worker)
        except JobFailure as exc:
            self.last_execution = exc.receipt
            raise
        return self.last_execution["artifact"]

    def recover_build(self, task, batch_id, action, rows, *, initialize=False, base=None):
        """恢复只核对同请求已有任务；返回None表示不能确认，不创建新训练。"""
        request = self.prepare(task, batch_id, action, rows, initialize=initialize, base=base)
        self.last_execution = self.runner.recover(request)
        return self.last_execution
