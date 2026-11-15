"""将已发布双修正版带回真实服务入口，与开发留出的原判、改判逐行核对；不续训。"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.contracts import BatchContext, EngineConfig
from harness.data.official_gas import read_official_gas
from harness.evaluation.gas_service import service_repair_summary
from harness.methods.planning.gas_continuation import GasPairPlanner
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import OfficialGasTools, PACKAGE_SHA
from harness.runtime.engine import Harness
from harness.runtime.execution import LocalJobRunner
from harness.runtime.storage import LocalStore


def audit():
    cfg = json.loads((ROOT/"engineering/configs/experiments/gas-frozen-future.json").read_bytes())
    archive = ROOT/cfg["official"]
    package = read_official_gas(archive)
    arm = cfg["arms"]["double"]
    content = (ROOT/arm["path"]).read_bytes()
    if sha256(content).hexdigest() != arm["sha256"]:
        raise ValueError("双修正版与既有对照的冻结权重不同")
    protocol = json.loads((ROOT/"engineering/configs/experiments/gas-source-followup.json").read_bytes())
    raw = (ROOT/protocol["development"]).read_bytes()
    if sha256(raw).hexdigest() != protocol["development_sha256"]:
        raise ValueError("开发留出记录与原对照不同")
    known = {r["id"]: r for r in json.loads(raw) if r["primary"]}
    rows = [s for s in package.validation if s.sample_id in known]
    if len(rows) != len(known) or len(rows) != 414:
        raise ValueError("公开留出身份不齐")
    # 用已冻结权重初始化服务实例，只替换初始模型加载；调用、恢复、反馈与汇总走正式实现。
    class FrozenGasTools(OfficialGasTools):
        def build(self, task, batch_id, action, training, *, initialize=False, parent=None):
            if not initialize or parent is not None:
                raise ValueError("本次复核只加载已存权重")
            return self.runner.artifacts.create({"package_sha256": PACKAGE_SHA, "frozen": arm["sha256"]}, lambda: content)

    with TemporaryDirectory(dir=ROOT/"scratch", prefix="gas-service-") as tmp:
        work = Path(tmp)
        runner = LocalJobRunner(work/"jobs", ModelArtifacts(work/"models"))
        task = package.task
        serve = BatchContext(task.task_id, "issue", 0, package.train + tuple(replace(s, role="serve", label=None) for s in rows))
        delayed = BatchContext(task.task_id, "wait", 1, ())
        answers = tuple(replace(s, available_at=2, revision=2) for s in rows)
        returned = BatchContext(task.task_id, "answers", 2, answers)
        resent = BatchContext(task.task_id, "replayed", 3, answers)
        batches = (serve, delayed, returned, resent)
        tools = FrozenGasTools(archive, package, batches, runner)
        engine = Harness(task, EngineConfig(execution_mode="real_model", recent_cap=len(package.train),
                         maintenance_trigger="never"), tools, GasPairPlanner("shape-blend"), LocalStore(work/"workflow"),
                         allow_real_execution=True)
        state = engine.run(batches[:2])
        if service_repair_summary(state)["outcomes"] != {"pending": len(rows)}:
            raise ValueError("答案返回前出现修正得失")
        issued = state["predictions"]
        inputs = tuple({"sample_id":s.sample_id, "values":list(s.values)} for s in rows)
        model = state["models"][state["current"]]["artifact"]
        recovered = tools.recover_service(model, inputs)
        for s in rows:
            p = issued[s.sample_id]
            trace = p["gas_repair"]
            old = known[s.sample_id]
            if (old["answer"] != s.label or trace["before"] != old["base"] or trace["after"] != old["double"]
                    or trace["after"] != p["prediction"] or trace != recovered["gas_repairs"][s.sample_id]):
                raise ValueError(f"实际服务路径或恢复回执不符：{s.sample_id}")
            expected = [] if old["base"] == old["double"] else [{"method":"shape-blend","reported":1,"target":old["double"]}]
            if trace["steps"] != expected:
                raise ValueError("实际改判方向与冻结输出不符")
        first = service_repair_summary(engine.run(batches[:3]))
        repeated = service_repair_summary(engine.run(batches))
        if first != repeated or first != json.loads((work/"workflow/summary.json").read_bytes())["gas_service_repairs"]:
            raise ValueError("重发、重启或正常导出的得失统计发生变化")
        jobs = list((work/"jobs").glob("*/state.json"))
        if len(jobs) != 1 or json.loads(jobs[0].read_bytes())["status"] != "completed":
            raise ValueError("一次服务请求应只对应一个已完成作业")
        feedback = engine.state["feedback"]
        changed = [{"sample_id":s.sample_id, "group":s.group, "prediction_revision":s.revision,
                    "label_revision":2, "answer":s.label,
                    **feedback[f"service:{s.sample_id}:2"]["gas_repair_result"]}
                   for s in rows if issued[s.sample_id]["gas_repair"]["steps"]]
        old_run = ROOT/"results/engineering/official-gas/runs/state.json"
        legacy = service_repair_summary(json.loads(old_run.read_bytes()))
        if legacy["issued"] != 20 or legacy["outcomes"] != {"unknown_trace":18, "pending":2}:
            raise ValueError("旧样例的已返回答案不能补造原判")
        return {"weights_sha256":arm["sha256"], "historical_rows_sha256":protocol["development_sha256"],
                "legacy_fixture": {"path":old_run.relative_to(ROOT).as_posix(), "summary":legacy},
                "issued_model":state["current"], "changed_rows":changed,
                "matched_before_after_and_rules":len(rows), "service_jobs":len(jobs), "pending_before_answers":len(rows),
                "matches_recovered_receipt":True, "unchanged_after_resend":True, "summary":first}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
