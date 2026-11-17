"""从已保存的运行事件生成SwanLab记录；记录失败可独立补交，不重做模型步骤。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

from harness.core.contracts import digest
from harness.adapters.storage.local import LocalStore, atomic_json


def append_epochs(records, state, event_id, history):
    """保留任务主指标及其原始单位，固定对照的训练使用单独事件身份。"""
    metric = state["task"]["metric"]
    for epoch in history["epochs"]:
        records.append({"step": len(records), "event_id": f"{event_id}/epoch-{epoch['epoch']}",
            "kind": "training_epoch", "evidence_kind": state["evidence_kind"],
            "metrics": {"training/loss": epoch["loss"], "training/step": epoch["step"],
                        f"training/validation_{metric}": epoch["validation"][metric]},
            "detail": {"parent_sha256": history["parent"], "candidate_sha256": history["candidate"], **epoch}})


def tracking_records(state: dict, audit: list[dict], training_histories=None, comparison=None) -> list[dict]:
    """按控制器的持久事件排序，关联计划请求与结果，不复制模型二进制。

    仅在相应操作完成/失败事件处附加终态回执。这样中途导出的前缀保持
    不变，后续补齐数据不会改写已经记录的步骤。指标沿用任务原单位。
    """
    contexts = {e["key"]: e["context"] for e in state["events"]
                if e["kind"] in {"plan_context", "inspection_context"}}
    records = []
    for event in state["events"]:
        detail = deepcopy(event)
        metrics = {}
        if event["kind"] in {"operation_completed", "operation_failed"}:
            op = state["operations"][event["key"]]
            detail["operation_receipt"] = {k: op[k] for k in
                ("status", "demo_units", "elapsed_seconds", "name", "batch_id")}
            if op.get("requested_action"):
                detail["operation_receipt"]["requested_action"] = deepcopy(op["requested_action"])
            metrics["operation/demo_units"] = op["demo_units"]
            metrics["operation/seconds"] = op["elapsed_seconds"]
            metrics["operation/failed"] = int(op["status"] == "failed")
            if op.get("execution"):
                execution = op["execution"]
                detail["execution"] = deepcopy(execution)
                metrics["worker/cpu_seconds"] = execution["cpu_seconds"]
                metrics["worker/observed_peak_rss_bytes"] = execution["observed_peak_rss_bytes"]
            value = op.get("value", {})
            if isinstance(value,dict) and {"score","purpose","objective"} <= set(value) and isinstance(value["objective"],dict):
                metric = state["task"]["metric"]
                purpose = value["purpose"]
                metrics[f"{purpose}/{metric}"] = value["score"]
                detail["evaluation"] = {k: value[k] for k in
                    ("score", "sample_ids", "sample_versions", "model_id", "purpose", "objective")}
            if op["name"].startswith("cv:") and op["name"].endswith(":score") and type(value) in {int,float}:
                metrics[f"cv/{state['task']['metric']}"] = value
            if op["name"].startswith(("plan:", "inspect_plan:")):
                index = op["name"].split(":")[1]
                kind = "inspection_context" if op["name"].startswith("inspect_plan:") else "plan_context"
                context_key = op.get("context_key", f"{op['batch_id']}:{kind}:{index}")
                matches = [a for a in audit if a.get("context_digest") == digest(contexts[context_key])]
                detail["context_key"] = context_key
                detail["service_receipts"] = [{k: a[k] for k in
                    ("request_id", "attempt", "status", "usage", "elapsed_seconds", "evidence_kind", "real_model_called")}
                    for a in matches]
                # 没有用量时保留未知标记；人工模拟值始终带来源标识。
                metrics["planner/usage_known"] = int(bool(matches) and all(a["usage"] is not None for a in matches))
                if matches and all(a["usage"] is not None for a in matches):
                    metrics["planner/tokens"] = sum(a["usage"]["total_tokens"] for a in matches)
        if event["kind"] == "adoption_decision":
            metrics["decision/comparison_approved"] = int(event["selected_model"] != event["current_model"])
            if event.get("improvement") is not None:
                metrics["decision/utility_gain"] = event["improvement"]
        if event["kind"] == "output_fallback":
            metrics["decision/output_fallback"] = 1
        if event["kind"] == "execution_progress":
            for key,value in event["progress"].get("metrics",{}).items():
                if type(value) in {int,float}:
                    metrics[f"training/{key}"] = value
        if event["kind"] == "feedback_received":
            key = "correct" if "correct" in event else "absolute_error"
            metrics[f"feedback/{key}"] = float(event[key])
        records.append({"step": len(records), "event_id": event["key"], "kind": event["kind"],
                        "evidence_kind": state["evidence_kind"], "metrics": metrics, "detail": detail})
        history = (training_histories or {}).get(event["key"])
        if history and event["kind"] == "operation_completed":
            append_epochs(records, state, event["key"], history)
    if comparison:
        # 事后对照排在已提交事件之后，独立费用不回填到计划器当时所见的状态。
        if comparison.get("control_history"):
            append_epochs(records, state, "fixed_control", comparison["control_history"])
        records.append({"step": len(records), "event_id": "post_run/comparison", "kind": "comparison",
                        "evidence_kind": state["evidence_kind"], "metrics": comparison["metrics"],
                        "detail": comparison["detail"]})
    # 数据类中的元组在JSON落盘后变成数组，比较前统一为同一种持久形态。
    return json.loads(json.dumps(records, ensure_ascii=False, allow_nan=False))


class SwanLabRecorder:
    """维护本地记录队列，按批导出到SwanLab离线SDK或其模拟器。

    enable_sdk=True时延迟导入真实SDK，
    且固定offline；本类不登录账户或上传云端。每次导出用独立分段实验，
    共同group对应同一次Harness运行，避免假定offline支持云端resume。
    cursor仅在finish返回后推进，表示SDK调用已完成，不表示云端验收。
    """
    def __init__(self, folder: Path, *, sdk=None, enable_sdk=False, project="harness-engineering"):
        if sdk is not None and getattr(sdk, "simulated", False) is not True and not enable_sdk:
            raise ValueError("真实SwanLab SDK执行尚未开启")
        self.folder, self.sdk = Path(folder), sdk
        self.enable_sdk, self.project = enable_sdk, project
        self.store = LocalStore(self.folder)

    def sync(self, state: dict, audit: list[dict], *, training_histories=None, comparison=None) -> dict:
        """保存全部本地记录并补交未完成部分；SDK错误只记录类别，可稍后重试。

        调用者在停止修改state后同步（例如运行结束或中断处）。本地状态或
        既有事件前缀不一致会报错；不会把不同运行拼成一条证据。
        """
        records = tracking_records(state, audit, training_histories, comparison)
        with self.store.exclusive():
            meta = self.store.load() or {
                "fingerprint": state["fingerprint"], "project": self.project,
                "run_id": "h" + uuid4().hex, "cursor": 0, "segments": [],
                "cloud_verified": False, "model_experiment": state["model_experiment"],
                "simulated_sdk": bool(getattr(self.sdk, "simulated", False)),
            }
            if meta["fingerprint"] != state["fingerprint"] or meta["project"] != self.project:
                raise ValueError("训练记录目录属于另一运行或项目")
            if meta["simulated_sdk"] != bool(getattr(self.sdk, "simulated", False)):
                raise ValueError("模拟与真实SDK记录应使用独立目录")
            outbox = self.folder / "records.json"
            previous = json.loads(outbox.read_text(encoding="utf-8")) if outbox.exists() else []
            if records[:len(previous)] != previous:
                raise ValueError("已记录事件被修改或缩短")
            atomic_json(outbox, records)
            self.store.save(meta)
            if meta["cursor"] == len(records):
                return meta
            if self.sdk is None and not self.enable_sdk:
                return {**meta, "status": "prepared"}
            segment = {"index": len(meta["segments"]), "start": meta["cursor"],
                       "end": len(records), "status": "started"}
            meta["segments"].append(segment)
            self.store.save(meta)
            run = None
            try:
                sdk = self.sdk
                if sdk is None:
                    import swanlab as sdk
                options = {}
                if self.enable_sdk:
                    # 工程离线记录只保存显式提供的事件，不自动收集账户、Git或硬件信息。
                    options["settings"] = sdk.Settings(api_key=None, interactive=False,
                        probe=sdk.Settings.Probe(hardware=False, runtime=False, requirements=False,
                            conda=False, git=False, swanlab=False, monitor=False),
                        terminal=sdk.Settings.Terminal(proxy_type="none"))
                run = sdk.init(project=self.project,
                    experiment_name=f"{state['config']['policy']}-{meta['run_id']}-part{segment['index']}",
                    group=meta["run_id"], mode="offline", logdir=str(self.folder / "swanlog"),
                    config={"task": state["task"], "harness": state["config"],
                            "evidence_kind": state["evidence_kind"], "fingerprint": state["fingerprint"],
                            "model_experiment": state["model_experiment"]}, **options)
                for record in records[segment["start"]:segment["end"]]:
                    payload = {**record["metrics"], "trace/event": sdk.Text(
                        json.dumps(record, ensure_ascii=False, allow_nan=False), caption=record["event_id"])}
                    run.log(payload, step=record["step"])
                run.finish()
            except Exception as exc:
                segment.update(status="failed", error=type(exc).__name__)
                if run is not None:
                    try:
                        run.finish(state="crashed", error="recording_failed")
                    except Exception:
                        pass  # 未完成的分段留在本地，下一次从原cursor补交。
            else:
                segment["status"] = "sdk_finished"
                meta["cursor"] = segment["end"]
            self.store.save(meta)
            return meta
