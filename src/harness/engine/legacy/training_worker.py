"""登记后端的独立工作入口；工程任务主动封闭网络和真实模型库导入。"""
from __future__ import annotations

import builtins
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time

from harness.core.contracts import ActionPlan, Sample, TaskSpec, digest
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.adapters.storage.local import atomic_json
from harness.learning.models.training_tools import SklearnBackend, SklearnRecipe, TrainingModelTools


def main():
    import psutil
    folder = Path(sys.argv[1]).resolve()
    payload = json.loads((folder / "input.json").read_text(encoding="utf-8"))
    nonce, request, backend_config = payload["nonce"], payload["request"], payload["backend"]
    process = psutil.Process()
    atomic_json(folder / "worker.json", {"nonce": nonce, "pid": os.getpid(),
                                         "process_created_at": process.create_time()})
    def no_network(*args, **kwargs):
        raise RuntimeError("training_worker_network_disabled")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    progress = []
    started = time.perf_counter()
    def report(phase, **details):
        progress.append({"phase": phase, "elapsed_seconds": time.perf_counter() - started,
                         "observed_at": time.time(), **details})
        atomic_json(folder / "progress.json", progress)
    try:
        if backend_config["kind"] in {"official_gas", "official_rul"} and backend_config.get("execution_enabled") is True:
            if backend_config["kind"] == "official_gas" and "gas_fit_query" in request:
                from harness.learning.models.gas_fit_queries import execute_query_job as execute_official_job
            elif backend_config["kind"] == "official_gas":
                from harness.learning.models.official_gas_cpu import execute_official_job
            elif "bearing_stump" in request:
                from harness.learning.models.bearing_stumps import execute_stump_job as execute_official_job
            elif "readout_recipe" in request:
                from harness.learning.models.readout_training import execute_readout_job as execute_official_job
            elif "trajectory_recipe" in request:
                from harness.learning.models.trajectory_training import execute_trajectory_job as execute_official_job
            else:
                from harness.learning.models.official_rul_cpu import execute_official_job
            result = execute_official_job(folder, payload, ModelArtifacts(Path(payload["artifact_root"])))
            cpu = process.cpu_times()
            atomic_json(folder / "result.json", {"nonce": nonce, "status": "completed", **result,
                        "cpu_seconds": cpu.user + cpu.system, "rss_bytes": process.memory_info().rss,
                        "finished_at": time.time()})
            return
        if backend_config["kind"] == "fixture":
            original_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split(".")[0] in {"sklearn", "catboost", "joblib", "torch", "transformers", "openai", "pytrio", "swanlab"}:
                    raise RuntimeError("fixture_worker_forbids_real_model_import")
                return original_import(name, *args, **kwargs)
            builtins.__import__ = guarded
            from harness.adapters.executors.fixture_backend import FixtureTrainingBackend
            backend = FixtureTrainingBackend(backend_config["fixture"])
        elif backend_config["kind"] == "sklearn" and backend_config.get("execution_enabled") is True:
            backend = SklearnBackend(SklearnRecipe(**backend_config["recipe"]), execution_enabled=True)
        else:
            raise ValueError("worker_backend_not_enabled")
        task = TaskSpec.parse(request["task"])
        if request.get("kind") == "predict":
            if request["backend_specification"] != backend.specification:
                raise ValueError("prediction_backend_mismatch")
            tools = TrainingModelTools(task,backend,ModelArtifacts(Path(payload["artifact_root"])))
            inputs = tuple(request["inputs"])
            if any(set(item)!={"sample_id","values"} for item in inputs):
                raise ValueError("prediction_input_contains_unknown_fields")
            report("predicting")
            value=tools.predict(request["artifact"],inputs)
            cpu=process.cpu_times()
            atomic_json(folder/"result.json",{"nonce":nonce,"status":"completed","value":value,
                "cpu_seconds":cpu.user+cpu.system,"rss_bytes":process.memory_info().rss,"finished_at":time.time()})
            return
        action = ActionPlan.parse(request["action"])
        rows = tuple(Sample.parse(row) for row in payload["rows"])
        artifacts = ModelArtifacts(Path(payload["artifact_root"]))
        tools = TrainingModelTools(task, backend, artifacts)
        actual = tools.prepare(task, request["batch_id"], action, rows, initialize=request["initialize"], base=request.get("base"))
        actual["execution"] = request["execution"]
        if digest(actual) != digest(request):
            raise ValueError("worker_request_mismatch")
        report("prepared")
        behavior = backend_config.get("fixture_behavior", {})
        time.sleep(behavior.get("sleep_seconds", 0))
        if behavior.get("fail_before_build"):
            raise RuntimeError("fixture_build_failure")
        report("building")
        for step in range(behavior.get("progress_steps", 0)):
            # 预写进度只核对上报与取消链路，明确标注其模拟来源。
            report("training", step=step + 1, metrics={"fixture_progress": step + 1},
                   evidence_kind="interface_fixture")
            time.sleep(.06)
        def produce():
            return tools.produce(request, action, rows)
        reference = artifacts.create(request, produce)
        report("published")
        if behavior.get("exit_after_publish"):
            os._exit(23)  # 故障样例：产物已发布，完成回执尚未写入。
        cpu = process.cpu_times()
        atomic_json(folder / "result.json", {"nonce": nonce, "status": "completed", "artifact": reference,
            "cpu_seconds": cpu.user + cpu.system, "rss_bytes": process.memory_info().rss, "finished_at": time.time()})
    except Exception as exc:
        cpu = process.cpu_times()
        atomic_json(folder / "result.json", {"nonce": nonce, "status": "failed", "error_type": type(exc).__name__,
            "error_message": str(exc)[:500], "cpu_seconds": cpu.user + cpu.system,
            "rss_bytes": process.memory_info().rss, "finished_at": time.time()})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
