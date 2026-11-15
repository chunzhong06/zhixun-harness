"""从已完成的首批筛选记录冻结共同起点；本模块不训练或加载模型对象。"""
from dataclasses import asdict
import json
from pathlib import Path

from harness.contracts import EngineConfig, digest
from harness.data.package_input import read_package
from harness.evaluation.metrics import prediction_score
from harness.evaluation.costs import operation_costs
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.storage import atomic_json


def freeze_initial(study_path, output):
    """等待所有登记候选完成或确认失败，再按首批explore选优。

    核对输入、配置、拟合身份和产物字节；只解析JSON及完整文件摘要。
    selected回执一旦形成即固定，后续结果变化需使用新输出目录。
    """
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    root = study_path.parent
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("schema_version") != 2:
        raise ValueError("起点冻结需要新版准备清单")
    def path(name):
        result = (root/name).resolve()
        if not result.is_relative_to(root):
            raise ValueError("准备清单路径超出研究目录")
        return result
    package = read_package(path("initial-screen/runtime/manifest.json"))
    if len(package.batches) != 1 or digest([asdict(b) for b in package.batches]) != study["screen_input_digest"]:
        raise ValueError("首批输入与准备清单不一致")
    batch = package.batches[0]
    if any(s.role not in {"fit","explore"} for s in batch.samples):
        raise ValueError("起点筛选只允许首批拟合和探索资料")
    rows = tuple(s for s in batch.samples if s.role=="explore")
    outcomes = []
    for entry in study["initial_profiles"]:
        config = EngineConfig(**json.loads(path(entry["screen_config_file"]).read_text(encoding="utf-8")))
        if digest(asdict(config)) != entry["screen_config_digest"]:
            raise ValueError("筛选配置在准备后被修改")
        item = {"profile":entry["profile"],"status":"pending","cost":None}
        folder = path(entry["run_directory"])
        state_path = folder/"state.json"
        if not state_path.exists():
            outcomes.append(item)
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (digest(state["config"]) != digest(asdict(config)) or digest(state["task"]) != digest(asdict(package.task))
                or state["evidence_kind"] != config.execution_mode
                or set(state.get("completed",{})) - {batch.batch_id}
                or set(state.get("samples",{})) != {s.sample_id for s in batch.samples}):
            raise ValueError("筛选记录混入了其他配置、任务或资料")
        item.update(snapshot_digest=digest(state),cost=operation_costs(list(state["operations"].values())),
                    evidence_kind=state["evidence_kind"])
        if any(op["status"]=="started" for op in state["operations"].values()):
            outcomes.append(item)
            continue
        mid = f"{batch.batch_id}/initialize"
        build = state["operations"].get(f"{batch.batch_id}:build:initialize",{})
        if build.get("status")=="failed":
            item.update(status="failed",reason=build.get("error","initial_build_failed"))
        elif batch.batch_id in state.get("completed",{}):
            if state["completed"][batch.batch_id]["input_digest"] != digest(asdict(batch)) or state.get("predictions"):
                raise ValueError("初始筛选批次包含不一致的服务输出")
            if set(state["models"]) != {mid}:
                raise ValueError("起点筛选包含额外模型尝试")
            model = state["models"][mid]
            if (model.get("pipeline") != entry["pipeline"]
                    or model["training_versions"] != entry["initialization"]["training_versions"]):
                raise ValueError("起点产物配方或拟合身份发生改变")
            if not (folder/"models").is_dir():
                raise ValueError("筛选缺少已保存的模型文件")
            manifest, _ = ModelArtifacts(folder/"models").read(model["artifact"])
            if manifest["request"]["data_digest"] != entry["initialization"]["training_digest"]:
                raise ValueError("起点产物拟合内容与共同清单不同")
            evaluation = state["operations"].get(f"{batch.batch_id}:explore:{mid}",{})
            if evaluation.get("status") != "completed":
                item.update(status="failed",reason="initial_exploration_incomplete")
            else:
                evidence = evaluation["value"]
                if (evidence["model_id"] != mid or evidence["sample_versions"] != [f"{s.sample_id}@{s.revision}" for s in rows]
                        or set(evidence["predictions"]) != {s.sample_id for s in rows}):
                    raise ValueError("起点探索比较的模型或资料不一致")
                score = prediction_score(package.task, rows, evidence["predictions"])
                if abs(score-evidence["score"]) > 1e-12:
                    raise ValueError("起点评分与已保存预测不一致")
                item.update(status="completed",score=score,artifact=model["artifact"],
                            artifact_content_sha256=manifest["sha256"])
        outcomes.append(item)
    ready = [item for item in outcomes if item["status"]=="completed"]
    pending = any(item["status"]=="pending" for item in outcomes)
    winner = max(ready,key=lambda item:package.task.utility(item["score"])) if ready and not pending else None
    receipt = {"schema_version":1,"study_digest":digest(study),"screen_input_digest":study["screen_input_digest"],
        "status":"pending_results" if pending else "selected" if winner else "no_valid_candidate",
        "selected":winner["profile"] if winner else None,"outcomes":outcomes,
        "selection_scope":"first_batch_explore_only","training_executed_by_this_command":False}
    if winner:
        entry = next(item for item in study["initial_profiles"] if item["profile"]==winner["profile"])
        selected_config = json.loads(path(entry["config_file"]).read_text(encoding="utf-8"))
        if digest(selected_config) != entry["config_digest"]:
            raise ValueError("共同维护配置在准备后被修改")
        receipt["selected_config_digest"] = digest(selected_config)
    saved = output/"selection-receipt.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous["study_digest"] != receipt["study_digest"] or previous["status"]=="selected" and previous!=receipt:
            raise ValueError("起点已冻结或准备清单已变化，请使用新的选择目录")
    # 输出目录不得指向输入本身；只写两个明确命名文件，不写运行状态。
    if output == root or output.is_relative_to(path("initial-screen")) or output.is_relative_to(path("executed")):
        raise ValueError("选择输出需要独立目录")
    if output.exists() and any(p.name not in {"selected-config.json","selection-receipt.json"} for p in output.iterdir()):
        raise ValueError("选择输出包含其他内容，请使用独立目录")
    config_output = output/"selected-config.json"
    if config_output.exists() and (not winner or digest(json.loads(config_output.read_text(encoding="utf-8"))) != receipt["selected_config_digest"]):
        raise ValueError("已有选择配置与本次冻结结果不同，请使用新的选择目录")
    if winner:
        atomic_json(output/"selected-config.json", selected_config)
    atomic_json(saved, receipt)
    return receipt
