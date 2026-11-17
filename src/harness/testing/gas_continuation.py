"""从官方数据生成确定的接口演练；预设分数与模拟产物始终标成工程样例。"""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json

from harness.core.contracts import BatchContext, EngineConfig, digest
from harness.adapters.datasets.official_gas import LABELS, exploration_jsonl, prediction_label, prediction_request
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.learning.models.training_tools import gas_continuation_argv
from harness.research.methods.planning.gas_continuation import GasContinuationPlanner, gas_issue
from harness.engine.legacy.engine import Harness
from harness.adapters.storage.local import LocalStore, atomic_json
from harness.adapters.datasets.gas_input import CLASS_NAMES


def rehearsal_inputs(package):
    """将验证行分配到互斥的探索、确认和服务组；模拟反馈只在服务之后发放。

    编排行为与预设预测只用于验收。十个优化轮次均来自同一公开验证源批。
    """
    rounds = [[] for _ in range(10)]
    position = Counter()
    for row in package.validation:
        index = position[row.label]
        position[row.label] += 1
        ordinal = index % len(rounds)
        occurrence = index // len(rounds)
        role = "explore" if occurrence % 2 == 0 else "adopt"
        if occurrence == 2 and row.label in (3, 4):
            role = "serve"
        rounds[ordinal].append(replace(row, role=role, available_at=ordinal))

    tables = {stage: {} for stage in ("initial", *(f"round-{i}" for i in range(1, 11)))}
    batches, returned = [], []
    for ordinal, validation in enumerate(rounds):
        bid = f"round-{ordinal+1}"
        rows = list(package.train) if not ordinal else list(returned)
        returned = []
        majority_index = 0
        for row in validation:
            for stage, table in tables.items():
                guess = row.label
                if stage == "initial":
                    if row.role in {"explore", "serve"} and row.label == 4:
                        guess = 5
                    elif ordinal == 0 and row.role == "adopt" and row.label == 5 and majority_index < 7:
                        guess = 2
                    elif ordinal > 0 and row.role == "adopt" and row.label == 4:
                        guess = 5
                elif stage == "round-1" and row.role == "adopt" and row.label == 4:
                    guess = 5
                elif stage == "round-2" and ordinal == 2 and row.label == 3:
                    guess = 5
                table[row.sample_id] = guess
            if row.role == "adopt" and row.label == 5:
                majority_index += 1
            if row.role == "serve":
                rows.append(replace(row, label=None))
                returned.append(replace(row, role="explore", revision=2, available_at=ordinal+1))
            else:
                rows.append(row)
        batches.append(BatchContext(package.task.task_id, bid, ordinal, tuple(rows)))
    return tuple(batches), tables


class GasRehearsalTools:
    """接收真实父产物与参数，只生成预设候选描述；从不执行随包代码。"""
    evidence_kind = "interface_fixture"
    capabilities = {"continuation_epochs": [1, 3], "fixed_public_training": True}

    def __init__(self, package, tables, folder, batches):
        self.package, self.tables = package, tables
        self.artifacts = ModelArtifacts(folder)
        self.train_ids = {r.sample_id: r for r in package.train}
        self.validation_inputs = {r.sample_id: r.values for r in package.validation}
        self.exploration = {b.batch_id: tuple(s for s in b.samples if s.role == "explore") for b in batches}
        self.folder = folder.parent
        self.identity = digest(["official-gas-rehearsal-v2", package.checkpoint_sha256,
                                package.train_sha256, package.validation_sha256, tables])

    def _request(self, task, batch_id, action, rows, initialize, parent):
        if (task != self.package.task or len(rows) != len(self.train_ids) or
                {r.sample_id for r in rows} != self.train_ids.keys() or
                any(r != self.train_ids[r.sample_id] for r in rows)):
            raise ValueError("本次续训只接受完整、未改动的官方训练清单")
        request = {"package": self.package.package_id, "source": self.package.checkpoint_sha256,
                   "train": self.package.train_sha256, "fit_count": len(rows),
                   "validation": self.package.validation_sha256, "evidence_kind": self.evidence_kind}
        if initialize:
            return {**request, "stage": "initial", "kind": "official_checkpoint"}
        if action.epochs not in self.capabilities["continuation_epochs"] or not parent:
            raise ValueError("续训缺少兼容的父检查点或轮数")
        manifest, _ = self.artifacts.read(parent)
        prior = manifest["request"]
        if prior["source"] != request["source"]:
            raise ValueError("父版本不属于本次起点")
        initial_parent = prior["kind"] == "official_checkpoint"
        parent_path = "checkpoint.npz" if initial_parent else f"{prior['stage']}/checkpoint.npz"
        # argv 描述真实工具的参数，候选目录在演练中只包含模拟产物。
        command = gas_continuation_argv(action, checkpoint=parent_path, initial_parent=initial_parent,
                                        training="public/train.jsonl", validation=f"development/{batch_id}/explore.jsonl",
                                        stage=batch_id, output=batch_id)
        return {**request, "stage": batch_id, "kind": "simulated_candidate", "parent": parent,
                "parent_stage": prior["stage"], "parent_sha256": manifest["sha256"], "epochs": action.epochs,
                "explore_sha256": sha256(exploration_jsonl(self.exploration[batch_id])).hexdigest(),
                "budget_sample_passes": len(rows) * action.epochs, "argv": command}

    def build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        request = self._request(task, batch_id, action, rows, initialize, parent)
        if not initialize:
            destination = self.folder / "development" / batch_id / "explore.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(exploration_jsonl(self.exploration[batch_id]))
        content = (self.package.checkpoint if initialize else
                   json.dumps({"simulated": True, "stage": batch_id, "parent": request["parent_sha256"],
                               "epochs": action.epochs}, sort_keys=True).encode())
        return self.artifacts.create(request, lambda: content)

    def recover_build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        request = self._request(task, batch_id, action, rows, initialize, parent)
        ref = self.artifacts.find(request)
        # 模拟构建仅按确定参数写本地文件。尚未发布的请求可在原身份下重建。
        if ref is None:
            ref = self.build(task, batch_id, action, rows, initialize=initialize, parent=parent)
        return {"status": "completed", "artifact": ref}

    def predict(self, artifact, inputs):
        manifest, _ = self.artifacts.read(artifact)
        table = self.tables[manifest["request"]["stage"]]
        predictions = {}
        for row in inputs:
            if set(row) != {"sample_id", "values"}:
                raise ValueError("预测路由只接受样本身份及测量值")
            request = prediction_request(row["values"])
            if tuple(request["inputs"]["features"]) != self.validation_inputs[row["sample_id"]]:
                raise ValueError("预测身份与官方测量值不匹配")
            # 身份只在模拟工具内定位预写响应；真正随包接口只接收 prediction_request。
            label = table[row["sample_id"]]
            response = {"outputs": [{"label": name, "score": float(index == label)}
                                    for index, name in enumerate(LABELS, 1)]}
            predictions[row["sample_id"]] = prediction_label(response)
        return predictions

    def recover_predict(self, artifact, inputs):
        # 纯读取预设响应，没有外部作业。相同请求可准确重建相同回执。
        return {"status": "completed", "artifact": self.predict(artifact, inputs)}


def trace_report(state, package, tools):
    """从已提交决定、真实传给替身的请求和后来返回的服务反馈汇成可读证据。"""
    entries = []
    for bid, batch in sorted(state["completed"].items(), key=lambda pair: int(pair[0].split("-")[1])):
        decision = batch["decision"]
        event = next((e for e in state["events"] if e["key"] == f"{bid}:plan_context:0"), None)
        issue = gas_issue(event["context"]) if event else None
        builds = []
        for trial in batch["trials"]:
            if trial["action"] == "reuse" or not trial.get("model_id"):
                continue
            artifact = state["models"][trial["model_id"]]["artifact"]
            manifest, _ = tools.artifacts.read(artifact)
            request = manifest["request"]
            operation = state["operations"][f"{bid}:build:{trial['action']}"]
            builds.append({"candidate": trial["model_id"], "parent": request["parent_stage"],
                           "parent_sha256": request["parent_sha256"], "artifact_sha256": manifest["sha256"],
                           "train_sha256": request["train"], "fit_count": request["fit_count"],
                           "epochs": request["epochs"], "planned_sample_passes": request["budget_sample_passes"],
                           "command_template": request["argv"], "reason": operation["requested_action"]["reason"]})
        scores = []
        for mid in dict.fromkeys((decision["current_model"], decision["nominee"])):
            value = state["operations"].get(f"{bid}:adopt:{mid}", {}).get("value")
            if value:
                scores.append({"model": mid, "macro_f1": value["score"], "accuracy": value["error_profile"]["accuracy"],
                               "classes": [{**c, "name": CLASS_NAMES[c["label"]]} for c in value["error_profile"]["classes"]]})
        issued = batch["predictions"]
        feedback = [r for r in state["feedback"].values() if r["source"] == "service_feedback"
                    and r["sample_id"] in issued and issue and r["label"] == issue["label"]]
        harmed = []
        if len(scores) == 2:
            previous = {r["label"]: r for r in scores[0]["classes"]}
            for row in scores[1]["classes"]:
                old = previous[row["label"]]
                if old["missed"] < row["missed"] or old["false_positive"] < row["false_positive"]:
                    harmed.append({"name": row["name"], "support": row["count"],
                                   "extra_misses": row["missed"] - old["missed"],
                                   "extra_false_positives": row["false_positive"] - old["false_positive"]})
        entries.append({"round": bid, "current": decision["current_model"], "issue": issue, "builds": builds,
                        "class_regressions": harmed,
                        "confirmed_on": scores, "selected": decision["selected_model"], "why": decision["reason"],
                        "later_feedback": {"status": "observed" if feedback else "no_new_evidence",
                                           "count": len(feedback), "missed": sum(not r["correct"] for r in feedback) if feedback else None,
                                           "observations": [{"id": r["sample_id"], "model": r["model_id"],
                                                             "received_in": r["received_in"]} for r in feedback]}})
    return {"evidence": "官方公开数据与初始权重；预设预测、模拟续训产物与反馈时序",
            "model_experiment": False, "policy": state["config"]["policy"], "metric": package.task.metric,
            "source": {"package": package.package_id, "checkpoint": package.checkpoint_sha256,
                       "training_rows": len(package.train), "validation_rows": len(package.validation)},
            "rounds": entries}


def readable_trace(report):
    lines = ["# 官方气体包维护演练", "", "数据和初始权重来自官方气体包。下表使用预设预测和模拟续训产物，分数仅用于核对决定。",
             "", "| 优化轮 | 当时的提议 | 共同确认资料上的结果 | 下一轮从哪个版本开始 | 后来对原问题的观察 |",
             "| --- | --- | --- | --- | --- |"]
    for r in report["rounds"]:
        attempts, scores, feedback = r["builds"], r["confirmed_on"], r["later_feedback"]
        proposal = "当前观察未提出新的续训动作"
        if attempts:
            parent = attempts[0]["parent"]
            parent_name = "官方初始模型" if parent == "initial" else f"第{parent.split('-')[1]}轮采用的版本"
            reason = attempts[0]["reason"]
            if r["issue"]:
                reason = reason.replace(r["issue"]["name"], CLASS_NAMES[r["issue"]["label"]])
            proposal = f"从{parent_name}开始。{reason}"
        result = r["why"]
        if len(scores) == 2:
            base, new = scores
            result = (f"Macro-F1 {base['macro_f1']:.3f} → {new['macro_f1']:.3f}，"
                      f"准确率 {base['accuracy']:.1%} → {new['accuracy']:.1%}。{result}。")
            for c in r["class_regressions"]:
                if c["extra_misses"] > 0:
                    result += f"{c['name']}有{c['support']}条，新增漏判{c['extra_misses']}条。"
        observed = "尚无该问题的新反馈" if r["issue"] else "当轮没有待复查的类别问题"
        if feedback["count"]:
            observed = f"{feedback['observations'][0]['received_in']} 返回{feedback['count']}条该类样例，漏判{feedback['missed']}条"
        lines.append(f"| {r['round']} | {proposal} | {result} | {r['selected']} | {observed} |")
    return "\n".join(lines) + "\n"


def run_rehearsal(package, output, policy, *, batches_to_run=None, after_event=None):
    """可在同一目录重启；正式训练与外部服务没有通往本演练的调用路径。"""
    batches, tables = rehearsal_inputs(package)
    tools = GasRehearsalTools(package, tables, output / "models", batches)
    config = EngineConfig(policy=policy, recent_cap=len(package.train), max_trials=1)
    engine = Harness(package.task, config, tools, GasContinuationPlanner(), LocalStore(output), after_event=after_event)
    state = engine.run(batches if batches_to_run is None else batches[:batches_to_run])
    report = trace_report(state, package, tools)
    atomic_json(output / "official-gas-trace.json", report)
    (output / "维护记录.md").write_text(readable_trace(report), encoding="utf-8")
    return report
