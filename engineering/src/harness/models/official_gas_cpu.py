"""官方气体包的真实 CPU 适配器。监督器持有作业，模型仓库存放完整 NPZ 权重。"""
from hashlib import sha256
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from harness.contracts import digest
from harness.data.official_gas import LABELS, exploration_jsonl
from harness.runtime.execution import JobFailure


# 该版本的训练代码和字段已随官方气体包核对。只加载登记的代码文件。
PACKAGE_SHA = "7a992b92d92c807869c3eccf25b3bf82dcdeba9174e161d6e8b61d82c3ec8851"
PACKAGE_FILES = ("__init__.py", "train.py", "data.py", "model.py", "predict.py", "configs/A.json")


class OfficialGasTools:
    evidence_kind = "real_model"
    capabilities = {"fixed_public_training": True, "continuation_epochs": [1, 3], "supervised_execution": True}

    def __init__(self, path, package, batches, runner, *, focus=False, reuse=False, pair_method=None, sequence=None,
                 pair_only=False):
        self.package = package
        self.train = {r.sample_id: r for r in package.train}
        self.explore = {b.batch_id: tuple(r for r in b.samples if r.role == "explore") for b in batches}
        self.runner = runner
        self.worker = {"kind": "official_gas", "execution_enabled": True, "archive": str(path.resolve())}
        self.identity = digest(["official-gas-cpu-service-trace-v1", PACKAGE_SHA, runner.specification])
        self.reuse_identical_fit = reuse
        self.capabilities = dict(type(self).capabilities)
        if pair_only:
            if sequence != "protected" or pair_method != "shape-blend":
                raise ValueError("此对照使用同一受保护的气体修正工具")
            self.capabilities["continuation_epochs"] = []
            self.identity = digest([self.identity, "pair-only-v1"])
        if focus or reuse:
            self.adapter_digest = digest([sha256(Path(__file__).read_bytes()).hexdigest(),
                sha256(Path(__file__).with_name("gas_sampling.py").read_bytes()).hexdigest()])
            self.identity = digest([self.identity, self.adapter_digest, focus, reuse])
        if focus:
            self.capabilities["focus_labels"] = list(package.task.classes)
        if pair_method:
            self.capabilities["gas_pair_methods"] = [pair_method]
            self.identity = digest([self.identity, pair_method,
                                    sha256(Path(__file__).with_name("gas_views.py").read_bytes()).hexdigest()])
        if sequence:
            if sequence not in ("cascade", "protected") or pair_method != "shape-blend":
                raise ValueError("连续修正仅开放已核对的形状融合方法")
            self.capabilities["gas_sequence"] = sequence
            self.identity = digest([self.identity, sequence])
        self.last_execution = None

    def _build_request(self, task, batch_id, action, rows, initialize, parent):
        if task != self.package.task or len(rows) != len(self.train) or any(
                self.train.get(row.sample_id) != row for row in rows) or len({r.sample_id for r in rows}) != len(rows):
            raise ValueError("官方训练只能使用完整的原训练清单")
        request = {"kind": "initial" if initialize else "continue", "batch_id": batch_id,
                   "package_sha256": PACKAGE_SHA, "train_sha256": self.package.train_sha256,
                   "checkpoint_sha256": self.package.checkpoint_sha256, "evidence_kind": self.evidence_kind}
        if not initialize:
            if parent is None or (not action.gas_pair and action.epochs not in self.capabilities["continuation_epochs"]):
                raise ValueError("续训需要当前父权重和已开放的训练遍数")
            request.update(parent=parent, epochs=action.epochs,
                           explore_sha256=sha256(exploration_jsonl(self.explore[batch_id])).hexdigest())
            if action.gas_pair:
                if action.gas_pair.method not in self.capabilities.get("gas_pair_methods", []):
                    raise ValueError("本次运行未开放此修正方法")
                request.update(gas_pair=asdict(action.gas_pair), implementation=self.identity)
                if "gas_sequence" in self.capabilities:
                    request["gas_sequence"] = self.capabilities["gas_sequence"]
            if action.focus_label is not None:
                if action.focus_label not in self.capabilities.get("focus_labels", ()):
                    raise ValueError("工具未开放该类定向续训")
                request.update(focus_label=action.focus_label, adapter_sha256=self.adapter_digest)
        return request

    def fit_identity(self, task, action, rows, parent):
        """官方训练从固定文件按原顺序取数，探索样本只记分，不参与梯度、早停或选权重。"""
        if task != self.package.task or len(rows) != len(self.train) or {r.sample_id: r for r in rows} != self.train:
            raise ValueError("复用需要同一任务和完整的原始训练资料")
        manifest, _ = self.runner.artifacts.read(parent)
        return digest([self.identity, PACKAGE_SHA, self.package.train_sha256, manifest["sha256"],
                       action.epochs, action.focus_label, asdict(action.gas_pair) if action.gas_pair else None])

    def verify_artifact(self, ref):
        self.runner.artifacts.read(ref)

    def _execute(self, request, data):
        self.last_execution = None
        try:
            self.last_execution = self.runner.run(request, data, self.worker)
        except JobFailure as exc:
            self.last_execution = exc.receipt
            raise
        return self.last_execution

    def build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        request = self._build_request(task, batch_id, action, rows, initialize, parent)
        data = "" if initialize else exploration_jsonl(self.explore[batch_id]).decode()
        return self._execute(request, data)["artifact"]

    def recover_build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        return self.runner.recover(self._build_request(task, batch_id, action, rows, initialize, parent))

    def _prediction_request(self, artifact, inputs):
        return {"kind": "predict", "artifact": artifact, "inputs_digest": digest(inputs),
                "package_sha256": PACKAGE_SHA, "evidence_kind": self.evidence_kind}

    def predict(self, artifact, inputs):
        return self._execute(self._prediction_request(artifact, inputs), inputs)["value"]

    def recover_predict(self, artifact, inputs):
        receipt = self.runner.recover(self._prediction_request(artifact, inputs))
        if receipt and receipt["status"] == "completed":
            return {**receipt, "artifact": receipt["value"]}
        return receipt

    def _service_request(self, artifact, inputs):
        # 普通评价只缓存最终类别；服务作业另有稳定身份，中断后取回同一份带过程的回执。
        return {**self._prediction_request(artifact, inputs), "prediction_format": "gas_repair_v1"}

    def _service_receipt(self, receipt, inputs):
        from harness.evaluation.gas_service import validate_repairs
        value = receipt["value"]
        predictions, repairs = value["predictions"], value["gas_repairs"]
        validate_repairs(inputs, predictions, repairs)
        return {**receipt, "value": predictions, "gas_repairs": repairs}

    def predict_service(self, artifact, inputs):
        receipt = self._execute(self._service_request(artifact, inputs), inputs)
        self.last_execution = self._service_receipt(receipt, inputs)
        return self.last_execution["value"]

    def recover_service(self, artifact, inputs):
        receipt = self.runner.recover(self._service_request(artifact, inputs))
        if receipt and receipt["status"] == "completed":
            try:
                saved = self._service_receipt(receipt, inputs)
            except (KeyError, TypeError, ValueError) as exc:
                return {**receipt, "status": "failed", "failure": f"无效的气体服务回执：{exc}"}
            return {**saved, "artifact": saved["value"]}
        return receipt


def execute_official_job(folder, payload, artifacts):
    """仅在受监督工作进程中执行。确认父权重、训练集、验证子集和优化器步数后发布。"""
    from harness.runtime.storage import atomic_json
    import numpy as np

    request = payload["request"]
    archive = Path(payload["backend"]["archive"])
    if request["package_sha256"] != PACKAGE_SHA or sha256(archive.read_bytes()).hexdigest() != PACKAGE_SHA:
        raise ValueError("气体包版本与已审阅代码不一致")
    with ZipFile(archive) as package, TemporaryDirectory(dir=folder) as work:
        work = Path(work)
        if request["kind"] == "initial":
            content = package.read("checkpoint.npz")
            if "sha256:" + sha256(content).hexdigest() != request["checkpoint_sha256"]:
                raise ValueError("初始权重摘要不符")
            return {"artifact": artifacts.create(request, lambda: content)}
        for name in PACKAGE_FILES:
            target = work / "baselines" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read("baselines/" + name))
        sys.path.insert(0, str(work))
        from baselines.model import Model, checkpoint_digest
        from baselines import train as trainer
        from harness.models.gas_views import pair_model
        Model = pair_model(Model)
        trainer.Model = Model

        prediction = request["kind"] == "predict"
        parent_ref = request["artifact"] if prediction else request["parent"]
        parent_manifest, content = artifacts.read(parent_ref)
        if parent_manifest["request"].get("package_sha256") != PACKAGE_SHA:
            raise ValueError("不兼容的父版本")
        parent = work / "parent.npz"
        parent.write_bytes(content)
        original = Model.load(parent)
        if original.meta["classes"] != list(LABELS) or original.meta["profile"] != "A":
            raise ValueError("权重与官方气体任务不符")
        if prediction:
            inputs = payload["rows"]
            if digest(inputs) != request["inputs_digest"] or any(set(r) != {"sample_id", "values"} for r in inputs):
                raise ValueError("预测请求身份或字段不符")
            matrix = np.asarray([r["values"] for r in inputs], dtype=np.float64)
            if matrix.shape != (len(inputs), 128) or not np.isfinite(matrix).all():
                raise ValueError("预测输入应为128项有限测量值")
            traced = request.get("prediction_format") == "gas_repair_v1"
            if traced:
                scores, traces = original.predict_with_repairs(matrix)
            else:
                scores = original.predict(matrix)
            if scores.shape != (len(inputs), len(LABELS)) or not np.isfinite(scores).all():
                raise ValueError("气体预测输出无效")
            labels = {r["sample_id"]: int(i)+1 for r, i in zip(inputs, scores.argmax(axis=1))}
            if traced:
                return {"value": {"predictions": labels, "gas_repairs": {
                    r["sample_id"]: t for r, t in zip(inputs, traces, strict=True)}}}
            return {"value": labels}
        if request["kind"] != "continue" or (not request.get("gas_pair") and request["epochs"] not in (1, 3)):
            raise ValueError("未知气体续训动作")
        train, validation = work / "train.jsonl", work / "explore.jsonl"
        train.write_bytes(package.read("public/train.jsonl"))
        validation.write_bytes(payload["rows"].encode("utf-8"))
        if sha256(validation.read_bytes()).hexdigest() != request["explore_sha256"]:
            raise ValueError("探索子集摘要不符")
        if request.get("gas_pair"):
            from time import perf_counter
            from harness.contracts import GasPairPatch
            from harness.models.gas_views import append_pair, build_pair_memory
            spec = asdict(GasPairPatch(**request["gas_pair"]))
            train_data = trainer.read_dataset(train, "classification", list(LABELS))
            explore = trainer.read_dataset(validation, "classification", list(LABELS), train_data["paths"])
            if train_data["sha256"] != request["train_sha256"] or set(train_data["ids"]) & set(explore["ids"]):
                raise ValueError("修正使用的训练和评价资料不符")
            before = trainer.metrics(original, explore)
            started = perf_counter()
            previous = original.meta.get("gas_pair_patch")
            if previous is None:
                memory = build_pair_memory(spec, train_data["x"], train_data["y"], original.mean, original.scale)
            else:
                memory = append_pair(previous, spec, request.get("gas_sequence"))
            original.meta["gas_pair_patch"] = memory
            fit_seconds = perf_counter()-started
            checkpoint = work / "pair.npz"
            original.save(checkpoint)
            restored = Model.load(checkpoint)
            after = trainer.metrics(restored, explore)
            report = {"gas_pair": spec, "train_sha256": train_data["sha256"], "parent_checkpoint_sha256": checkpoint_digest(parent),
                      "checkpoint_sha256": checkpoint_digest(checkpoint), "training_seconds": 0.,
                      "adapter_fit_seconds": fit_seconds, "initial_epochs": original.meta["epochs"],
                      "final_epochs": restored.meta["epochs"], "validation_before": before, "validation_after": after}
            atomic_json(folder / "training.json", report)
            (folder / "history.jsonl").write_bytes(b"")
            return {"artifact": artifacts.create(request, checkpoint.read_bytes)}
        output = work / "candidate"
        draws = []
        label = request.get("focus_label")
        if label is not None:
            from harness.models.gas_sampling import focused_model
            if type(label) is not int or not 1 <= label <= len(LABELS):
                raise ValueError("指定类别无效")
            # 每个作业独享进程。只替换本次加载的模型类，保留原训练器的权限、报告和检查点。
            trainer.Model = focused_model(Model, label-1, draws)
        report = trainer.train_run("A", train, validation, output, resume=parent, epochs=request["epochs"],
                                   stage_id=request["batch_id"])
        if label is not None:
            report.update(focus_label=label, sampling_history=draws, adapter_sha256=request["adapter_sha256"])
        checkpoint = output / "checkpoint.npz"
        model = Model.load(checkpoint)
        epochs = request["epochs"]
        updates = epochs * ((report["train_rows"] + report["batch_size"] - 1) // report["batch_size"])
        if (report["train_sha256"] != request["train_sha256"] or report["parent_checkpoint_sha256"] != checkpoint_digest(parent)
                or report["checkpoint_sha256"] != checkpoint_digest(checkpoint)
                or model.meta["steps"] != original.meta["steps"] + updates
                or model.meta["epochs"] != original.meta["epochs"] + epochs
                or not np.array_equal(original.mean, model.mean) or not np.array_equal(original.scale, model.scale)):
            raise ValueError("候选的父权重、优化器步数或标准化状态核验失败")
        atomic_json(folder / "training.json", report)
        (folder / "history.jsonl").write_bytes((output / "history.jsonl").read_bytes())
        return {"artifact": artifacts.create(request, checkpoint.read_bytes)}
