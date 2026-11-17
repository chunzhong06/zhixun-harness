"""两项寿命任务从在用检查点续训；所有新权重的计算均交给带限额的工作进程。"""
from hashlib import sha256
from contextlib import contextmanager
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from harness.core.contracts import digest
from harness.adapters.datasets.official_rul import ARCHIVE_SHA
from harness.engine.legacy.execution import JobFailure
from harness.adapters.storage.local import atomic_json


def read_parent(artifacts, reference, source):
    """沿已发布产物逐级核对任务、内容及父引用，直到官方根权重。兼容首次维护的旧清单。"""
    first, content = artifacts.read(reference)
    node, seen = first, {reference["key"]}
    depth = 0
    while True:
        request = node["request"]
        if any(request.get(k) != v for k, v in source.items()):
            raise ValueError("寿命权重来自其他任务或官方起点")
        if request["kind"] == "initial":
            if node["sha256"] != source["checkpoint_sha256"]:
                raise ValueError("初始权重内容没有对应官方文件")
            return first, content, depth
        if request["kind"] != "continue" or request.get("epochs") != 1:
            raise ValueError("权重没有受支持的续训来源")
        ancestor = request["parent"]
        if ancestor["key"] in seen:
            raise ValueError("父权重链重复")
        seen.add(ancestor["key"])
        node, _ = artifacts.read(ancestor)
        if "parent_sha256" in request and request["parent_sha256"] != node["sha256"]:
            raise ValueError("续训清单与实际父权重不一致")
        depth += 1


class RulPilotTools:
    evidence_kind = "real_model"
    capabilities = {"fixed_public_training": True, "continuation_epochs": [1],
                    "group_error_direction": True, "device_bias": True, "supervised_execution": True}

    def __init__(self, task_name, archive, package, fit, discovery, original_predictions, runner, *, batches=None,
                 reuse_identical_fit=False):
        self.package, self.runner = package, runner
        if reuse_identical_fit and (task_name != "bearing" or batches is None):
            raise ValueError("同权重复用目前只核对了轴承逐轮官方训练器")
        self.reuse_identical_fit = reuse_identical_fit
        self.worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        if reuse_identical_fit:
            import numpy as np
            self.numeric_runtime = (sys.version_info[:3], np.__version__)
        self.fit = {r.sample_id: r for r in fit}
        self.discovery_ids = sorted(r.sample_id for r in discovery)
        self.values = {r.sample_id: r.values for r in package.validation}
        self.original = original_predictions
        if self.original is not None and self.values.keys() != self.original.keys():
            raise ValueError("原版预测须覆盖同一份公开验证输入")
        self.worker = {"kind": "official_rul", "task": task_name, "archive": str(archive.resolve()),
                       "execution_enabled": True}
        self.source = {"package_sha256": ARCHIVE_SHA[task_name], "checkpoint_sha256": package.checkpoint_sha}
        self.identity = digest(["official-rul-single-update-v1", self.source, sorted(self.fit),
                                self.discovery_ids, digest(self.original), runner.specification])
        self.round_discovery = None
        if batches is not None:
            self.round_discovery = {b.batch_id: sorted(s.sample_id for s in b.samples if s.role == "explore"
                                                       and s.available_at == b.ordinal) for b in batches}
            if len(self.round_discovery) != len(batches) or not all(self.round_discovery.values()):
                raise ValueError("每个续训轮需要独立身份和当轮可见的探索资料")
            self.identity = digest([self.identity, self.round_discovery, self.worker_sha, reuse_identical_fit,
                                    getattr(self, "numeric_runtime", None)])
        self.last_execution = None

    def initial(self):
        return self.runner.artifacts.create({"kind": "initial", **self.source}, lambda: self.package.checkpoint)

    def _fit_request(self, task, action, rows, parent):
        if task != self.package.task or len(rows) != len(self.fit) or {r.sample_id:r for r in rows} != self.fit:
            raise ValueError("训练清单与冻结的公开训练输入不符")
        if not parent or action.epochs != 1 or action.recipe != "recent":
            raise ValueError("本次寿命工具从父权重接续一遍")
        request = {"kind": "continue", "evidence_kind": self.evidence_kind, **self.source,
                   "worker_sha256": self.worker_sha, "parent": parent, "epochs": 1, "fit_ids": sorted(self.fit)}
        if self.round_discovery is None:
            if parent != self.initial():
                raise ValueError("单次对照的父权重必须是官方起点")
        else:
            manifest, _, _ = read_parent(self.runner.artifacts, parent, self.source)
            request["parent_sha256"] = manifest["sha256"]
        if self.reuse_identical_fit:
            request["fit_runtime"] = self.numeric_runtime
        return request

    def fit_identity(self, task, action, rows, parent):
        """冻结原包含训练文件、官方代码和配方，完整父权重含优化器、尺度和种子。"""
        return digest(self._fit_request(task, action, rows, parent))

    def verify_artifact(self, reference):
        read_parent(self.runner.artifacts, reference, self.source)

    def _diagnostic_inputs(self, batch_id):
        if self.round_discovery is None:
            return {"exploration_ids": self.discovery_ids}
        if batch_id not in self.round_discovery:
            raise ValueError("本轮未登记探索资料")
        return {"stage": batch_id, "exploration_ids": self.round_discovery[batch_id]}

    def continuation_request(self, task, batch_id, action, rows, parent):
        request = self._fit_request(task, action, rows, parent)
        diagnostics = self._diagnostic_inputs(batch_id)
        return request if self.reuse_identical_fit else {**request, **diagnostics}

    def _execute(self, request, data, backend=None):
        self.last_execution = None
        try:
            self.last_execution = self.runner.run(request, data, backend or self.worker)
        except JobFailure as exc:
            self.last_execution = exc.receipt
            raise
        return self.last_execution

    def build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        self.last_execution = None
        if initialize:
            return self.initial()
        request = self.continuation_request(task, batch_id, action, rows, parent)
        backend = self.worker
        if self.reuse_identical_fit:
            # 仅首个物理作业写官方训练日志；各策略仍在各自当轮的探索和确认样本上打分。
            backend = {**backend, "training_diagnostics": self._diagnostic_inputs(batch_id)}
        return self._execute(request, [], backend)["artifact"]

    def recover_build(self, task, batch_id, action, rows, *, initialize=False, parent=None):
        if initialize:
            return {"status": "completed", "artifact": self.initial()}
        return self.runner.recover(self.continuation_request(task, batch_id, action, rows, parent))

    def _prediction_request(self, artifact, inputs):
        return {"kind": "predict", "evidence_kind": self.evidence_kind, **self.source,
                "artifact": artifact, "inputs_digest": digest(inputs)}

    def predict(self, artifact, inputs):
        self.last_execution = None
        for row in inputs:
            if set(row) != {"sample_id", "values"} or tuple(row["values"]) != self.values[row["sample_id"]]:
                raise ValueError("预测输入与公开测量值不符")
        if self.original is not None and artifact == self.initial():
            return {r["sample_id"]: self.original[r["sample_id"]] for r in inputs}
        return self._execute(self._prediction_request(artifact, inputs), inputs)["value"]

    def recover_predict(self, artifact, inputs):
        if self.original is not None and artifact == self.initial():
            return {"status": "completed", "artifact": self.predict(artifact, inputs)}
        receipt = self.runner.recover(self._prediction_request(artifact, inputs))
        return {**receipt, "artifact": receipt["value"]} if receipt and receipt["status"] == "completed" else receipt


def _subset(archive, split, ids, path):
    """按原始文件次序取行。训练和训练器自带的进度评价使用两个不同的公开文件。"""
    remaining = set(ids)
    if not remaining or len(remaining) != len(ids):
        raise ValueError("训练或探索清单为空或含重复身份")
    with archive.open(f"public/{split}.jsonl") as source, path.open("wb") as out:
        for line in source:
            sid = json.loads(line)["id"]
            if sid in remaining:
                out.write(line)
                remaining.remove(sid)
    if remaining:
        raise ValueError("指定样本不在对应的官方划分中")


@contextmanager
def _trainer(archive, work, profile):
    """每份原包的相对导入归到独立包名；在同一进程连续检查两份包不会复用上次的模块。"""
    sources = json.loads(archive.read("training.json"))["source_sha256"]
    code = work / "baselines"
    for name in ("__init__.py", *sources, f"configs/{profile}.json"):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read("baselines/" + name)
        if name in sources and "sha256:" + sha256(data).hexdigest() != sources[name]:
            raise ValueError("随包训练源码与官方报告不符")
        path.write_bytes(data)
    alias = "_official_rul_" + work.name
    spec = spec_from_file_location(alias, code / "__init__.py", submodule_search_locations=[str(code)])
    module = module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        yield import_module(".train", alias)
    finally:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]


def execute_official_job(folder, payload, artifacts, *, inspect_resume=False):
    import numpy as np
    request, backend = payload["request"], payload["backend"]
    archive_path = Path(backend["archive"])
    checksum = ARCHIVE_SHA[backend["task"]]
    if request["package_sha256"] != checksum or sha256(archive_path.read_bytes()).hexdigest() != checksum:
        raise ValueError("寿命任务的官方包或版本不匹配")
    profile = {"bearing": "B", "engine": "C"}[backend["task"]]
    with ZipFile(archive_path) as archive, TemporaryDirectory(dir=folder) as tmp, _trainer(archive, Path(tmp), profile) as trainer:
        work = Path(tmp)
        predicting = request["kind"] == "predict"
        if inspect_resume and predicting:
            raise ValueError("检查权重交接时仅接受续训请求")
        official_sha = sha256(archive.read("checkpoint.npz")).hexdigest()
        source = {"package_sha256": checksum, "checkpoint_sha256": official_sha}
        manifest, content, depth = read_parent(artifacts, request["artifact"] if predicting else request["parent"], source)
        if request["checkpoint_sha256"] != official_sha:
            raise ValueError("本次请求与官方权重不符")
        parent = work / "parent.npz"
        parent.write_bytes(content)
        model = trainer.Model.load(parent)
        if model.meta["profile"] != profile or model.meta["task"] != "regression":
            raise ValueError("预测器与任务不符")
        if predicting:
            rows = payload["rows"]
            if digest(rows) != request["inputs_digest"] or any(set(r) != {"sample_id", "values"} for r in rows):
                raise ValueError("候选预测请求含非测量数据或身份不一致")
            values = model.predict(np.asarray([r["values"] for r in rows], dtype=np.float64))
            if values.shape != (len(rows),) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("寿命输出的数量或数值无效")
            return {"value": {r["sample_id"]: float(v) for r, v in zip(rows, values, strict=True)}}
        diagnostic = backend.get("training_diagnostics", request)
        if "fit_runtime" in request and (backend["task"] != "bearing"
                or digest(request["fit_runtime"]) != digest([sys.version_info[:3], np.__version__])):
            raise ValueError("可复用权重的训练环境与请求不一致")
        if (request["kind"] != "continue" or request["epochs"] != 1
                or request["worker_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest()
                or request.get("parent_sha256", official_sha) != manifest["sha256"]
                or (depth and "stage" not in diagnostic)):
            raise ValueError("续训请求与实际选定父版本不符")
        train, explore = work / "train.jsonl", work / "explore.jsonl"
        _subset(archive, "train", request["fit_ids"], train)
        _subset(archive, "validation", diagnostic["exploration_ids"], explore)
        if inspect_resume:
            # 按官方训练器的实际加载路径核对每组张量，停在第一次预测或梯度计算之前。
            with np.load(parent, allow_pickle=False) as saved:
                tensors = {**model.params, **{"adam_first_" + k: v for k, v in model.first.items()},
                           **{"adam_second_" + k: v for k, v in model.second.items()},
                           "mean": model.mean, "scale": model.scale}
                if set(saved.files) != {*tensors, "metadata"} or json.loads(str(saved["metadata"])) != model.meta:
                    raise ValueError("载入的参数目录或训练状态不完整")
                for name, value in tensors.items():
                    if value.dtype != saved[name].dtype or not np.array_equal(value, saved[name]):
                        raise ValueError(f"载入时改变了检查点张量：{name}")
            return {"stage": diagnostic["stage"], "parent_sha256": manifest["sha256"], "depth": depth,
                    "epochs": model.meta["epochs"], "steps": model.meta["steps"],
                    "tensors": {k: sha256(v.tobytes()).hexdigest() for k, v in tensors.items()},
                    "train_sha256": sha256(train.read_bytes()).hexdigest(),
                    "explore_sha256": sha256(explore.read_bytes()).hexdigest()}
        output = work / "candidate"
        report = trainer.train_run(str(work / "baselines/configs" / f"{profile}.json"), train, explore, output,
                           resume=parent, epochs=1, stage_id=diagnostic.get("stage", "rul-one-epoch"))
        child = (output / "checkpoint.npz").read_bytes()
        trained = trainer.Model.load(output / "checkpoint.npz")
        steps = (len(request["fit_ids"]) + report["batch_size"] - 1) // report["batch_size"]
        if (report["parent_checkpoint_sha256"] != "sha256:" + sha256(content).hexdigest()
                or report["checkpoint_sha256"] != "sha256:" + sha256(child).hexdigest()
                or report["train_sha256"] != "sha256:" + sha256(train.read_bytes()).hexdigest()
                or report["validation_sha256"] != "sha256:" + sha256(explore.read_bytes()).hexdigest()
                or report["initial_epochs"] != model.meta["epochs"] or report["initial_steps"] != model.meta["steps"]
                or trained.meta["epochs"] != model.meta["epochs"] + 1 or trained.meta["steps"] != model.meta["steps"] + steps
                or not np.array_equal(model.mean, trained.mean) or not np.array_equal(model.scale, trained.scale)):
            raise ValueError("候选来源、优化器接续或标准化状态不匹配")
        atomic_json(folder / "training.json", report)
        (folder / "history.jsonl").write_bytes((output / "history.jsonl").read_bytes())
        return {"artifact": artifacts.create(request, lambda: child)}
