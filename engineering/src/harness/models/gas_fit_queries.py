"""把已提出的气体筛选条件变成官方训练行。验证数据只用于回测或确认。"""
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import operator
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from harness.contracts import ActionPlan, digest
from harness.data.gas_input import feature_schema
from harness.data.official_gas import LABELS
from harness.models.official_gas_cpu import OfficialGasTools, PACKAGE_SHA, PACKAGE_FILES
from harness.runtime.storage import atomic_json


OPERATORS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
FEATURES = tuple(field["name"] for field in feature_schema())


def observation_key(package, batch, baseline):
    """防止别的轮次或分支的提案混入。确认行和未来服务标签都不参与此身份。"""
    explore = [r for r in batch.samples if r.role == "explore" and r.available_at == batch.ordinal]
    predictions = baseline["predictions"]
    if not explore or {r.sample_id for r in explore} != set(predictions):
        raise ValueError("诊断观察与在用版的探索预测不匹配")
    observations = [(r.sample_id, r.revision, r.values, r.label, predictions[r.sample_id]) for r in explore]
    return digest([package.train_sha256, batch.batch_id, baseline["model_id"], observations])


@dataclass(frozen=True)
class RemovalQuery:
    """条件为真时剔除该行，训练使用其补集，与作者的 queries_remove 方向一致。"""

    feature: str
    op: str
    threshold: float

    def __post_init__(self):
        if self.feature not in FEATURES or self.op not in OPERATORS:
            raise ValueError("筛选仅支持已登记的测量字段与四种大小比较")
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold):
            raise ValueError("阈值须为有限测量值")

    @property
    def name(self):
        return "exclude-" + digest(asdict(self))

    def select(self, rows):
        column = FEATURES.index(self.feature)
        return tuple(r for r in rows if not OPERATORS[self.op](r.values[column], self.threshold))


class GasQueryTools(OfficialGasTools):
    """按批锁定已提出的条件。历史作业继续使用旧适配器与原来的恢复身份。"""

    def __init__(self, path, package, batches, runner, proposals):
        super().__init__(path, package, batches, runner)
        self.capabilities["continuation_epochs"] = [1]
        if not isinstance(proposals, dict) or set(proposals) - set(self.explore):
            raise ValueError("清单须归属当前协议中已知的批次")
        self.queries = {}
        self.by_batch = {}
        self.selected = {}
        self.provenance = {}
        self.dispositions = {}
        for bid, record in proposals.items():
            if (not isinstance(record, dict) or set(record) != {"source", "observation_key", "parent_sha256", "conditions"}
                    or not isinstance(record["source"], str) or not record["source"].strip()
                    or not isinstance(record["conditions"], list)
                    or not isinstance(record["parent_sha256"], str) or not record["parent_sha256"].startswith("sha256:")
                    or not isinstance(record["observation_key"], str) or len(record["observation_key"]) != 64):
                raise ValueError("每批提案须提供来源、父权重、当时的观察身份和有序条件")
            self.provenance[bid] = {key: record[key] for key in ("source", "observation_key", "parent_sha256")}
            choices, outcomes = [], []
            for index, condition in enumerate(record["conditions"]):
                try:
                    query = RemovalQuery(**condition)
                except (TypeError, ValueError, OverflowError):
                    outcomes.append({"index": index, "status": "unsupported_condition"})
                    continue
                kept = query.select(package.train)
                if not kept or len(kept) == len(package.train):
                    outcomes.append({"index": index, "status": "empty" if not kept else "unchanged"})
                    continue
                ids = tuple(row.sample_id for row in kept)
                same = next((name for name in choices if self.selected[name] == ids), None)
                if same is not None:
                    outcomes.append({"index": index, "status": "duplicate_rows", "manifest": same})
                    continue
                self.selected[query.name] = ids
                self.queries[query.name] = query
                choices.append(query.name)
                outcomes.append({"index": index, "status": "eligible", "manifest": query.name})
            self.by_batch[bid] = choices
            self.dispositions[bid] = outcomes
        self.identity = digest([self.identity, sha256(Path(__file__).read_bytes()).hexdigest(), proposals])

    def fit_manifest_context(self, batch, baseline):
        origin = self.provenance.get(batch.batch_id)
        if origin and (not baseline or origin["observation_key"] != observation_key(self.package, batch, baseline)):
            raise ValueError("筛选建议来自另一批观察或不同模型的预测，不能据此回放")
        names = self.by_batch.get(batch.batch_id, [])
        counts = Counter(r.label for r in self.package.train)
        coverage = {}
        for name in names:
            kept = Counter(self.train[sid].label for sid in self.selected[name])
            coverage[name] = {label: [counts[i], kept[i]] for i, label in enumerate(LABELS, 1)}
        return {"names": names, "origin": origin, "proposal_dispositions": self.dispositions.get(batch.batch_id, []),
                "conditions": {name: asdict(self.queries[name]) for name in names},
                "training_class_counts_before_after": coverage}

    def select_fit_rows(self, name, eligible):
        # 以原文件顺序返回。相同编号但版本、测量或用途变了，也不能冒充训练记录。
        if name not in self.selected:
            raise ValueError("未登记的气体筛选条件")
        supplied = {r.sample_id: r for r in eligible}
        if len(supplied) != len(eligible) or supplied != self.train:
            raise ValueError("气体选样需要原始训练资料，且每条只出现一次")
        return tuple(supplied[sid] for sid in self.selected[name])

    def _build_request(self, task, batch_id, action, rows, initialize, parent):
        if initialize:
            return super()._build_request(task, batch_id, action, rows, True, parent)
        name = action.fit_manifest
        if name not in self.by_batch.get(batch_id, ()) or action.epochs != 1 or action.gas_pair or action.focus_label is not None:
            raise ValueError("该批只接受已登记的筛选清单和一次续训")
        expected = tuple(self.train[sid] for sid in self.selected[name])
        if tuple(rows) != expected:
            raise ValueError("续训输入与冻结的原文件行不符")
        request = super()._build_request(task, batch_id, action, self.package.train, False, parent)
        return {**request, "gas_fit_query": asdict(self.queries[name]), "fit_ids_digest": digest(self.selected[name]),
                "proposal_origin": self.provenance[batch_id],
                "implementation": self.identity}


class GasQueryReplayPlanner:
    """顺序执行外部已冻结的建议；是否采用由公共的探索评分与独立确认决定。"""

    identity = "gas-query-replay-v1"

    def propose(self, context):
        tried = {t["action"] for t in context["trials"]}
        ref = next(f["id"] for f in context["facts"]["evidence"] if f["kind"] == "performance")
        catalog = context["fit_manifests"]
        for name in catalog["names"]:
            condition = catalog["conditions"][name]
            counts = catalog["training_class_counts_before_after"][name].values()
            removed = sum(before - after for before, after in counts)
            reason = (f"回测 {condition['feature']} {condition['op']} {condition['threshold']} 这项剔除建议。"
                      f"该条件从官方训练清单中移除 {removed} 条，使用剩余测量从在用权重续训一遍。")
            action = ActionPlan("recent", epochs=1, fit_manifest=name, evidence_refs=(ref,),
                                reason=reason)
            if action.key not in tried:
                return {"actions": [action.to_record()], "stop": False}
        return {"actions": [], "stop": True}


def execute_query_job(folder, payload, artifacts, *, inspect_only=False):
    """核对父权重与真实输入后才续训。只读核验走同一入口，止于官方数据读取器。"""
    import sys
    import numpy as np
    request = payload["request"]
    path = Path(payload["backend"]["archive"])
    if (request["kind"] != "continue" or request["epochs"] != 1 or request["package_sha256"] != PACKAGE_SHA
            or sha256(path.read_bytes()).hexdigest() != PACKAGE_SHA):
        raise ValueError("筛选续训工具、预算或官方包不匹配")
    query = RemovalQuery(**request["gas_fit_query"])
    with TemporaryDirectory(dir=folder) as tmp, ZipFile(path) as archive:
        tmp = Path(tmp)
        source = archive.read("public/train.jsonl")
        if "sha256:" + sha256(source).hexdigest() != request["train_sha256"]:
            raise ValueError("训练原件不符")
        column = FEATURES.index(query.feature)
        lines, ids = [], []
        for line in source.splitlines(keepends=True):
            raw = json.loads(line)
            if not OPERATORS[query.op](raw["inputs"]["features"][column], query.threshold):
                lines.append(line)
                ids.append(raw["id"])
        if not ids or len(lines) == len(source.splitlines()) or digest(ids) != request["fit_ids_digest"]:
            raise ValueError("工作进程得到的清单与已登记的训练编号不符")
        train, validation, parent = tmp / "train.jsonl", tmp / "explore.jsonl", tmp / "parent.npz"
        train.write_bytes(b"".join(lines))
        validation.write_bytes(payload["rows"].encode("utf-8"))
        if sha256(validation.read_bytes()).hexdigest() != request["explore_sha256"]:
            raise ValueError("探索输入不符")
        manifest, content = artifacts.read(request["parent"])
        if (manifest["request"].get("package_sha256") != PACKAGE_SHA
                or "sha256:" + sha256(content).hexdigest() != request["proposal_origin"]["parent_sha256"]):
            raise ValueError("父版本与官方气体包不兼容")
        parent.write_bytes(content)
        for name in PACKAGE_FILES:
            target = tmp / "baselines" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read("baselines/" + name))
        # 运行于独占的短命工作进程；只引入已核对原包的代码。
        previous_modules = set(sys.modules)
        sys.path.insert(0, str(tmp))
        try:
            from baselines import train as trainer
            from baselines.model import Model, checkpoint_digest
            if not Path(trainer.__file__).resolve().is_relative_to(tmp):
                raise ValueError("官方工具需在独立工作进程中加载")
            fit = trainer.read_dataset(train, "classification", list(LABELS))
            explore = trainer.read_dataset(validation, "classification", list(LABELS), fit["paths"])
            original = Model.load(parent)
            if (fit["ids"] != ids or set(ids) & set(explore["ids"]) or original.meta["paths"] != fit["paths"]
                    or original.meta["profile"] != "A" or original.meta["classes"] != list(LABELS)
                    or "gas_pair_patch" in original.meta):
                raise ValueError("训练、回测资料或父模型与筛选续训不兼容")
            provenance = {"query": asdict(query), "proposal_origin": request["proposal_origin"],
                          "source_sha256": request["train_sha256"],
                          "fit_ids_digest": digest(ids), "train_sha256": fit["sha256"],
                          "training_rows": len(ids), "exploration_rows": len(explore["ids"]),
                          "validation_sha256": explore["sha256"], "parent_checkpoint_sha256": checkpoint_digest(parent)}
            if inspect_only:
                return provenance
            output = tmp / "candidate"
            report = trainer.train_run("A", train, validation, output, resume=parent, epochs=1, stage_id=request["batch_id"])
            checkpoint = output / "checkpoint.npz"
            model = Model.load(checkpoint)
            steps = (len(ids) + report["batch_size"] - 1) // report["batch_size"]
            if (any(report[k] != provenance[k] for k in ("train_sha256", "validation_sha256", "parent_checkpoint_sha256"))
                    or report["checkpoint_sha256"] != checkpoint_digest(checkpoint)
                    or model.meta["steps"] != original.meta["steps"] + steps
                    or model.meta["epochs"] != original.meta["epochs"] + 1
                    or not np.array_equal(original.mean, model.mean) or not np.array_equal(original.scale, model.scale)):
                raise ValueError("候选输入、步数、标准化或权重不符")
            atomic_json(folder / "training.json", {**report, "gas_fit_query": provenance})
            (folder / "history.jsonl").write_bytes((output / "history.jsonl").read_bytes())
            return {"artifact": artifacts.create(request, checkpoint.read_bytes)}
        finally:
            sys.path.remove(str(tmp))
            for name in set(sys.modules) - previous_modules:
                if name == "baselines" or name.startswith("baselines."):
                    del sys.modules[name]
