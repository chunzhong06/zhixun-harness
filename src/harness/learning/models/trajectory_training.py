"""单条寿命预测器的配对续训。设备及观测时钟只用于组织官方训练行。"""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from harness.core.contracts import digest
from harness.adapters.datasets.official_rul import ARCHIVE_SHA
from harness.learning.models.official_rul_cpu import RulPilotTools, _subset, _trainer, read_parent
from harness.adapters.storage.local import atomic_json


POLICIES = {"trajectory-order": "order", "trajectory-difference": "difference"}


def training_pairs(index):
    """每台设备前后半段按秩一一配对。奇数长度的中央观测仍参加绝对寿命训练。"""
    devices = defaultdict(list)
    for j, (unit, time, label) in enumerate(index):
        devices[unit].append((time, label, j))
    pairs, singles = [], []
    for seq in devices.values():
        seq.sort()
        if len({r[0] for r in seq}) != len(seq):
            raise ValueError("训练设备包含重复时刻")
        half = len(seq) // 2
        if len(seq) % 2:
            singles.append(seq[half][2])
        for a, b in zip(seq[:half], seq[len(seq)-half:], strict=True):
            if a[1] - b[1] != b[0] - a[0]:
                raise ValueError("寿命标签差与同机经过时间不一致")
            pairs.append((a[2], b[2]))
    return np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(singles, dtype=int)


def batch_delta(error, pair_starts, use_difference):
    """损失为逐条平方误差，加半倍成对差分平方和除以本批对数，均在原目标尺度标准化后计算。"""
    loss = np.mean(error**2)
    gradient = 2 * error / len(error)
    if use_difference and len(pair_starts):
        diff = error[pair_starts] - error[pair_starts + 1]
        loss += .5 * np.mean(diff**2)
        gradient[pair_starts] += diff / len(diff)
        gradient[pair_starts + 1] -= diff / len(diff)
    return float(loss), gradient[:, None]


def fit_epoch(model, x, y, pairs, singles, recipe):
    meta = model.meta
    if recipe not in POLICIES.values() or meta["task"] != "regression" or y.shape != (len(x),) or not len(x):
        raise ValueError("该配方仅支持寿命数据的一遍续训")
    x = model.normalized(x)
    target = (y-meta["target_mean"])/meta["target_scale"]
    rng = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epochs"]]))
    ordered = np.concatenate((pairs[rng.permutation(len(pairs))].ravel(), singles[rng.permutation(len(singles))]))
    if not np.array_equal(np.sort(ordered), np.arange(len(x))):
        raise ValueError("配对清单须恰好覆盖全部训练行一次")
    total_loss = 0.
    for start in range(0, len(x), meta["batch_size"]):
        ids = ordered[start:start+meta["batch_size"]]
        batch = x[ids]
        hidden, logits = model.forward(batch)
        # 若其它检查点的小批长度为奇数，只给当前小批内的完整配对加差分项。
        end = min(start+len(ids), 2*len(pairs))
        pair_starts = np.arange(start + start % 2, end-1, 2, dtype=int) - start
        loss, delta = batch_delta(logits[:, 0]-target[ids], pair_starts, recipe == "difference")
        total_loss += len(ids)*loss
        hidden_delta = (delta @ model.params["w2"].T) * (1-hidden**2)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                     "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
        norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("非有限梯度")
        meta["steps"] += 1
        for k, grad in gradients.items():
            grad *= min(1., 5./max(norm, 1e-12))
            model.first[k] = .9*model.first[k] + .1*grad
            model.second[k] = .999*model.second[k] + .001*grad**2
            a = model.first[k]/(1-.9**meta["steps"])
            b = model.second[k]/(1-.999**meta["steps"])
            model.params[k] -= meta["learning_rate"]*a/(np.sqrt(b)+1e-8)
    meta["epochs"] += 1
    model.validate()
    return total_loss/len(x)


class TrajectoryRulTools(RulPilotTools):
    def __init__(self, *args, recipe, **kwargs):
        super().__init__(*args, **kwargs)
        if recipe not in POLICIES.values():
            raise ValueError("未知的轨迹配方")
        self.recipe = recipe
        self.recipe_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        self.identity = digest([self.identity, self.recipe, self.recipe_sha])

    def _fit_request(self, task, action, rows, parent):
        if action.fit_manifest not in (None, "full"):
            raise ValueError("等样本轨迹对照只使用完整的固定训练清单")
        return {**super()._fit_request(task, action, rows, parent),
                "trajectory_recipe": self.recipe, "trajectory_sha256": self.recipe_sha}


def train_subset(trainer, profile, train, explore, work, parent, recipe, stage):
    """工作进程私有导入的原包训练器负责读输入、写标准权重和评价；仅本次 fit 更换目标。"""
    index, ids = [], []
    with train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            meta = row["meta"]
            unit, time = ((meta["bearing"], meta["snapshot"]) if profile == "B" else
                          ((meta["source_subset"], meta["unit"]), meta["cycle"]))
            index.append((unit, time, row["answer"]))
            ids.append(row["id"])
    pairs, singles = training_pairs(index)
    base = trainer.Model

    class PairedModel(base):
        def fit(self, x, y, epochs):
            if epochs != 1 or not np.array_equal(y, [r[2] for r in index]):
                raise ValueError("原包训练行序与配对索引不同")
            return [fit_epoch(self, x, y, pairs, singles, recipe)]

    output = work/"candidate"
    trainer.Model = PairedModel
    try:
        report = trainer.train_run(str(work/"baselines"/"configs"/(profile+".json")), train, explore,
                                   output, resume=parent, epochs=1, stage_id=stage)
    finally:
        trainer.Model = base
    report["trajectory"] = {"recipe": recipe, "pairs": len(pairs), "singletons": len(singles),
                            "pair_ids_sha256": digest([[ids[a], ids[b]] for a, b in pairs]),
                            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    return report


def execute_trajectory_job(folder, payload, artifacts):
    request, backend = payload["request"], payload["backend"]
    identity = sha256(Path(__file__).read_bytes()).hexdigest()
    archive = Path(backend["archive"])
    profile = {"bearing": "B", "engine": "C"}[backend["task"]]
    if (request["trajectory_sha256"] != identity or request["trajectory_recipe"] not in POLICIES.values()
            or request["worker_sha256"] != sha256(Path(__file__).with_name("official_rul_cpu.py").read_bytes()).hexdigest()
            or request["epochs"] != 1 or request["kind"] != "continue" or payload["rows"]
            or request["package_sha256"] != ARCHIVE_SHA[backend["task"]]
            or sha256(archive.read_bytes()).hexdigest() != request["package_sha256"]):
        raise ValueError("轨迹续训的来源、动作或程序版本不匹配")
    with TemporaryDirectory(dir=folder) as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), profile) as trainer:
        work = Path(tmp)
        source = {"package_sha256": request["package_sha256"], "checkpoint_sha256": sha256(z.read("checkpoint.npz")).hexdigest()}
        header, blob, _ = read_parent(artifacts, request["parent"], source)
        if request["parent_sha256"] != header["sha256"] or request["checkpoint_sha256"] != source["checkpoint_sha256"]:
            raise ValueError("父权重身份不匹配")
        if "fit_runtime" in request:
            import sys
            if backend["task"] != "bearing" or digest(request["fit_runtime"]) != digest([sys.version_info[:3], np.__version__]):
                raise ValueError("候选复用的数值环境不匹配")
        parent, train, explore = work/"parent.npz", work/"train.jsonl", work/"explore.jsonl"
        parent.write_bytes(blob)
        diagnostic = backend.get("training_diagnostics", request)
        _subset(z, "train", request["fit_ids"], train)
        _subset(z, "validation", diagnostic["exploration_ids"], explore)
        report = train_subset(trainer, profile, train, explore, work, parent, request["trajectory_recipe"], diagnostic["stage"])
        child_path = work/"candidate"/"checkpoint.npz"
        initial, child = trainer.Model.load(parent), trainer.Model.load(child_path)
        content = child_path.read_bytes()
        if (child.meta["steps"] != initial.meta["steps"] + math.ceil(len(request["fit_ids"])/initial.meta["batch_size"])
                or child.meta["epochs"] != initial.meta["epochs"]+1
                or not np.array_equal(child.mean, initial.mean) or not np.array_equal(child.scale, initial.scale)
                or report["parent_checkpoint_sha256"] != "sha256:"+header["sha256"]
                or report["checkpoint_sha256"] != "sha256:"+sha256(content).hexdigest()):
            raise ValueError("续训权重的起点、成本或尺度不匹配")
        atomic_json(folder/"training.json", report)
        (folder/"history.jsonl").write_bytes((work/"candidate"/"history.jsonl").read_bytes())
        return {"artifact": artifacts.create(request, lambda: content)}
