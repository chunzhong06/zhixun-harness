"""在同一官方 Adam 步上只应用输出层更新，复用已完成的全模型步骤，独立评价真实预测。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.compare_bearing_fit import EXPECTED_IDS, OUTPUT as SOURCE, freeze, load
from engineering.cli.study.run_rul_maintenance import PACKAGES, describe
from harness.contracts import digest
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _subset, _trainer, read_parent
from harness.runtime.storage import LocalStore, atomic_json

OUTPUT = SOURCE.parent / "readout-step"


def read_models(work, trainer, package, checksum, previous, decision):
    store = ModelArtifacts(SOURCE / "models")
    origin = {"package_sha256": checksum, "checkpoint_sha256": package.checkpoint_sha}
    manifest, content, depth = read_parent(store, decision["candidate"], origin)
    request = manifest["request"]
    if (depth != 1 or request["parent"] != previous["parent"]
            or request["fit_ids"] != previous["fit_ids"][1]):
        raise ValueError("全模型候选与已冻结的父权重、训练 ID 不一致")
    parent_path, full_path = work / "parent.npz", work / "full.npz"
    parent_path.write_bytes(package.checkpoint)
    full_path.write_bytes(content)
    return trainer.Model.load(parent_path), trainer.Model.load(full_path), manifest["sha256"]


def readout_checkpoint(work, original, updated, trainer):
    """单步中所有梯度来自同一个父版本。保留原步的全梯度裁剪，复用输出层更新和对应的 Adam 记忆。"""
    import numpy as np

    model = trainer.Model.load(work / "parent.npz")
    for field in ("steps", "epochs"):
        if updated.meta[field] != model.meta[field] + 1:
            raise ValueError("多步训练不满足此处的单步分解条件")
        model.meta[field] = updated.meta[field]
    if (model.meta != updated.meta or not np.array_equal(model.mean, updated.mean)
            or not np.array_equal(model.scale, updated.scale)):
        raise ValueError("匹配组改变了其他训练参数或尺度")
    for key in ("w2", "b2"):
        for group in ("params", "first", "second"):
            getattr(model, group)[key] = getattr(updated, group)[key].copy()
    path = work / "readout.npz"
    model.save(path)
    reloaded = trainer.Model.load(path)
    for key in model.params:
        expected = original if key in ("w1", "b1") else updated
        for group in ("params", "first", "second"):
            if not np.array_equal(getattr(reloaded, group)[key], getattr(expected, group)[key]):
                raise ValueError(f"重载后参数或优化器记忆不一致：{group}/{key}")
    return reloaded, sha256(path.read_bytes()).hexdigest()


def signal_check(package, groups, original):
    """训练库内每台设备最多一票；未知记录只用振动值选参照。真值在投票结束后统计。"""
    import numpy as np

    train = package.train
    x = np.asarray([r.values for r in train])
    normalized = original.normalized(x)
    hidden, _ = original.forward(normalized)
    need_raise = np.asarray([r.label for r in train]) > original.predict(x)
    units = defaultdict(list)
    for i, row in enumerate(train):
        units[row.unit_id].append(i)
    indices = [np.asarray(members) for _, members in sorted(units.items())]
    output = {}
    for name in ("early", "late", "adopt", "serve"):
        rows = groups[name]
        values = np.asarray([r.values for r in rows])
        raw = original.normalized(values)
        embedding, _ = original.forward(raw)
        truth = np.asarray([r.label for r in rows]) > original.predict(values)
        output[name] = {}
        for space, library, queries in (("normalized", normalized, raw), ("hidden", hidden, embedding)):
            votes = []
            for query in queries:
                distance = np.sum((library - query) ** 2, axis=1)
                nearby = [members[np.argmin(distance[members])] for members in indices]
                three = sorted(nearby, key=lambda i: (distance[i], train[i].unit_id))[:3]
                votes.append(sum(bool(need_raise[i]) for i in three) >= 2)
            raise_flag = np.asarray(votes)
            output[name][space] = {
                "rows": len(rows), "need_raise": int(truth.sum()),
                "correct_raise": int((raise_flag & truth).sum()),
                "wrong_raise": int((raise_flag & ~truth).sum())}
    return output


def run(output, execute):
    import numpy as np

    name, checksum, reader = PACKAGES["bearing"]
    archive = ROOT / "data/official/bearing" / name
    if sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError("训练包版本已变")
    previous, decision = load(SOURCE / "protocol.json"), load(SOURCE / "decision.json")
    if previous["archive"] != checksum or decision["protocol"] != digest(previous):
        raise ValueError("原训练与决定不匹配")
    fit = previous["fit_ids"][1]
    if len(fit) != 72 or sha256(("\n".join(fit) + "\n").encode()).hexdigest() != EXPECTED_IDS[1]:
        raise ValueError("与上轮按轨迹长度分配的 72 条记录不一致")
    report_path, = (SOURCE / "jobs").glob("*/training.json")
    training = load(report_path)
    if training["final_steps"] - training["initial_steps"] != 1 or training["batch_size"] < len(fit):
        raise ValueError("原训练回执并非一个批次一步")
    protocol = {"source_protocol": digest(previous), "reference_result": sha256((SOURCE / "result.json").read_bytes()).hexdigest(),
                "source_training": sha256(report_path.read_bytes()).hexdigest(),
                "fit_ids_sha256": EXPECTED_IDS[1], "fit_rows": len(fit), "groups": previous["groups"],
                "updated": ["w2", "b2"], "frozen": ["w1", "b1"], "clipping": "same full gradient norm as paired step",
                "select": "lower early RMSE and underestimation vs official parent; late RMSE, late positive bias and short confirmation RMSE no worse; report service after selection",
                "diagnosis_if_rejected": "nearest reading per training device, three nearest different devices vote on parent underprediction; normalized features and hidden features; descriptive only"}
    freeze(output / "protocol.json", protocol)
    if not execute:
        return
    if (output / "result.json").is_file():
        result = load(output / "result.json")
        if result["protocol"] != digest(protocol) or result["decision"] != digest(load(output / "decision.json")):
            raise ValueError("已保存的结果与输入或决定不符")
        return
    package = reader(archive)
    by_id = {r.sample_id: r for r in package.validation}
    groups = {k: [by_id[i] for i in ids] for k, ids in previous["groups"].items()}
    reference = load(SOURCE / "result.json")["predictions"]
    with ZipFile(archive) as z, TemporaryDirectory(dir=output) as tmp, _trainer(z, Path(tmp), "B") as trainer:
        work = Path(tmp)
        _subset(z, "train", fit, work / "train.jsonl")
        if "sha256:" + sha256((work / "train.jsonl").read_bytes()).hexdigest() != training["train_sha256"]:
            raise ValueError("训练文件原始字节不匹配")
        original, full, full_sha = read_models(work, trainer, package, checksum, previous, decision)
        if "sha256:" + full_sha != training["checkpoint_sha256"]:
            raise ValueError("旧更新与训练回执不符")
        started = process_time()
        model, candidate_sha = readout_checkpoint(work, original, full, trainer)
        construction_seconds = process_time() - started
        scores, predictions = {}, {}
        def measure(name):
            rows = groups[name]
            values = np.asarray([r.values for r in rows])
            outputs = {"original": original.predict(values), "length": full.predict(values),
                       "readout": model.predict(values)}
            # 在同一批测量上独立重算输出层变化，再经官方非负截断，检查重载后实际预测。
            h, logits = original.forward(original.normalized(values))
            delta = h @ (full.params["w2"] - original.params["w2"]) + full.params["b2"] - original.params["b2"]
            expected = np.maximum(0, (logits + delta)[:, 0] * original.meta["target_scale"] + original.meta["target_mean"])
            np.testing.assert_allclose(outputs["readout"], expected, rtol=1e-12, atol=1e-9)
            for arm in ("original", "length"):
                np.testing.assert_allclose(outputs[arm], [reference[r.sample_id][arm] for r in rows], rtol=1e-12, atol=1e-9)
            scores[name] = {}
            for arm, result in outputs.items():
                pairs = {r.sample_id: float(v) for r, v in zip(rows, result, strict=True)}
                scores[name][arm] = describe(package.task, rows, pairs)
                if arm == "readout":
                    predictions.update(pairs)
        for group in ("early", "transition", "late", "adopt"):
            measure(group)
        early, late, short = scores["early"], scores["late"], scores["adopt"]
        checks = {"early_rmse": early["readout"]["rmse"] < early["original"]["rmse"],
                  "early_underestimate": max(0, -early["readout"]["mean_signed_error"]) < max(0, -early["original"]["mean_signed_error"]),
                  "late_rmse": late["readout"]["rmse"] <= late["original"]["rmse"],
                  "late_overestimate": max(0, late["readout"]["mean_signed_error"]) <= max(0, late["original"]["mean_signed_error"]),
                  "short_rmse": short["readout"]["rmse"] <= short["original"]["rmse"]}
        selected = "readout" if all(checks.values()) else "original"
        receipt = {"protocol": digest(protocol), "model_sha256": candidate_sha,
                   "checks": checks, "selected": selected, "development": dict(scores)}
        freeze(output / "decision.json", receipt)
        measure("serve")
        diagnostic = signal_check(package, groups, original) if selected == "original" else None
        atomic_json(output / "result.json", {"protocol": digest(protocol), "decision": digest(receipt),
                    "scores": scores, "selected": selected, "selected_service": scores["serve"][selected],
                    "predictions": predictions, "new_gradient_steps": 0, "reused_gradient_steps": 1,
                    "checkpoint_build_cpu_seconds": construction_seconds or None, "signal_check": diagnostic})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="重建单步输出层候选，用官方模型作本地预测")
    with LocalStore(OUTPUT).exclusive():
        run(OUTPUT, parser.parse_args().evaluate)
