"""官方气体包的只读边界。保留原文件顺序、公开用途与类别编码。"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZipFile

from harness.core.contracts import Sample, TaskSpec
from harness.adapters.datasets.gas_input import feature_schema


LABELS = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")


def prediction_request(values):
    """与随包 predict.py 的单行接口一致，追溯字段只留在 Harness 内部。"""
    if len(values) != 128 or not all(type(x) in (int, float) and math.isfinite(x) for x in values):
        raise ValueError("气体测量需要128个有限数值")
    return {"task": "classification", "inputs": {"features": list(values)}}


def prediction_label(response):
    """按检查点的类别顺序取最大分数，并列时沿用官方 argmax 的首项。"""
    scores = response["outputs"]
    if len(scores) != len(LABELS) or {s["label"] for s in scores} != set(LABELS):
        raise ValueError("气体输出须覆盖六类且各出现一次")
    values = {s["label"]: s["score"] for s in scores}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError("类别分数应为有限数值")
    return 1 + max(range(len(LABELS)), key=lambda i: values[LABELS[i]])


def exploration_jsonl(rows):
    """生成随包训练工具可读的开发验证子集。采用确认和待预测行不进入此文件。"""
    lines = []
    for row in rows:
        if row.role != "explore" or type(row.label) is not int or not 1 <= row.label <= len(LABELS):
            raise ValueError("工具验证子集只接收已公开答案的探索行")
        lines.append(json.dumps({"id": row.sample_id, "inputs": prediction_request(row.values)["inputs"],
                                 "answer": LABELS[row.label-1]}, separators=(",", ":"), allow_nan=False))
    if not lines:
        raise ValueError("缺少用于续训比较的探索行")
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True)
class OfficialGasPackage:
    task: TaskSpec
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    checkpoint: bytes
    checkpoint_sha256: str
    package_id: str
    train_sha256: str
    validation_sha256: str


def _split_rows(raw, name, role, report, seen):
    rows = []
    for line in raw.splitlines():
        record = json.loads(line)
        sid, inputs = record["id"], record["inputs"]
        if not isinstance(sid, str) or not sid or sid in seen or set(inputs) != {"features"}:
            raise ValueError("气体样本身份重复或输入字段不符")
        seen.add(sid)
        values = prediction_request(inputs["features"])["inputs"]["features"]
        rows.append(Sample(sid, 1, tuple(values), LABELS.index(record["answer"]) + 1,
                           role, f"source-{record['meta']['source_batch']}", 0))
    if len(rows) != report[f"{name}_rows"]:
        raise ValueError(f"{name} 行数与随包记录不符")
    return tuple(rows)


def read_gas_training(path):
    """推理阶段只打开源端训练记录与训练行，不读取公开验证样本。"""
    with ZipFile(path) as archive:
        report = json.loads(archive.read("training.json"))
        raw = archive.read("public/train.jsonl")
        if report["profile"] != "A" or "sha256:" + sha256(raw).hexdigest() != report["train_sha256"]:
            raise ValueError("气体源训练清单身份不符")
        return _split_rows(raw, "train", "fit", report, set())


def read_official_gas(path: Path) -> OfficialGasPackage:
    """核对随包记录后解析两份独立清单；验证行初始只用于开发评价。"""
    task = TaskSpec("official-gas-A", tuple(f["name"] for f in feature_schema()),
                    tuple(range(1, 7)), metric="macro_f1")
    with ZipFile(path) as archive:
        info = json.loads(archive.read("baseline-info.json"))
        report = json.loads(archive.read("training.json"))
        config = json.loads(archive.read("baselines/configs/A.json"))
        if info["dataset_key"] != "A" or report["profile"] != "A" or tuple(config["classes"]) != LABELS:
            raise ValueError("训练包与气体任务或类别顺序不符")

        def checked(member, key):
            data = archive.read(member)
            if "sha256:" + sha256(data).hexdigest() != report[key]:
                raise ValueError(f"随包文件摘要不符：{member}")
            return data

        seen = set()
        def read_split(name, role):
            raw = checked(f"public/{name}.jsonl", f"{name}_sha256")
            return _split_rows(raw, name, role, report, seen)

        checkpoint = checked("checkpoint.npz", "checkpoint_sha256")
        if report["checkpoint_sha256"] != info["checkpoint_sha256"]:
            raise ValueError("基线说明与训练记录的父检查点不一致")
        return OfficialGasPackage(task, read_split("train", "fit"), read_split("validation", "explore"),
                                  checkpoint, report["checkpoint_sha256"], info["id"],
                                  report["train_sha256"], report["validation_sha256"])
