"""读取UCI 224气体原包并保留来源；比赛用途和标签释放由显式映射决定。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import zipfile

from harness.core.contracts import BatchContext, Sample, TaskSpec, digest


SOURCE_URL = "https://archive.ics.uci.edu/dataset/224/gas"
REGISTERED_SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
CLASS_NAMES = {1: "乙醇", 2: "乙烯", 3: "氨", 4: "乙醛", 5: "丙酮", 6: "甲苯"}
FEATURE_KINDS = ("delta_r", "normalized_delta_r", "ema_rise_0.001", "ema_rise_0.01", "ema_rise_0.1",
                 "ema_decay_0.001", "ema_decay_0.01", "ema_decay_0.1")
# 来源页给出批次对应的研究月份，但没有逐行时间戳或独立设备身份。
MONTHS = {1: [1, 2], 2: [3, 4, 8, 9, 10], 3: [11, 12, 13], 4: [14, 15], 5: [16],
          6: [17, 18, 19, 20], 7: [21], 8: [22, 23], 9: [24, 30], 10: [36]}


def feature_schema():
    """按原文8项特征×16个传感器的顺序命名；传感器编号不充当样本独立单位。"""
    return [{"index": (sensor - 1) * 8 + position + 1,
             "name": f"sensor_{sensor:02d}_{kind}", "sensor": sensor, "kind": kind}
            for sensor in range(1, 17) for position, kind in enumerate(FEATURE_KINDS)]


def gas_task():
    """公开数据的读取目标；accuracy是现有内部指标，不指定比赛正式计分。"""
    return TaskSpec("uci-gas-drift", tuple(f["name"] for f in feature_schema()), tuple(CLASS_NAMES),
                    diagnostic_features=tuple(f"sensor_{i:02d}_delta_r" for i in range(1, 17)))


@dataclass(frozen=True)
class GasRecord:
    """原始观测，没有fit/explore/adopt用途或可见时刻；源行位置只用于定位。"""
    sample_id: str
    source_line: int
    label: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class GasSourceBatch:
    batch_number: int
    member: str
    archive_sha256: str
    member_sha256: str
    records: tuple[GasRecord, ...]


def _parse_line(line, member, number, archive_hash):
    fields = line.split()
    if len(fields) != 129 or fields[0] not in {str(c) for c in CLASS_NAMES}:
        raise ValueError(f"{member}:{number}需要已声明气体类别和完整128项特征")
    values = {}
    for field in fields[1:]:
        index, separator, value = field.partition(":")
        if not separator or not re.fullmatch(r"[1-9][0-9]*", index):
            raise ValueError(f"{member}:{number}特征索引格式错误")
        index = int(index)
        if not 1 <= index <= 128 or index in values:
            raise ValueError(f"{member}:{number}特征索引重复或超界")
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{member}:{number}特征不是数值") from None
        if not math.isfinite(value):
            raise ValueError(f"{member}:{number}特征必须为有限数值")
        values[index] = value
    # 完整源包版本+成员+原行号定义本地身份；相同读数仍保留各自记录。
    sample_id = f"uci224:{archive_hash}:{member}:{number}"
    return GasRecord(sample_id, number, int(fields[0]), tuple(values[i] for i in range(1, 129)))


def read_gas_batches(path: Path, *, expected_sha256: str, batch_number: int | None = None):
    """逐批只读ZIP，不解压落盘；可只解析选定批次，输出保持来源顺序。

    调用者须声明来源哈希。原件改变时先核对新版本，不能静默重用原行身份。
    原包的十个批次是数据组织信息，尚未被指定为比赛发放事件。
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("需要明确的来源SHA256")
    if batch_number is not None and (type(batch_number) is not int or batch_number not in MONTHS):
        raise ValueError("原始批次号须为1至10")
    with Path(path).open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if archive_hash != expected_sha256:
            raise ValueError("气体数据包与声明来源不同")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = [i.filename for i in archive.infolist() if not i.is_dir()]
            expected_members = {f"Dataset/batch{b}.dat" for b in MONTHS}
            if len(members) != len(set(members)) or set(members) != expected_members:
                raise ValueError("气体原包须包含互不重复的十个批次文件")
            for batch in ([batch_number] if batch_number is not None else MONTHS):
                member = f"Dataset/batch{batch}.dat"
                records, checksum = [], hashlib.sha256()
                with archive.open(member) as source:
                    for number, raw in enumerate(source, start=1):
                        checksum.update(raw)
                        line = raw.decode("utf-8").strip()
                        if line:
                            records.append(_parse_line(line, member, number, archive_hash))
                if not records:
                    raise ValueError(f"{member}没有观测记录")
                yield GasSourceBatch(batch, member, archive_hash, checksum.hexdigest(), tuple(records))


def bind_gas_batch(source: GasSourceBatch, *, batch_id: str, ordinal: int, assignments: dict):
    """按外部明确的用途/可见批次映射到公共记录；本函数不产生划分方案。

    assignments以原sample_id为键，每项须声明role和available_at。允许选择
    当前获准的子集；serve记录移除答案。group只记来源批次，unit_id和逐行
    时间保持未知。revision=1表示这份哈希固定的原件，非官方修订协议。
    """
    if not isinstance(batch_id, str) or not batch_id or type(ordinal) is not int or ordinal < 0:
        raise ValueError("需要明确的运行批次身份与序号")
    by_id = {r.sample_id: r for r in source.records}
    if not isinstance(assignments, dict) or not assignments or not set(assignments) <= by_id.keys():
        raise ValueError("用途映射需要非空且属于当前原批次的身份")
    task, samples = gas_task(), []
    for record in source.records:
        if record.sample_id not in assignments:
            continue
        assignment = assignments[record.sample_id]
        if (not isinstance(assignment, dict) or set(assignment) != {"role", "available_at"}
                or type(assignment["available_at"]) is not int):
            raise ValueError("用途映射须明确role和整数available_at")
        role = assignment["role"]
        if not isinstance(role, str) or role not in {"fit", "explore", "adopt", "serve"}:
            raise ValueError("用途映射含未知role")
        sample = Sample(record.sample_id, 1, record.values, None if role == "serve" else record.label,
                        role, f"source-batch-{source.batch_number}", assignment["available_at"])
        sample.validate(task, ordinal)
        samples.append(sample)
    return BatchContext(task.task_id, batch_id, ordinal, tuple(samples))


def inspect_gas(path: Path, *, expected_sha256=REGISTERED_SHA256):
    """检查全部原行并生成可核对的结构报告；不分配用途，不创建训练/计划任务。"""
    summaries = []
    for batch in read_gas_batches(path, expected_sha256=expected_sha256):
        counts = Counter(r.label for r in batch.records)
        canonical = hashlib.sha256()
        for record in batch.records:
            canonical.update(digest(asdict(record)).encode("ascii"))
        first, last = batch.records[0], batch.records[-1]
        summaries.append({"batch": batch.batch_number, "member": batch.member,
            "member_sha256": batch.member_sha256, "rows": len(batch.records),
            "class_counts": {str(c): counts[c] for c in CLASS_NAMES},
            "missing_classes": [c for c in CLASS_NAMES if not counts[c]],
            "source_months": MONTHS[batch.batch_number], "record_digest": canonical.hexdigest(),
            "first_sample_id": first.sample_id, "last_sample_id": last.sample_id,
            "preview": {"source_line": first.source_line, "label": first.label,
                        "values_head": list(first.values[:8]), "feature_count": len(first.values)}})
    return {"source": str(Path(path).resolve()), "source_url": SOURCE_URL, "sha256": expected_sha256,
            "reader_version": 1, "task": asdict(gas_task()), "class_names": CLASS_NAMES,
            "feature_schema": feature_schema(), "batches": summaries,
            "total_rows": sum(b["rows"] for b in summaries), "inspection_only": True,
            "model_training": False, "model_called": False, "prediction_scores_computed": False,
            "runtime_roles_assigned": False, "official_layout_confirmed": False,
            "identity_rule": "固定原包SHA256+成员名+原行号；重复读数不合并",
            "unknown_fields": ["逐行时间", "独立设备或测量组", "比赛批次与标签释放", "训练/探索/采用划分", "正式计分和输出"]}
