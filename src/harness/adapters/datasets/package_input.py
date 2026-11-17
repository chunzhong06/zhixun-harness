"""用显式字段映射读取表格数据；清单格式属于本项目，不假定官方A包布局。"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from harness.core.contracts import BatchContext, Sample, TaskSpec, digest
from harness.adapters.datasets.selection import ingest


@dataclass(frozen=True)
class ImportedPackage:
    task: TaskSpec
    batches: tuple[BatchContext, ...]
    origin: str
    source_digest: str
    inspection: dict


def _integer(value) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError("身份版本或可见批次必须是整数")


def _local_file(root: Path, relative: str) -> Path:
    """解析清单相对路径，保持输入文件位于所选资料目录内。"""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("资料清单需要相对路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("资料文件不存在或超出清单目录")
    return path


def _rows(path: Path, file_format: str, encoding: str):
    if file_format == "csv":
        with path.open(encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("CSV表头为空或存在重复列")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("CSV行与表头列数不一致")
                yield row
    elif file_format == "jsonl":
        with path.open(encoding=encoding) as f:
            for line in f:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL每行必须是一个样本对象")
                yield value
    else:
        raise ValueError("当前表格接入口支持csv或jsonl，其他格式需相应解码器")


def read_package(manifest_path: Path, *, batch_index: int | None = None) -> ImportedPackage:
    """预检整包或只读取指定批次；逐批调用不预读后续文件。

    清单显式给出样本身份、标签映射、用途及首次可用批次；不从行号、
    当前文件位置或目录名称推断这些含义。返回后才可交给公共控制器。
    """
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "harness_table_manifest":
        raise ValueError("不支持的数据映射清单")
    origin = manifest.get("origin")
    if origin not in {"interface_fixture", "user_supplied"}:
        raise ValueError("必须明确资料是接口样例还是用户提供的实际资料")
    task = TaskSpec.parse(manifest["task"])
    columns = manifest["columns"]
    required = {"sample_id", "revision", "features", "label", "role", "group", "available_at"}
    optional={"unit_id","time_start","time_end"}
    if not required <= set(columns) <= required|optional or not isinstance(columns["features"], list):
        raise ValueError("字段映射不完整")
    if len(columns["features"]) != len(task.features):
        raise ValueError("特征列与任务输入顺序不匹配")
    if not all(isinstance(c, str) and c for c in columns["features"]):
        raise ValueError("特征列名称无效")
    if not all(isinstance(columns[k], str) and columns[k] for k in set(columns) - {"features"}):
        raise ValueError("身份、标签及用途列名称无效")
    role_values = manifest["role_values"]
    label_values = manifest.get("label_values", {})
    if not isinstance(role_values, dict) or not isinstance(label_values, dict):
        raise ValueError("用途和标签映射需要对象")
    if task.task_type == "classification":
        if not label_values or any(type(v) is not int or v not in task.classes for v in label_values.values()):
            raise ValueError("标签映射超出任务类别或未声明")
    elif label_values:
        raise ValueError("连续目标直接按声明单位读取数值，不使用类别映射")
    encoding = manifest.get("encoding", "utf-8-sig")
    batches, inspections, sources = [], [], []
    records, seen_batches, last_ordinal = {}, set(), -1
    if batch_index is not None and (type(batch_index) is not int or not 0 <= batch_index < len(manifest["batches"])):
        raise ValueError("指定批次索引无效")
    for index, item in enumerate(manifest["batches"]):
        if batch_index is not None and index != batch_index:
            continue
        ordinal = item["ordinal"]
        if type(ordinal) is not int or ordinal <= last_ordinal or item["batch_id"] in seen_batches:
            raise ValueError("批次顺序或身份重复")
        path = _local_file(root, item["path"])
        samples = []
        for index, row in enumerate(_rows(path, item["format"], encoding), start=1):
            try:
                role = role_values[str(row[columns["role"]])]
                raw_label = row[columns["label"]]
                if raw_label is None or raw_label == "":
                    label = None
                elif task.task_type == "classification":
                    label = label_values[str(raw_label)]
                else:
                    if type(raw_label) not in {str, int, float}:
                        raise ValueError("连续目标需要数值，不能使用布尔值或其他对象")
                    label = float(raw_label)
                if not all(isinstance(row[columns[k]], str) and row[columns[k]] for k in ("sample_id", "group")):
                    raise ValueError("样本和分组身份必须为非空字符串")
                if any(type(row[c]) is bool for c in columns["features"]):
                    raise ValueError("特征不能使用布尔值冒充数值")
                if any(type(row[columns[k]]) is bool for k in ("time_start","time_end") if k in columns):
                    raise ValueError("时间区间不能使用布尔值冒充数值")
                sample = Sample(
                    sample_id=row[columns["sample_id"]],
                    revision=_integer(row[columns["revision"]]),
                    values=tuple(float(row[c]) for c in columns["features"]),
                    label=label, role=role, group=row[columns["group"]],
                    available_at=_integer(row[columns["available_at"]]),
                    unit_id=(row[columns["unit_id"]] or None) if "unit_id" in columns else None,
                    **{k:(float(row[columns[k]]) if row[columns[k]] not in (None,"") else None)
                       for k in ("time_start","time_end") if k in columns},
                )
                sample.validate(task, ordinal)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{item['path']}第{index}条数据无法按声明映射：{exc}") from exc
            samples.append(sample)
        if not samples:
            raise ValueError("批次文件没有样本")
        batch = BatchContext(task.task_id, item["batch_id"], ordinal, tuple(samples))
        records, counts = ingest(task, batch, records)
        # 预检保留来源身份；隐藏或后续才返回的答案不能在导入时补入。
        inspections.append({"batch_id": batch.batch_id, "ordinal": ordinal,
                            "rows": len(samples), **counts,
                            "roles": {role: sum(s.role == role for s in samples)
                                      for role in ("fit", "explore", "adopt", "serve")}})
        sources.append({"path": item["path"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        batches.append(batch)
        seen_batches.add(batch.batch_id)
        last_ordinal = ordinal
    if not batches:
        raise ValueError("清单没有声明批次")
    source_digest = digest([manifest, sources])
    inspection = {"origin": origin, "official_layout_confirmed": False,
                  "model_called": False, "task": asdict(task), "objective": task.objective,
                  "source_digest": source_digest,
                  "sources": sources, "batches": inspections}
    return ImportedPackage(task, tuple(batches), origin, source_digest, inspection)
