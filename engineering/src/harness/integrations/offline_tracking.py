"""重新读取SwanLab离线文件，核对实际落盘的指标和事件；不访问云端。"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from harness.runtime.storage import LocalStore, atomic_json


def verify_offline_records(folder: Path) -> dict:
    """按成功分段核查文件内容，失败分段保留但不重复计入。

    读取器来自已核查的SDK内部存储模块，随0.10.4固定；升级时需重新核查
    格式。SDK调用回执与文件核验分别保存，后者失败不会重跑模型任务。
    """
    if version("swanlab") != "0.10.4":
        raise ValueError("离线文件读取适配目前对应swanlab 0.10.4")
    from swanlab.sdk.internal.core_python.store import DataStoreReader
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    from swanlab.proto.swanlab.run.v1.run_pb2 import RUN_STATE_FINISHED
    import yaml

    folder = Path(folder)
    meta = LocalStore(folder).load()
    expected = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    if not meta or meta["simulated_sdk"] or meta["cursor"] != len(expected):
        raise ValueError("真实离线SDK尚未完整写入本地事件")
    runs = {}
    for path in (folder / "swanlog").rglob("*.swanlab"):
        reader = DataStoreReader()
        frames = []
        try:
            reader.open(path)
            for payload in reader:
                frame = Record()
                frame.ParseFromString(payload)
                frames.append(frame)
        finally:
            reader.close()
        if frames and frames[0].HasField("start"):
            name = frames[0].start.name
            if name in runs:
                raise ValueError("同一分段存在重复的SwanLab实验文件")
            runs[name] = path, frames
    checked, scalar_count, files = [], 0, []
    for segment in meta["segments"]:
        if segment["status"] != "sdk_finished":
            continue
        suffix = f"-{meta['run_id']}-part{segment['index']}"
        matches = [item for name, item in runs.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("没有找到唯一的已完成离线分段")
        path, frames = matches[0]
        start = frames[0].start
        if start.project != meta["project"] or start.group != meta["run_id"]:
            raise ValueError("离线文件的项目或运行身份不匹配")
        finishes = [f.finish for f in frames if f.HasField("finish")]
        if len(finishes) != 1 or finishes[0].state != RUN_STATE_FINISHED:
            raise ValueError("离线文件缺少正常结束记录")
        config = yaml.safe_load((path.parent / "files/config.yaml").read_text(encoding="utf-8"))
        config = {key: item["value"] for key, item in config.items()}
        if config.get("fingerprint") != meta["fingerprint"] or config.get("model_experiment") is not meta.get("model_experiment", False):
            raise ValueError("离线配置与来源运行身份不匹配")
        scalars, texts = {}, {}
        for frame in frames:
            if frame.HasField("scalar"):
                record = frame.scalar
                key = record.step, record.key
                if key in scalars:
                    raise ValueError("同一步骤出现重复指标")
                scalars[key] = record.value.number
            elif frame.HasField("media") and frame.media.key == "trace/event":
                record = frame.media
                if record.step in texts or len(record.value.items) != 1:
                    raise ValueError("事件文本重复或缺失")
                item = record.value.items[0]
                paths = list(path.parent.rglob(item.filename))
                if item.payload:
                    data = item.payload
                elif len(paths) == 1:
                    data = paths[0].read_bytes()
                else:
                    raise ValueError("找不到唯一的事件文本文件")
                if len(data) != item.size or hashlib.sha256(data).hexdigest() != item.sha256:
                    raise ValueError("事件文本文件校验失败")
                texts[record.step] = json.loads(data), item.caption
        expected_scalars = {}
        batch = expected[segment["start"]:segment["end"]]
        for record in batch:
            if texts.get(record["step"]) != (record, record["event_id"]):
                raise ValueError("离线事件内容与原始记录不同")
            for key, number in record["metrics"].items():
                expected_scalars[record["step"], key] = number
            checked.append(record["step"])
        if len(texts) != len(batch) or scalars.keys() != expected_scalars.keys():
            raise ValueError("离线指标或事件数量不匹配")
        for key, number in expected_scalars.items():
            if not math.isclose(scalars[key], number, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("离线指标数值与原始记录不同")
        scalar_count += len(scalars)
        files.append({"path": str(path.relative_to(folder)),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if checked != list(range(len(expected))):
        raise ValueError("离线分段没有完整覆盖所有事件")
    report = {"sdk_version": version("swanlab"), "mode": "offline", "verified": True,
              "events": len(checked), "scalar_values": scalar_count, "files": files,
              "model_experiment": meta.get("model_experiment", False), "cloud_verified": False}
    atomic_json(folder / "verification.json", report)
    return report
