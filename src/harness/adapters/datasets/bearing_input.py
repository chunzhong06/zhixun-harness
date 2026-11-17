"""XJTU-SY两路振动的逐文件特征入口；保留设备和1.28秒观察窗口。

依据作者数据说明：25.6kHz，每分钟保存32768点。这里生成常用时域特征，
不复现作者完整寿命算法，也不由文件后缀推断健康/故障分类标签。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from harness.core.contracts import Sample, TaskSpec

STATISTICS = ("rms", "std", "abs_peak", "peak_to_peak", "kurtosis", "crest")
CONDITIONS = {1: (2100., 12.), 2: (2250., 11.), 3: (2400., 10.)}


@dataclass(frozen=True)
class VibrationRecord:
    sample_id: str
    unit_id: str
    file_index: int
    condition: int
    sha256: str
    values: tuple[float, ...]


def bearing_task():
    features = ("rpm", "load_kn", *(f"{axis}_{stat}" for axis in ("horizontal", "vertical") for stat in STATISTICS))
    return TaskSpec("xjtu-sy-time-features-rul", features, metric="rmse", task_type="regression",
        target_unit="minutes", diagnostic_features=features,
        condition_features=("rpm", "load_kn"), condition_decimals=(0, 0))


def read_vibration(path, *, bearing, condition, file_index, expected_sha256=None, expected_points=32768):
    """读取一个源CSV并计算时域统计；哈希覆盖表头和全部原始字节。

    常量信号的峰度与峰值因子定义为0，报告据此可辨认恒定输入；表头必须
    为两列且包含Horizontal/Vertical，不把非数字坏行当成额外表头跳过。
    """
    if condition not in CONDITIONS or type(file_index) is not int or file_index < 1 or not bearing:
        raise ValueError("需明确设备、工况与采样序号")
    path = Path(path)
    with path.open("rb") as raw:
        checksum = hashlib.file_digest(raw, "sha256").hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("振动原件与声明身份不符")
    # 一次保留一个采样文件，内存用量不随全寿命文件数累积。
    channels = [[], []]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if len(header) != 2 or "horizontal" not in header[0].lower() or "vertical" not in header[1].lower():
            raise ValueError("振动CSV表头与作者两路信号定义不符")
        for row in reader:
            if len(row) != 2:
                raise ValueError("振动记录须为两列")
            for values, value in zip(channels, row):
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("振动读数必须有限")
                values.append(number)
    if len(channels[0]) != expected_points or expected_points < 2:
        raise ValueError("振动采样长度与声明不一致")
    features = list(CONDITIONS[condition])
    for values in channels:
        n = len(values)
        mean = math.fsum(values) / n
        variance = math.fsum((v-mean)**2 for v in values) / n
        rms = math.sqrt(math.fsum(v*v for v in values) / n)
        peak = max(abs(v) for v in values)
        kurtosis = math.fsum((v-mean)**4 for v in values) / n / variance**2 if variance else 0.
        features.extend((rms, math.sqrt(variance), peak, max(values)-min(values), kurtosis, peak/rms if rms else 0.))
    if not all(math.isfinite(x) for x in features):
        raise ValueError("振动特征超出有限数值范围")
    unit = f"xjtu-sy:condition{condition}:{bearing}"
    return VibrationRecord(f"{unit}:file{file_index}:{checksum}", unit, file_index, condition, checksum, tuple(features))


def bind_vibration(record, *, ordinal, role, failure_index=None):
    """完整离线寿命获准后显式提供failure_index；服务输入始终不带答案。

    失效时点用于离线标签，不参与时域特征。文件序号表示每分钟采样顺序；
    time_start/end统一使用分钟，窗口长度为1.28/60分钟。
    """
    if role not in {"fit", "explore", "adopt", "serve"}:
        raise ValueError("未知用途")
    if role != "serve" and (type(failure_index) is not int or failure_index < record.file_index):
        raise ValueError("有答案用途需要明确合法的失效采样序号")
    start = float(record.file_index-1)
    sample = Sample(record.sample_id, 1, record.values,
        None if role == "serve" else float(failure_index-record.file_index), role,
        f"condition{record.condition}", ordinal, unit_id=record.unit_id,
        time_start=start, time_end=start+1.28/60.)
    sample.validate(bearing_task(), ordinal)
    return sample
