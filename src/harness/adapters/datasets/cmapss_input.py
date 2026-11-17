"""从固定C-MAPSS原包读取完整轨迹；目标与因果输入分别生成。

列定义依据NASA原包readme及TACDA作者处理代码。训练寿命由完整离线轨迹
计算，只能在轨迹答案获准后使用；测试答案单独输出，绝不进入serve记录。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import zipfile

from harness.core.contracts import Sample, TaskSpec

REGISTERED_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"


@dataclass(frozen=True)
class EngineTrajectory:
    source_sha256: str
    member: str
    unit: int
    values: tuple[tuple[float, ...], ...]  # 每行3项设置和21项读数，cycle由位置+1给出。

    @property
    def unit_id(self):
        return f"cmapss:{self.source_sha256}:{self.member}:unit{self.unit}"


def read_cmapss(path: Path, *, subset="FD001", partition="train", expected_sha256=REGISTERED_SHA256):
    """只读取指定子集；核对26列、连续周期及设备身份，不读取测试答案。"""
    if subset not in {"FD001", "FD002", "FD003", "FD004"} or partition not in {"train", "test"}:
        raise ValueError("C-MAPSS子集或用途无效")
    with Path(path).open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != expected_sha256:
            raise ValueError("发动机原包与声明来源不同")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            name = f"{partition}_{subset}.txt"
            members = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == name]
            if len(members) != 1:
                raise ValueError("子集文件缺失或重复")
            tracks = {}
            with archive.open(members[0]) as raw:
                for line in raw:
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 26:
                        raise ValueError("发动机记录须为26列")
                    unit, cycle = int(fields[0]), int(fields[1])
                    values = tuple(float(v) for v in fields[2:])
                    rows = tracks.setdefault(unit, [])
                    if unit < 1 or cycle != len(rows) + 1 or not all(math.isfinite(v) for v in values):
                        raise ValueError("设备身份、周期连续性或读数无效")
                    rows.append(values)
            if not tracks or sorted(tracks) != list(range(1, len(tracks) + 1)):
                raise ValueError("设备编号须完整连续")
            return tuple(EngineTrajectory(checksum, members[0], unit, tuple(rows))
                         for unit, rows in sorted(tracks.items()))


def cmapss_task(subset="FD001", *, window=20):
    if type(window) is not int or window < 1:
        raise ValueError("窗口须为正整数")
    settings = tuple(f"setting{i}" for i in range(1, 4))
    sensors = tuple(f"sensor{i}" for i in range(1, 22))
    features = (*settings, *sensors, *(f"{s}_mean" for s in sensors), *(f"{s}_slope" for s in sensors))
    # 同时覆盖设置与不同传感器；有界联合搜索不再隐式只看最前几列。
    diagnostics = (*settings, *(f"sensor{i}" for i in (2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21)))
    return TaskSpec(f"cmapss-{subset}-window{window}", features, metric="rmse", task_type="regression",
                    target_unit="cycles", diagnostic_features=diagnostics,
                    condition_features=settings, condition_decimals=(0, 2, 0))


def condition_key(settings):
    """舍入只用于工况报告；保留原始模型输入，并将正负零归在同一工况。"""
    return "condition:" + ",".join(str(round(v, d) + 0.0) for v, d in zip(settings, (0, 2, 0), strict=True))


def trajectory_samples(track, *, window, stride, role, ordinal, final_only=False, rul_cap=None, short_prefix="reject"):
    """窗口止于当前周期；抽样发生在窗口构造之后。

    train轨迹才可产生离线寿命标签。serve可读取train或test前缀，均移除标签；
    预测时不使用终点、剩余寿命或相对全寿命百分比作为特征。
    """
    if type(window) is not int or window < 1 or type(stride) is not int or stride < 1:
        raise ValueError("窗口与步长须为正整数")
    if role not in {"fit", "explore", "adopt", "serve"} or (role != "serve" and not track.member.rsplit("/", 1)[-1].startswith("train_")):
        raise ValueError("测试轨迹只能生成无答案服务输入")
    if rul_cap is not None and (type(rul_cap) not in {int, float} or not math.isfinite(rul_cap) or rul_cap <= 0):
        raise ValueError("寿命截断须为显式有限正值")
    n = len(track.values)
    if short_prefix not in {"reject", "available"}:
        raise ValueError("未知短前缀处理规则")
    if n < window and short_prefix == "reject":
        raise ValueError("已观察前缀短于所配置完整窗口")
    effective_window = min(window, n)
    ends = [n] if final_only else sorted(set([*range(effective_window, n + 1, stride), n]))
    result = []
    for end in ends:
        prefix = track.values[end-effective_window:end]
        last = prefix[-1]
        mean = tuple(sum(row[i] for row in prefix) / effective_window for i in range(3, 24))
        slope = tuple((last[i] - prefix[0][i]) / max(1, effective_window-1) for i in range(3, 24))
        label = None if role == "serve" else float(n-end if rul_cap is None else min(rul_cap, n-end))
        result.append(Sample(f"{track.unit_id}:cycle{end}:window{window}", 1,
            (*last, *mean, *slope), label, role, condition_key(last[:3]), ordinal,
            unit_id=track.unit_id, time_start=end-effective_window+1, time_end=end))
    return tuple(result)


def test_answers(path, subset, tracks, *, expected_sha256=REGISTERED_SHA256):
    """评价侧读取每台测试发动机最后周期的剩余寿命；不补入运行清单。"""
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("评价答案来源不匹配")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            names = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == f"RUL_{subset}.txt"]
            if len(names) != 1:
                raise ValueError("测试答案文件缺失或重复")
            values = [float(x) for x in archive.read(names[0]).split()]
    if len(values) != len(tracks) or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("测试答案数量或数值无效")
    return {t.unit_id: values[t.unit-1] for t in tracks}
