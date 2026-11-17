"""读取公开寿命轨迹。两个任务共用整台划分、逐时刻标签和原始检查点的核验。"""
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from zipfile import ZipFile

from harness.core.contracts import Sample, TaskSpec

ARCHIVE_SHA = {
    "bearing": "68db6a72c29bac11a4f86e97d9c1f524d35695464fd5f3f6d131f1431bc69d68",
    "engine": "6cb8784e2cab7ee93453e756abf54c168721c667ca0c2c1327f5faae96e4211f",
}


@dataclass(frozen=True)
class RulPackage:
    task: TaskSpec
    package_id: str
    train: tuple
    validation: tuple
    train_sha: str
    validation_sha: str
    checkpoint_sha: str
    checkpoint: bytes


def read_trajectories(path, profile, task, decode):
    """仅接受带完整失效终点的公开训练/验证轨迹。截断的隐藏序列走平台反馈入口。"""
    with ZipFile(path) as archive:
        info = json.loads(archive.read("baseline-info.json"))
        report = json.loads(archive.read("training.json"))
        if (info["dataset_key"] != profile or report["profile"] != profile
                or info["task"] != "regression" or report["input_features"] != len(task.features)
                or info.get("target_unit", "cycles" if profile == "C" else None) != task.target_unit):
            raise ValueError("官方寿命任务、单位或特征数量不匹配")
        ids = set()

        def split(name, role):
            digest = sha256()
            rows = []
            series = defaultdict(dict)
            with archive.open(f"public/{name}.jsonl") as source:
                for line in source:
                    digest.update(line)
                    raw = json.loads(line)
                    sid, answer = raw["id"], raw["answer"]
                    unit, time, group, values = decode(raw)
                    if (not isinstance(sid, str) or not sid or sid in ids
                            or not isinstance(unit, str) or not unit
                            or type(time) != int or time < 1 or time in series[unit]
                            or type(answer) != int or answer < 0
                            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values)):
                        raise ValueError("公开轨迹的样本、设备、观测时间、特征或寿命标签无效")
                    series[unit][time] = answer
                    ids.add(sid)
                    rows.append(Sample(sid, 1, values, answer, role, group, 0, unit, time, time))
            if (not rows or len(rows) != report[name + "_rows"]
                    or "sha256:" + digest.hexdigest() != report[name + "_sha256"]):
                raise ValueError(f"{name} 的行数或内容与官方报告不符")
            for unit, trajectory in series.items():
                length = len(trajectory)
                if (max(trajectory) != length
                        or any(target != length - t for t, target in trajectory.items())):
                    raise ValueError(f"{unit} 的公开完整轨迹或剩余寿命不连续")
            return tuple(rows), digest.hexdigest(), set(series)

        train, train_sha, train_units = split("train", "fit")
        validation, validation_sha, validation_units = split("validation", "explore")
        checkpoint = archive.read("checkpoint.npz")
        checkpoint_sha = sha256(checkpoint).hexdigest()
        if (train_units & validation_units or info["checkpoint_sha256"] != "sha256:" + checkpoint_sha
                or report["checkpoint_sha256"] != info["checkpoint_sha256"]):
            raise ValueError("训练验证设备重合或官方初始权重不匹配")
        return RulPackage(task, info["id"], train, validation, train_sha, validation_sha,
                          checkpoint_sha, checkpoint)
