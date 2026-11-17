"""官方 C 包的整台发动机、运行条件与剩余循环。输入顺序跟随官方预测器。"""
import math
from harness.core.contracts import TaskSpec
from harness.adapters.datasets.cmapss_input import condition_key
from harness.adapters.datasets.official_rul import read_trajectories


SETTINGS = tuple(f"settings.{i}" for i in range(3))
SENSORS = tuple(f"sensors.{i}" for i in range(21))
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_official_engine(path):
    task = TaskSpec("official-engine-C", SENSORS + SETTINGS, metric="rmse", task_type="regression",
                    target_unit="cycles", condition_features=SETTINGS, condition_decimals=(0, 2, 0))

    def decode(raw):
        meta, fields = raw["meta"], raw["inputs"]
        subset, number = meta["source_subset"], meta["unit"]
        if (subset not in SUBSETS or type(number) != int or number < 1 or meta["source_split"] != "train"
                or type(meta["window_size"]) != int or meta["window_size"] != 1 or set(fields) != {"settings", "sensors"}
                or not isinstance(fields["settings"], list) or len(fields["settings"]) != 3
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in fields["settings"])
                or not isinstance(fields["sensors"], list) or len(fields["sensors"]) != 21):
            raise ValueError("需要含子集和设备身份的单循环官方发动机观测")
        settings = fields["settings"]
        # 工况只使用当时可见的三项读数。跨子集重复的设备号用不同身份追踪。
        return (f"{subset}:u{number:04d}", meta["cycle"], condition_key(settings),
                tuple(fields["sensors"] + settings))

    return read_trajectories(path, "C", task, decode)
