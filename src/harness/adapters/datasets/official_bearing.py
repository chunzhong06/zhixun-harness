"""读取官方 B 包的两通道振动特征、设备身份和剩余分钟数，保留公开划分。"""
from harness.core.contracts import TaskSpec
from harness.adapters.datasets.official_rul import RulPackage as BearingPackage, read_trajectories

CHANNELS = ("horizontal", "vertical")
# 对应随包预测适配器的字段路径排序。设备、采样序号和答案不进入模型特征。
MEASURES = tuple(sorted(("rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness",
                        "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
                        "spectral_centroid", "spectral_rms", "spectral_kurtosis", "dominant_freq",
                        "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k")))


def read_official_bearing(path):
    features = tuple(f"{channel}.{name}" for channel in CHANNELS for name in MEASURES)
    task = TaskSpec("official-bearing-B", features, metric="rmse", task_type="regression", target_unit="minutes")

    def decode(raw):
        inputs = raw["inputs"]
        channels = inputs["features"]
        if (set(inputs) != {"features"} or set(channels) != set(CHANNELS)
                or any(set(channels[c]) != set(MEASURES) for c in CHANNELS)):
            raise ValueError("振动快照需要两通道的官方34项特征")
        unit = raw["meta"]["bearing"]
        values = tuple(channels[c][name] for c in CHANNELS for name in MEASURES)
        return unit, raw["meta"]["snapshot"], unit, values

    return read_trajectories(path, "B", task, decode)
