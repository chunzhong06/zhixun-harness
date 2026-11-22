"""整趟飞行落地后，只从此次实测读数及源端封存工具估计剩余飞行次数。"""
from hashlib import file_digest
from pathlib import Path

import joblib
import numpy as np

from harness.learning.models.flight.flight_forest_selection import select_flight_forest


class FlightHealth:
    def __init__(self, artifact, expected_sha256):
        # 只加载本地训练流水线发布且摘要匹配的模型。
        with Path(artifact).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError("飞行健康参照与发布身份不一致")
            stream.seek(0)
            self.preparation, full, supported = joblib.load(stream)
        self.full = full[0]
        self.supported = supported[0]

    def predict(self, sensors, settings, method="coverage"):
        sensors, settings = np.asarray(sensors, dtype=float), np.asarray(settings, dtype=float)
        if (sensors.ndim != 2 or sensors.shape[1] != 14 or len(sensors) == 0 or
                settings.shape != (len(sensors), 4) or not np.isfinite(sensors).all() or
                not np.isfinite(settings).all()):
            raise ValueError("每次完整飞行需要逐时刻对齐的14项传感器和4项工况读数")
        scale, healthy, centers, radius = self.preparation
        operating = scale.transform(settings[::16])
        residual = sensors[::16] - healthy.predict(operating)
        supported = centers.transform(operating).min(axis=1) <= radius
        avg = residual.mean(axis=0)
        comparable = residual[supported].mean(axis=0) if supported.any() else avg
        if method == "whole":
            return float(self.full.predict(avg[None, :])[0])
        if method == "supported":
            return float(self.supported.predict(comparable[None, :])[0])
        if method != "coverage":
            raise ValueError("没有此飞行估计方式")
        whole = self.full.predict(avg[None, :])[0]
        local = self.supported.predict(comparable[None, :])[0]
        return float(select_flight_forest(whole, local, np.mean(supported)))
