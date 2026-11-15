"""从单次航班的可见测量生成所有候选读法，离线研究与逐航班调用使用同一实现。"""
import numpy as np


def flight_views(prep, mean, rows):
    if rows.ndim != 2 or rows.shape[1] != 18 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("单次飞行需要14个传感器和4个工况")
    scale, healthy, _, _ = prep
    residual = rows[:, :14] - healthy.predict(scale.transform(rows[:, 14:]))
    signed = residual.mean(axis=0)
    # 全程标准差与相邻采样差区分航班内慢变化和短期起伏；仅依赖已落地的当次飞行。
    fast = np.abs(np.diff(residual, axis=0)).mean(axis=0) if len(residual) > 1 else np.zeros(14)
    return {"原始均值": mean, "健康偏差": signed, "偏差幅度": np.abs(residual).mean(axis=0),
            "偏差与波动": np.r_[signed, residual.std(axis=0), fast]}


def fleet_prediction(bundle, mean, rows):
    from harness.models.flight_load_neighbors import FlightLoadNeighbors
    from harness.models.flight_range_choice import FlightRangeChoice
    if isinstance(bundle, (FlightLoadNeighbors, FlightRangeChoice)):
        return bundle.forecast(rows)
    if isinstance(bundle, dict):
        from harness.methods.flight_disagreement import combine
        views = flight_views(bundle["preparation"], mean, rows)
        p = np.array([[est.predict(views[name.split("/")[0]][None,:])[0]
                       for name, est in bundle["tools"].items()]])
        return float(combine(p, bundle["weights"])[0])
    selected, tools, prep = bundle
    view = selected.split("/")[0]
    return float(tools[selected].predict(flight_views(prep, mean, rows)[view][None, :])[0])
