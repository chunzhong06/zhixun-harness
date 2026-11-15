"""每次飞行在相同负荷下比较实测与健康机的读数，交接后才打开服务寿命标签。"""
from pathlib import Path
import json

import h5py
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

from score_ds02_forward import read_flights
from score_ncmapss_reference import DEST as OLD, RAW, MEMBER_SHA, file_hash, measures, row_groups
from score_ncmapss_field import forest
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from catboost import CatBoostRegressor

DEST = OLD.parent / "engine-flight-load"
SOURCE = {4, 5, 6}


def signals(dataset, part, allowed=None):
    """按连续航班的物理采样顺序下采样，各模型均看不到设备号和飞行号。"""
    aux = dataset["A_" + part][:, :2]
    if not np.array_equal(aux, aux.astype(int)):
        raise ValueError("设备和飞行序号不可还原")
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(aux, axis=0), axis=1)) + 1, len(aux)]
    flights = {}
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        unit, cycle = map(int, aux[start])
        if allowed is not None and unit not in allowed:
            continue
        key = (unit, cycle)
        if key in flights:
            raise ValueError("同一飞行的采样不是一个连续区间")
        sensor = dataset["X_s_" + part][start:end]
        settings = dataset["W_" + part][start:end]
        data = np.column_stack((sensor, settings))
        if not np.isfinite(data).all():
            raise ValueError("飞行含无效读数")
        flights[key] = (data.mean(axis=0), data[::16].copy())
    return flights


def fit_tools(flights, labels):
    early = []
    for unit in sorted(SOURCE):
        keys = sorted(k for k in flights if k[0] == unit)
        if len(keys) < 8:
            raise ValueError("开工飞行不足")
        early.extend(flights[k][1] for k in keys[:8])
    ref = np.concatenate(early)
    scale = StandardScaler().fit(ref[:, 14:])
    z = scale.transform(ref[:, 14:])
    healthy = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.)).fit(z, ref[:, :14])
    centers = KMeans(n_clusters=32, random_state=17, n_init=3).fit(z)
    radius = np.quantile(centers.transform(z).min(axis=1), .95)
    prep = (scale, healthy, centers, radius)
    all_features, supported_features, _ = features(flights, prep)
    truth = np.array([labels[k] for k in sorted(flights)])
    def train(x):
        return forest().fit(x, truth), make_pipeline(StandardScaler(), Ridge(alpha=1.)).fit(x, truth)
    return prep, train(all_features), train(supported_features)


def features(flights, prep):
    scale, healthy, centers, radius = prep
    all_features, supported, coverage = [], [], []
    for key in sorted(flights):
        _, rows = flights[key]
        z = scale.transform(rows[:, 14:])
        residual = rows[:, :14] - healthy.predict(z)
        keep = centers.transform(z).min(axis=1) <= radius
        avg = residual.mean(axis=0)
        all_features.append(avg)
        supported.append(residual[keep].mean(axis=0) if keep.any() else avg)
        coverage.append(np.mean(keep))
    return np.array(all_features), np.array(supported), np.array(coverage)


def old_arms(x):
    scores = json.loads((OLD / "scores.json").read_bytes())
    req = scores["requests"]["service"]
    payload = json.loads((OLD / "accepted" / (req + ".json")).read_bytes())["payload"]
    if digest(["service", payload]) != req:
        raise ValueError("旧工具身份改变")
    for name, checksum in payload["models"].items():
        path = OLD / ("preparation.joblib" if name == "preparation" else name + ".cbm")
        if file_hash(path) != checksum:
            raise ValueError("旧训练产物身份改变")
    op, cluster, scales, global_scale, simple = joblib.load(OLD / "preparation.joblib")
    group = row_groups(x, op, cluster)
    inputs = {"raw": x, "operating": x[:, 14:], "global": global_scale.transform(x),
              **{k: np.column_stack((sc.transform(group)[:, :14], x[:, 14:])) for k, sc in scales.items()}}
    out = {k: CatBoostRegressor().load_model(str(OLD / (k + ".cbm"))).predict(v, thread_count=1)
           for k, v in inputs.items()}
    out.update({k: m.predict(x) for k, m in simple.items()})
    out["half"] = (out["forest"] + out["ridge"]) / 2
    return out


def deliver(receiver, site, flights, bundle):
    prep, full, supported = bundle
    x = np.array([flights[k][0] for k in sorted(flights)])
    pred = old_arms(x)
    a, b, support = features(flights, prep)
    for name, pair, vals in (("全飞行偏差", full, a), ("可比工况偏差", supported, b)):
        tree, ridge = pair
        pred[name + "森林"] = tree.predict(vals)
        pred[name + "岭"] = ridge.predict(vals)
        pred[name] = (pred[name + "森林"] + pred[name + "岭"]) / 2
    pred["固定折半"] = .5 * (pred["half"] + pred["全飞行偏差"])
    pred["按覆盖选用"] = support * pred["全飞行偏差"] + (1 - support) * pred["half"]
    payload = {"ids": [list(k) for k in sorted(flights)], "coverage": support.tolist(),
               "model": file_hash(DEST / "tools.joblib"), "predictions": {k: v.tolist() for k, v in pred.items()}}
    key = digest([site, payload])
    ack = receiver.submit(key, payload)
    if not ack["accepted"] or ack["request_id"] != key:
        raise RuntimeError("飞行交接未完成")
    return key


def score(site, receiver, key, dataset):
    payload = json.loads((receiver.output / (key + ".json")).read_bytes())["payload"]
    if digest([site, payload]) != key:
        raise ValueError("交接后预测身份改变")
    truth = read_flights(dataset, "test", answers=True)
    keys = [tuple(k) for k in payload["ids"]]
    if set(keys) != set(truth) or len(keys) != len(truth):
        raise ValueError("服务飞行与真值未对齐")
    y, owners = np.array([truth[k] for k in keys]), np.array([k[0] for k in keys])
    table = {k: measures(y, np.array(p), owners) for k, p in payload["predictions"].items()}
    print(json.dumps({site: {k: {"误差": v["rmse"], "最差台": v["worst"]} for k, v in table.items()}},
                     ensure_ascii=False), flush=True)
    return {"request": key, "scores": table, "rows": len(keys)}


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有飞行工况结果")
        third, = RAW.parent.glob("N-CMAPSS_DS03-*.h5")
        checksum = json.loads(third.with_suffix(".source.json").read_bytes())["sha256"]
        if file_hash(RAW) != MEMBER_SHA or file_hash(third) != checksum:
            raise ValueError("源机队或已见第三机队身份改变")
        with h5py.File(RAW) as first, h5py.File(third) as other:
            for k in ("W_var", "X_s_var", "A_var"):
                if not np.array_equal(first[k][...], other[k][...]):
                    raise ValueError("字段不一致")
            source = signals(first, "dev", SOURCE)
            # 开发包其它设备有合法答案，此处只读取三台源训练设备的标签。
            aux = first["A_dev"][:, :2].astype(int)
            labels = {}
            for start in np.r_[0, np.flatnonzero(np.any(np.diff(aux, axis=0), axis=1)) + 1]:
                key = tuple(map(int, aux[start]))
                if key in source:
                    labels[key] = float(first["Y_dev"][start].item())
            joblib.dump(fit_tools(source, labels), DEST / "tools.joblib")
            bundle = joblib.load(DEST / "tools.joblib")
            receiver = LocalAcceptance()
            receiver.output = DEST / "accepted"
            datasets = {"DS01": first, "DS03": other}
            requests = {k: deliver(receiver, k, signals(f, "test"), bundle) for k, f in datasets.items()}
            results = {k: score(k, receiver, requests[k], f) for k, f in datasets.items()}
            for site, reference in (("DS01", OLD / "scores.json"), ("DS03", OLD.parent / "ncmapss-ds03-forward/scores.json")):
                past = json.loads(reference.read_bytes())
                previous = past["sites"]["service"] if site == "DS01" else past["scores"]["test"]
                for arm in ("forest", "ridge", "half", "raw", "early", "late"):
                    np.testing.assert_allclose(results[site]["scores"][arm]["rmse"], previous[arm]["rmse"], atol=1e-7, rtol=0)
            atomic_json(DEST / "scores.json", results)


if __name__ == "__main__":
    main()
