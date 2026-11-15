"""DS04整批冻结后首次评分；复用三支旧机队实际原始记录检查重叠。"""
from hashlib import sha256
import json
import sys

import h5py
import joblib
import numpy as np

from score_engine_flight_load import DEST as TRAINED, RAW, signals, features, old_arms, score
from score_ncmapss_field import file_hash
from harness.contracts import digest
from harness.models.flight_health import FlightHealth
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
sys.path.insert(0, str(RAW.parents[3]))
from engineering.cli.workflow.flight_input_predict import infer

DEST = TRAINED.parent / "engine-flight-forward"


def signature(rows):
    return sha256(np.ascontiguousarray(rows, dtype="<f8").tobytes()).hexdigest()


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("第四机队已评分")
        seen, weather = set(), set()
        for subset in ("DS01", "DS02", "DS03"):
            path, = RAW.parent.glob(f"N-CMAPSS_{subset}-*.h5")
            with h5py.File(path) as f:
                for part in ("dev", "test"):
                    for _, arr in signals(f, part).values():
                        seen.add(signature(arr))
                        weather.add(signature(arr[:, 14:]))
        target = RAW.parent / "N-CMAPSS_DS04.h5"
        meta = json.loads(target.with_suffix(".source.json").read_bytes())
        if file_hash(target) != meta["sha256"]:
            raise ValueError("第四机队原件身份不符")
        bundle = joblib.load(TRAINED / "tools.joblib")
        train_scores = json.loads((TRAINED / "scores.json").read_bytes())
        prev_key = train_scores["DS01"]["request"]
        prev = json.loads((TRAINED / "accepted" / (prev_key + ".json")).read_bytes())["payload"]
        if digest(["DS01", prev]) != prev_key or file_hash(TRAINED / "tools.joblib") != prev["model"]:
            raise ValueError("与先前开发比较的发布产物不同")
        with h5py.File(target) as dataset, h5py.File(RAW) as source:
            for field in ("W_var", "X_s_var", "A_var"):
                if not np.array_equal(dataset[field][...], source[field][...]):
                    raise ValueError("传感器、工况或设备航班字段不一致")
            flights = signals(dataset, "test")
            keys = sorted(flights)
            fingerprints = [signature(flights[k][1]) for k in keys]
            if len(set(fingerprints)) != len(keys) or seen.intersection(fingerprints):
                raise ValueError("第四机队飞行测量与历史重复")
            reused = sum(signature(flights[k][1][:, 14:]) in weather for k in keys)
            x = np.array([flights[k][0] for k in keys])
            out = old_arms(x)
            a, b, cover = features(flights, bundle[0])
            for name, vals, pair in (("全飞行偏差", a, bundle[1]), ("可比工况偏差", b, bundle[2])):
                out[name + "森林"] = pair[0].predict(vals)
                out[name + "岭"] = pair[1].predict(vals)
                out[name] = .5 * (out[name + "森林"] + out[name + "岭"])
            full, comparable = out["全飞行偏差森林"], out["可比工况偏差森林"]
            for weight in (.25, .5, .75):
                out[f"固定全飞行权重{weight}"] = weight * full + (1 - weight) * comparable
            out["固定折半"] = .5 * (out["half"] + out["全飞行偏差"])
            out["按覆盖选用"] = cover * out["全飞行偏差"] + (1 - cover) * out["half"]
            # 交付候选实际走外部调用者使用的整趟飞行入口；研究侧批量固定对照独立计算。
            model = FlightHealth(TRAINED / "tools.joblib", prev["model"])
            aux = dataset["A_test"][:, :2].astype(int)
            bounds = np.r_[0, np.flatnonzero(np.any(np.diff(aux, axis=0), axis=1)) + 1, len(aux)]
            by_id = {}
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                flight = tuple(map(int, aux[start]))
                request = {"inputs": {"sensors": dataset["X_s_test"][start:end], "settings": dataset["W_test"][start:end]}}
                response, = infer(model, [request], "coverage")
                by_id[flight] = response["outputs"]["value"]
            out["现场两森林"] = np.array([by_id[k] for k in keys])
            np.testing.assert_allclose(out["现场两森林"], cover * full + (1 - cover) * comparable, rtol=0, atol=1e-9)
            payload = {"ids": [list(k) for k in keys], "predictions": {k: v.tolist() for k, v in out.items()},
                       "model": prev["model"], "coverage": cover.tolist(), "source": meta["sha256"]}
            receiver = LocalAcceptance()
            receiver.output = DEST / "accepted"
            key = digest(["DS04", payload])
            ack = receiver.submit(key, payload)
            if not ack["accepted"] or ack["request_id"] != key:
                raise RuntimeError("独立飞行预测交接未完成")
            result = score("DS04", receiver, key, dataset)
            result.update({"source": meta, "prior_fingerprints": len(seen), "reused_operating_sequences": reused,
                           "test_units": sorted({u for u, _ in keys}), "overlapping_measured_flights": 0})
            atomic_json(DEST / "scores.json", result)


if __name__ == "__main__":
    main()
