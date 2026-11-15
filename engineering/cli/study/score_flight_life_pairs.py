"""统一五组源dev训练，已见六组test实际交接，在同一固定对手上核算训练和调度收益。"""
import json

import h5py
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from score_flight_complement import BASE, receipt, digest, measures, LocalAcceptance, LocalStore, atomic_json
from score_flight_load_neighbors import SITES, signals, dataset_path, read_flights, file_hash
from harness.models.flight_range_choice import range_views
from harness.methods.flight_complement import fixed_mix
from harness.methods.flight_life_pairs import LifeDifferenceFit

DEST = BASE / "flight-life-pairs"
PRIOR = BASE / "flight-correlated-tools"
PREP = BASE / "flight-range-choice/range.joblib"
PAPER = "整趟序列"


def extract(site, part, prep):
    with h5py.File(dataset_path(site)) as f:
        flights = signals(f, part)
        answers = read_flights(f, part, answers=True)
    keys = sorted(flights)
    if set(keys) != set(answers):
        raise ValueError("训练或回放测量与作者寿命不一致")
    rows = []
    for k in keys:
        views = range_views(prep, flights[k][1])
        rows.append(np.concatenate(views[3:]))
    return keys, np.asarray(rows), np.asarray([answers[k] for k in keys])


def source_features():
    prep = joblib.load(PREP).prep
    data = {s: extract(s, "dev", prep) for s in SITES}
    keys = [(s, u, c) for s, v in data.items() for u, c in v[0]]
    x = np.vstack([v[1] for v in data.values()])
    y = np.concatenate([v[2] for v in data.values()])
    return prep, x, y, keys


def mix_inputs(p):
    return np.column_stack((p, p.std(axis=1), p[:, 2] - p[:, 0], p[:, -1] - p[:, 0]))


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("同台训练已实测")
        prep, x, y, keys = source_features()
        fit = LifeDifferenceFit(x, y, keys)
        joblib.dump(fit, DEST / "pair.joblib")
        prior = json.loads((PRIOR / "scores.json").read_bytes())["sites"]
        records = {}
        for site, entry in prior.items():
            ids, vals, truth = extract(site, "test", prep)
            old = receipt(PRIOR, site, entry)
            if old["ids"] != [list(k) for k in ids]:
                raise ValueError("与旧模型比较的航班未对齐")
            fresh = fit.predict(vals)
            outputs = old["predictions"]
            matrix = np.column_stack([*fresh.values(), outputs["三段联合森林"], outputs[PAPER]])
            records[site] = (ids, matrix, truth, {**outputs, **{k: v.tolist() for k, v in fresh.items()}})
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        scores = {}
        for site, (ids, m, truth, outputs) in records.items():
            sources = {k: v for k, v in records.items() if k != site}
            train = np.vstack([v[1] for v in sources.values()])
            labels = np.concatenate([v[2] for v in sources.values()])
            fleets = np.concatenate([np.repeat(k, len(v[2])) for k, v in sources.items()])
            w = fixed_mix(train, labels, fleets)
            sw = np.array([len(fleets) / (len(sources) * len(sources[k][2])) for k in fleets])
            error = ExtraTreesRegressor(n_estimators=80, max_depth=3, min_samples_leaf=20, n_jobs=1, random_state=17)
            error.fit(mix_inputs(train), abs(train - labels[:, None]), sample_weight=sw)
            def confidence(z):
                weight = np.maximum(1, error.predict(mix_inputs(z))) ** -2
                return weight / weight.sum(axis=1, keepdims=True)
            weights = confidence(m)
            soft = (m * weights).sum(axis=1)
            fixed = m @ w
            out = dict(outputs)
            out.update({"新旧五工具固定": fixed.tolist(), "差分现场估错": soft.tolist(),
                        "差分现场与固定各半": ((fixed + soft) / 2).tolist(),
                        "差分现场固定均值": (m @ np.average(confidence(train), axis=0, weights=sw)).tolist()})
            payload = {"ids": [list(k) for k in ids], "predictions": out}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("同台训练预测未取得回执")
            rows = receipt(DEST, site, {"request": key})["predictions"]
            owners = np.array([k[0] for k in ids])
            scores[site] = {"request": key, "rows": len(ids),
                            "scores": {k: measures(truth, np.asarray(v), owners) for k, v in rows.items()}}
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in rows}
        choices = ("差分现场估错", "差分现场与固定各半")
        dynamic = {*choices, "可比工况共同误差", "窗口估错调权", "窗口差异校正", "前次现场规则", "现场与六工具固定各半", "航班互补调权"}
        fixed = min((k for k in means if k not in dynamic), key=means.get)
        selected = min(choices, key=means.get)
        gain = 1 - means[selected] / means[fixed]
        decision = {"最强固定": fixed, "主候选": selected, "开发降幅": gain, "进入独立验证": gain >= .01 and means[selected] < means[PAPER]}
        atomic_json(DEST / "scores.json", {"sites": scores, "means": means, "decision": decision,
                                          "source_flights": len(keys), "pairs": fit.pairs, "model": file_hash(DEST / "pair.joblib")})
        print(json.dumps({"同台航班对": fit.pairs, "等机队平均": means, "取舍": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
