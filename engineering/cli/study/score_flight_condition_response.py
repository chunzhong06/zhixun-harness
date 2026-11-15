"""评估源端六个固定工况的响应读法及当前飞行能否自动判断哪些参照可信。"""
import json
import h5py
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from score_flight_wear_transfer import BASE, DEST as PRIOR, PRIMARY as WEAR_RULE
from score_flight_fault_experts import issue, ANCHOR, PAPER
from score_flight_segment_training import SITES, PREP, dataset_path, signals, read_flights, LocalStore, atomic_json, receipt, file_hash
from harness.methods.flight_complement import fixed_mix
from harness.models.flight_condition_response import read_conditions, reference_points, ConditionReadout

DEST = BASE / "flight-condition-response"
PRIMARY = "用当前覆盖与预测分歧选择工况"
POINT_NAMES = [f"固定工况{i+1}" for i in range(6)]


def acquire(site, part, prep, points):
    with h5py.File(dataset_path(site)) as f:
        rows = signals(f, part)
        answers = read_flights(f, part, answers=True)
    keys = sorted(rows)
    if set(keys) != set(answers):
        raise ValueError("工况读数与作者航班身份不同")
    views, distance = zip(*(read_conditions(rows[k][1], prep, points) for k in keys), strict=True)
    return keys, np.stack(views), np.stack(distance), np.array([answers[k] for k in keys])


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("工况实测已经交付")
        prep = joblib.load(PREP).prep
        points = reference_points(prep)
        training = [acquire(s, "dev", prep, points) for s in SITES]
        model = ConditionReadout().fit(np.concatenate([v[1] for v in training]), np.concatenate([v[3] for v in training]))
        weights = DEST / "condition.joblib"
        joblib.dump({"readout": model, "points": points, "prep": prep}, weights)
        previous = json.loads((PRIOR / "scores.json").read_bytes())
        records = {}
        for s, item in previous["sites"].items():
            ids, x, dist, y = acquire(s, "test", prep, points)
            old = receipt(PRIOR, s, item)
            if [list(k) for k in ids] != old["ids"]:
                raise ValueError("开发航班与旧回执不符")
            pred = dict(old["predictions"])
            values, spread = model.readings(x)
            for i, name in enumerate(POINT_NAMES):
                pred[name] = values[:, i]
            pred["六工况固定均值"] = values.mean(1)
            near = (1 + dist) ** -2
            conf = (1 + spread) ** -2
            for name, w in (("工况覆盖固定公式", near), ("工况分歧固定公式", conf), ("覆盖和分歧乘积", near * conf)):
                pred[name] = np.sum(w * values, axis=1) / w.sum(1)
            pred["旧三段与覆盖分歧各半"] = (pred["覆盖和分歧乘积"] + pred[ANCHOR]) / 2
            matrix = np.column_stack((values, pred[ANCHOR], pred[PAPER]))
            context = np.column_stack((dist, spread, values - values.mean(1)[:, None], matrix[:, -2:]))
            records[s] = (ids, y, pred, matrix, context)
        scores = {}
        for s, (ids, truth, pred, matrix, context) in records.items():
            others = {k:v for k,v in records.items() if k != s}
            y, m, c = [np.concatenate([v[i] for v in others.values()]) for i in (1, 3, 4)]
            sites = np.concatenate([np.repeat(k, len(v[1])) for k,v in others.items()])
            sw = np.array([len(y)/(len(others)*len(others[k][1])) for k in sites])
            fixed = fixed_mix(m, y, sites)
            pred["六工况与旧三段及论文固定组合"] = matrix @ fixed
            gate = ExtraTreesRegressor(n_estimators=96, max_depth=4, min_samples_leaf=24, n_jobs=1, random_state=17)
            gate.fit(c, abs(m - y[:, None]), sample_weight=sw)
            def fractions(info):
                w = gate.predict(info).clip(1) ** -2
                return w / w.sum(axis=1, keepdims=True)
            pred["工况源端平均权重"] = matrix @ np.average(fractions(c), axis=0, weights=sw)
            dynamic = np.sum(fractions(context) * matrix, axis=1)
            pred[PRIMARY] = (dynamic + matrix @ fixed) / 2
            scores[s] = issue(DEST, s, ids, pred, truth)
        means = {k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in pred}
        controls = [previous["decision"]["最强固定"], ANCHOR, *POINT_NAMES, "六工况固定均值", "工况覆盖固定公式",
                    "工况分歧固定公式", "覆盖和分歧乘积", "旧三段与覆盖分歧各半", "六工况与旧三段及论文固定组合", "工况源端平均权重"]
        fixed = min(controls, key=means.get)
        paper = min((previous["decision"]["同工具论文"], PAPER, "六工况与旧三段及论文固定组合"), key=means.get)
        gain = 1 - means[PRIMARY] / min(means[fixed], means[paper])
        decision = {"主候选": PRIMARY, "固定": fixed, "同工具论文": paper, "最小降幅": gain, "进独立":gain >= .01}
        atomic_json(DEST/"scores.json", {"sites":scores, "means":means, "decision":decision, "权重SHA256":file_hash(weights)})
        print(json.dumps({"等机队误差":{k:means[k] for k in (PRIMARY, fixed, paper, "覆盖和分歧乘积", ANCHOR)}, "判定":decision},ensure_ascii=False))


if __name__ == "__main__":
    run()
