"""检验同航班标签监督短航段的训练方式，在六个已见开发机队轮流交付、核验。"""
import json

import h5py
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from score_flight_life_pairs import BASE, PREP, SITES, dataset_path, signals, read_flights, file_hash
from score_flight_complement import receipt, measures, LocalAcceptance, LocalStore, digest, atomic_json
from harness.methods.flight_complement import fixed_mix
from harness.models.flight_segment_training import SegmentLearner, segments

DEST = BASE / "flight-segment-training"
PRIOR = BASE / "flight-matched-pairs"
TOOLS = ("前段独立估计", "中段独立估计", "末段独立估计", "共享线性三段均值", "整趟均值训练")
OLD_TOOLS = ("偏差波动森林", "三段联合森林", "整趟序列", "前半序列", "中间半程", "后半序列")
PRIMARY = "按片段一致程度估错"


def load_site(site, subset, prep):
    with h5py.File(dataset_path(site)) as f:
        flights = signals(f, subset)
        truth = read_flights(f, subset, answers=True)
    keys = sorted(flights)
    if set(keys) != set(truth):
        raise ValueError("飞行身份与寿命答案不同")
    return keys, np.stack([segments(prep, flights[k][1]) for k in keys]), np.array([truth[k] for k in keys])


def reliability(matrix):
    early = matrix[:, :3]
    return np.column_stack((early.mean(1), early.std(1), early - early.mean(1)[:, None],
                            matrix[:, 3:5], matrix[:, 7] - early.mean(1),
                            matrix[:, 8:] - matrix[:, 7:8]))


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有同协议开发交付")
        prep = joblib.load(PREP).prep
        history = [load_site(site, "dev", prep) for site in SITES]
        source = np.concatenate([item[1] for item in history])
        y = np.concatenate([item[2] for item in history])
        learner = SegmentLearner().fit(source, y)
        joblib.dump(learner, DEST / "segment.joblib")
        prior = json.loads((PRIOR / "scores.json").read_bytes())
        records = {}
        for site, item in prior["sites"].items():
            keys, views, truth = load_site(site, "test", prep)
            old = receipt(PRIOR, site, item)
            if [list(k) for k in keys] != old["ids"]:
                raise ValueError("同航班开发预测顺序改变")
            pred = dict(old["predictions"])
            generated = learner.outputs(views)
            for name, values in zip(TOOLS, generated.T, strict=True):
                pred[name] = values.clip(0).tolist()
            pred["三段固定等权"] = generated[:, :3].mean(1).clip(0).tolist()
            pred["论文与三段固定各半"] = ((np.asarray(pred["整趟序列"]) + generated[:, :3].mean(1)) / 2).clip(0).tolist()
            matrix = np.column_stack([pred[k] for k in (*TOOLS, *OLD_TOOLS)])
            records[site] = (keys, matrix, truth, pred)
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        scores = {}
        for site, (ids, matrix, truth, pred) in records.items():
            refs = {s: v for s, v in records.items() if s != site}
            train = np.vstack([v[1] for v in refs.values()])
            labels = np.concatenate([v[2] for v in refs.values()])
            sites = np.concatenate([np.repeat(s, len(v[2])) for s, v in refs.items()])
            sw = np.array([len(sites) / (len(refs) * len(refs[k][2])) for k in sites])
            fixed = fixed_mix(train, labels, sites)
            model = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=24, random_state=17, n_jobs=1)
            model.fit(reliability(train), abs(train - labels[:, None]), sample_weight=sw)
            def weights(values):
                inverse = np.maximum(1., model.predict(reliability(values))) ** -2
                return inverse / inverse.sum(axis=1, keepdims=True)
            pred[PRIMARY] = (weights(matrix) * matrix).sum(1).tolist()
            pred["现场分量冻结为源均值"] = (matrix @ np.average(weights(train), axis=0, weights=sw)).tolist()
            pred["十一工具固定最优"] = (matrix @ fixed).tolist()
            pred["现场与十一工具各半"] = ((np.array(pred[PRIMARY]) + pred["十一工具固定最优"]) / 2).tolist()
            payload = {"ids": [list(k) for k in ids], "predictions": pred}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("航段开发回执不成功")
            sent = receipt(DEST, site, {"request": key})["predictions"]
            owners = np.array([k[0] for k in ids])
            scores[site] = {"request": key, "rows": len(ids),
                            "scores": {k: measures(truth, np.array(v), owners) for k, v in sent.items()}}
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in pred}
        dynamic = {PRIMARY, "现场与十一工具各半", "差分现场估错", "差分现场与固定各半", "可比工况共同误差",
                   "窗口估错调权", "窗口差异校正", "前次现场规则", "现场与六工具固定各半", "航班互补调权"}
        fixed = min((k for k in means if k not in dynamic), key=means.get)
        gain = 1 - means[PRIMARY] / means[fixed]
        decision = {"主候选": PRIMARY, "最强固定": fixed, "相对固定降幅": gain,
                    "进入独立": gain >= .01 and means[PRIMARY] <= .99 * means["整趟序列"]}
        atomic_json(DEST / "scores.json", {"sites": scores, "means": means, "decision": decision,
                    "源端航班数": len(y), "model": file_hash(DEST / "segment.joblib")})
        print(json.dumps({"等机队误差": {k: means[k] for k in (PRIMARY, fixed, *TOOLS, "三段固定等权", "整趟序列")},
                          "判定": decision}, ensure_ascii=False))


if __name__ == "__main__":
    run()
