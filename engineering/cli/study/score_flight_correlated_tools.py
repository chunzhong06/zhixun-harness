"""在可比飞行工况中测量模型是否一起犯错，再交接整组留出航班的真实预测。"""
import json

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from score_flight_complement import BASE, NAMES, history, receipt, digest, measures, LocalAcceptance, LocalStore, atomic_json
from score_flight_window_stability import WINDOWS, dataset_path, signals

DEST = BASE / "flight-correlated-tools"


def simplex(v):
    """逐行投影到非负且和为一的权重，保留只用一个工具的端点。"""
    ordered = -np.sort(-v, axis=1)
    levels = (np.cumsum(ordered, axis=1) - 1) / np.arange(1, v.shape[1]+1)
    active = (ordered > levels).sum(axis=1).clip(1) - 1
    return np.maximum(0, v - levels[np.arange(len(v)), active, None])


def allocations(grams):
    # 加权二次误差是半正定的。小幅等权收缩避免同源工具在近乎奇异的局部矩阵上跳动。
    n, count, _ = grams.shape
    penalty = .025 * np.trace(grams, axis1=1, axis2=2) / count
    grams = grams + penalty[:, None, None] * np.eye(count)
    step = 1 / np.linalg.eigvalsh(grams)[:, -1].clip(1e-9)
    weights = np.full((n, count), 1 / count)
    momentum = weights.copy()
    for iteration in range(100):
        gradient = np.einsum("nij,nj->ni", grams, momentum)
        new = simplex(momentum - step[:, None] * gradient)
        momentum = new + iteration / (iteration + 3) * (new - weights)
        weights = new
    return weights


def context(flights, keys, predictions):
    operating = []
    for key in keys:
        settings = flights[key][1][:, 14:]
        operating.append(np.r_[np.mean(settings, axis=0), np.std(settings, axis=0)])
    trees, windows = predictions[:, :len(NAMES)], predictions[:, len(NAMES):]
    return np.column_stack((operating, trees.mean(axis=1), trees.std(axis=1), windows[:, 0],
                            windows[:, 1:] - windows[:, :1], windows[:, 0] - trees.mean(axis=1)))


class CorrelatedTools:
    def __init__(self, train):
        contexts = np.vstack([c for c, _, _, _ in train])
        self.scale = RobustScaler().fit(contexts)
        self.groups = []
        for ctx, forecasts, answers, owners in train:
            residuals = forecasts - answers[:, None]
            for unit in np.unique(owners):
                pick = owners == unit
                errors = residuals[pick]
                gram = np.einsum("ni,nj->nij", errors, errors)
                self.groups.append((self.transform(ctx[pick]), gram, 1 / (len(train) * len(np.unique(owners)))))
        self.global_gram = sum(weight * gram.mean(axis=0) for _, gram, weight in self.groups)
        self.global_weights = allocations(self.global_gram[None, :, :])[0]

    def transform(self, ctx):
        return np.clip(self.scale.transform(ctx), -4, 4)

    def weights(self, ctx):
        target = self.transform(ctx)
        local = np.zeros((len(ctx), *self.global_gram.shape))
        for source, gram, share in self.groups:
            distance = cdist(target, source, metric="sqeuclidean") / source.shape[1]
            count = min(12, source.shape[0])
            indices = np.argpartition(distance, count - 1, axis=1)[:, :count]
            local += share * gram[indices].mean(axis=1)
        return allocations(.5 * local + .5 * self.global_gram)


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("协同误差实验已经发出结果")
        cohorts = history()
        prior_dir = BASE / "flight-window-stability"
        prior = json.loads((prior_dir / "scores.json").read_bytes())["sites"]
        records, payloads = {}, {}
        for site, (keys, _, truth, _) in cohorts.items():
            payload = receipt(prior_dir, site, prior[site])
            if payload["ids"] != [list(k) for k in keys]:
                raise ValueError("相同工具没有对齐同一航班")
            preds = np.column_stack([payload["predictions"][k] for k in (*NAMES, *WINDOWS)])
            with h5py.File(dataset_path(site)) as f:
                flights = signals(f, "test")
            records[site] = (context(flights, keys, preds), preds, truth, np.array([u for u, _ in keys]))
            payloads[site] = payload
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        summary = {}
        for site, (ctx, preds, truth, owners) in records.items():
            train = [v for k, v in records.items() if k != site]
            rule = CorrelatedTools(train)
            weights = rule.weights(ctx)
            # 固定化对照用五个历史机队自身测量求平均，不读取当前留出机队来调整常数。
            constant = np.mean([rule.weights(v[0]).mean(axis=0) for v in train], axis=0)
            observations = dict(payloads[site]["predictions"])
            observations.update({"可比工况共同误差": np.sum(weights * preds, axis=1).tolist(),
                                 "共同误差固定平均分量": (preds @ constant).tolist(),
                                 "全局共同误差分量": (preds @ rule.global_weights).tolist()})
            payload = {"ids": payloads[site]["ids"], "predictions": observations}
            key = digest([site, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("协同误差预测未交接")
            actual = receipt(DEST, site, {"request": key})["predictions"]
            table = {k: measures(truth, np.array(v), owners) for k, v in actual.items()}
            summary[site] = {"request": key, "scores": table, "rows": len(owners)}
        means = {k: float(np.mean([x["scores"][k]["rmse"] for x in summary.values()])) for k in table}
        candidate = "可比工况共同误差"
        dynamics = {candidate, "窗口估错调权", "窗口差异校正", "前次现场规则", "现场与六工具固定各半", "航班互补调权"}
        fixed = min((k for k in means if k not in dynamics), key=means.get)
        gain = 1 - means[candidate] / means[fixed]
        decision = {"最强固定": fixed, "开发降幅": gain, "值得独立验证": gain >= .01 and means[candidate] < means["整趟序列"]}
        atomic_json(DEST / "scores.json", {"sites": summary, "means": means, "decision": decision})
        print(json.dumps({"整机队留出误差": means, "取舍": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
