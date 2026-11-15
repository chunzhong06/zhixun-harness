"""用同趟飞行的前、中、后半段复查序列预测稳定性，在整机队留出上比较固定和现场做法。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from score_flight_complement import (BASE, NAMES, history, receipt, digest, measures,
                                     LocalAcceptance, LocalStore, atomic_json, file_hash)
from score_flight_load_neighbors import batch_sequence, call_paper, dataset_path, signals
from harness.methods.flight_complement import fixed_mix

DEST = BASE / "flight-window-stability"
WINDOWS = ("整趟序列", "前半序列", "中间半程", "后半序列")


def window_inputs(rows):
    width = max(1, (len(rows) + 1) // 2)
    starts = (0, (len(rows) - width) // 2, len(rows) - width)
    return [batch_sequence(rows), *(batch_sequence(rows[i:i + width]) for i in starts)]


def read_windows(cohorts):
    """各窗口用同一已发布论文权重；原全程输出须复现以前回执。"""
    paper = BASE / "flight-load-neighbors/paper"
    if file_hash(paper / "paper.pt") != json.loads((paper / "scores.json").read_bytes())["model"]:
        raise ValueError("论文模型已变")
    result = {}
    with TemporaryDirectory(dir=DEST) as tmp:
        inp, out = Path(tmp) / "inputs.npy", Path(tmp) / "pred.npy"
        for site, (keys, pred, _, _) in cohorts.items():
            with h5py.File(dataset_path(site)) as data:
                flights = signals(data, "test")
            if set(flights) != set(keys):
                raise ValueError("模型读数与回执飞行未对齐")
            np.save(inp, np.stack([v for k in keys for v in window_inputs(flights[k][1])]))
            call_paper(["predict", str(inp), str(out), str(paper / "paper.pt")])
            windows = np.load(out).reshape(len(keys), len(WINDOWS))
            np.testing.assert_allclose(windows[:, 0], pred[:, -1], atol=2e-5, rtol=1e-6)
            result[site] = np.column_stack((pred[:, :-1], windows))
    return result


class WindowPolicy:
    def __init__(self, x, y, sites):
        self.weights = fixed_mix(x, y, sites)
        baseline = x @ self.weights
        features = self.features(x)
        sample_weights = np.zeros(len(x))
        for name in np.unique(sites):
            mask = sites == name
            sample_weights[mask] = len(x) / (mask.sum() * len(np.unique(sites)))
        # 同一答案的多次观测既可能揭示可靠程度，也可能暴露同向偏差，故分别拟合两类量。
        self.errors = ExtraTreesRegressor(n_estimators=80, max_depth=3, min_samples_leaf=20,
                                          n_jobs=1, random_state=17)
        self.errors.fit(features, np.abs(x - y[:, None]), sample_weight=sample_weights)
        self.bias = Ridge(alpha=len(x) * .2)
        self.center = features.mean(axis=0)
        self.scale = features.std(axis=0).clip(.25)
        self.bias.fit(self.standard(features), y - baseline, sample_weight=sample_weights)
        self.static = Ridge(alpha=len(x), fit_intercept=True).fit(x, y, sample_weight=sample_weights)
        self.static_tree = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=20,
                                              n_jobs=1, random_state=17)
        self.static_tree.fit(np.column_stack((x, features)), y - baseline, sample_weight=sample_weights)
        self.mean_weight = np.average(self.confidence(features), axis=0, weights=sample_weights)

    @staticmethod
    def features(x):
        windows = x[:, len(NAMES):]
        differences = windows[:, 1:] - windows[:, :1]
        forest = x[:, :len(NAMES)]
        return np.column_stack((windows[:, 0], forest.mean(axis=1), differences, np.abs(differences),
                                windows.std(axis=1), forest.std(axis=1),
                                windows[:, 0] - forest.mean(axis=1), forest[:, 0] - forest[:, 2]))

    def standard(self, features):
        return np.clip((features - self.center) / self.scale, -3, 3)

    def confidence(self, features):
        scores = 1 / np.maximum(1, self.errors.predict(features)) ** 2
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, x):
        features = self.features(x)
        base = x @ self.weights
        confidence = (x * self.confidence(features)).sum(axis=1)
        correction = self.bias.predict(self.standard(features))
        result = {"九工具固定最优": base, "九工具固定线性": self.static.predict(x),
                  "固定残差森林": np.maximum(0, base + self.static_tree.predict(np.column_stack((x, features)))),
                  "全程与三窗固定各半": .5 * (x[:, 5] + x[:, 6:].mean(axis=1)),
                  "窗口估错平均分量": x @ self.mean_weight, "窗口估错调权": confidence,
                  "窗口差异校正": np.maximum(0, base + correction),
                  "三个半程均值": x[:, 6:].mean(axis=1)}
        result.update(dict(zip(WINDOWS, x[:, 5:].T, strict=True)))
        return result


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有开发验收")
        cohorts = history()
        matrices = read_windows(cohorts)
        old = BASE / "flight-sequence-complement"
        previous = json.loads((old / "scores.json").read_bytes())["sites"]
        receiver = LocalAcceptance()
        receiver.output = DEST / "accepted"
        summary = {}
        for held, (keys, _, truth, _) in cohorts.items():
            others = [site for site in cohorts if site != held]
            x = np.vstack([matrices[s] for s in others])
            y = np.concatenate([cohorts[s][2] for s in others])
            sites = np.concatenate([np.repeat(s, len(cohorts[s][2])) for s in others])
            policy = WindowPolicy(x, y, sites)
            control = receipt(old, held, previous[held])
            if control["ids"] != [list(k) for k in keys]:
                raise ValueError("上次同预算对照顺序不同")
            outputs = dict(control["predictions"])
            outputs.update({k: v.tolist() for k, v in policy.predict(matrices[held]).items()})
            payload = {"ids": control["ids"], "predictions": outputs}
            key = digest([held, payload])
            if not receiver.submit(key, payload)["accepted"]:
                raise ValueError("稳定性留组预测未发出")
            observed = receipt(DEST, held, {"request": key})["predictions"]
            owner = np.array([k[0] for k in keys])
            scores = {k: measures(truth, np.array(v), owner) for k, v in observed.items()}
            summary[held] = {"request": key, "scores": scores, "rows": len(keys)}
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in summary.values()])) for k in scores}
        choices = ("窗口估错调权", "窗口差异校正")
        dynamic = {*choices, "航班互补调权", "现场与六工具固定各半", "前次现场规则"}
        best = min(choices, key=means.get)
        best_fixed = min((k for k in means if k not in dynamic), key=means.get)
        gain = 1 - means[best] / means[best_fixed]
        decision = {"候选": best, "最强固定": best_fixed, "开发降幅": gain,
                    "值得独立验证": gain >= .01 and means[best] < means["固定用论文"]}
        atomic_json(DEST / "scores.json", {"sites": summary, "means": means, "decision": decision})
        print(json.dumps({"整机队留出误差": means, "取舍": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
