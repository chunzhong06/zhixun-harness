"""在已见五组的同一选案资料上给固定对手调权，并封存逐航班权重的均值。"""
import json

import h5py
import joblib
import numpy as np
from scipy.optimize import minimize

from score_flight_range_choice import DEST, SITES, dataset_path, signals, read_flights, matrices
from harness.contracts import digest
from harness.models.flight_range_choice import NAMES, gate_inputs
from harness.runtime.storage import LocalStore, atomic_json


def main():
    output = DEST / "fixed-controls.json"
    with LocalStore(DEST / "fixed-lock").exclusive():
        if output.exists():
            raise ValueError("同一选案资料的固定权重已经封存")
        scores = json.loads((DEST / "scores.json").read_bytes())
        bundle = joblib.load(DEST / "range.joblib")
        batches, average_weights = [], []
        for site in SITES:
            request = scores[site]["request"]
            payload = json.loads((DEST / "accepted" / (request + ".json")).read_bytes())["payload"]
            if digest([site, payload]) != request:
                raise ValueError("开发预测回执身份改变")
            with h5py.File(dataset_path(site)) as data:
                flights = signals(data, "test")
                answers = read_flights(data, "test", answers=True)
            keys = sorted(flights)
            if [list(k) for k in keys] != payload["ids"] or set(keys) != set(answers):
                raise ValueError("固定对手使用的航班身份不同")
            prediction = np.column_stack([payload["predictions"][name] for name in NAMES])
            labels = np.asarray([answers[k] for k in keys])
            context = matrices(flights, bundle.prep)[-1]
            errors = np.maximum(1., bundle.error.predict(gate_inputs(prediction, context)))
            weights = 1 / errors ** 2
            weights /= weights.sum(axis=1, keepdims=True)
            np.testing.assert_allclose((weights * prediction).sum(axis=1), payload["predictions"]["估错软组合"])
            average_weights.append(weights.mean(axis=0))
            batches.append((prediction, labels))

        def loss(weights):
            return float(np.mean([np.sqrt(np.mean((pred @ weights - y) ** 2)) for pred, y in batches]))

        count = len(NAMES)
        equal = np.full(count, 1 / count)
        fit = minimize(loss, equal, method="SLSQP", bounds=[(0, 1)] * count,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"ftol": 1e-10, "maxiter": 250})
        if not fit.success:
            raise ValueError(fit.message)
        fitted = np.clip(fit.x, 0, 1)
        fixed = {"开发最优固定权重": fitted / fitted.sum(), "现场权重固定为历史均值": np.mean(average_weights, axis=0)}
        result = {"顺序": list(NAMES), "固定权重": {k: w.tolist() for k, w in fixed.items()},
                  "开发等机队平均误差": {k: loss(w) for k, w in fixed.items()}}
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
