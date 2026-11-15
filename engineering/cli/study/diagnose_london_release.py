"""用已开分的伦敦资料检验拒绝换版、反馈修正与后续再选择各自损失多少。"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow.london_field_driver import prepare, readings, roles
from harness.models.sparse_reference import correct, setting_of, window_scores
from harness.models.station_residual import Rule, error, gate, tests
from harness.runtime.storage import LocalStore, atomic_json


def reconsider(predictions, reference, clock, service_start, initial, days):
    """回标后再选其后要用的已训练模型，仅保留过去指定天数的实际预测误差。"""
    predictions = np.asarray(predictions)
    output = np.empty(len(reference))
    chosen = initial
    history = []
    changes = 0
    for i, now in enumerate(clock):
        output[i] = predictions[i, chosen]
        if np.isfinite(reference[i]):
            history.append((now, (predictions[i] - reference[i]) ** 2))
            history = [(at, loss) for at, loss in history if at > now - days * 24]
            if now >= service_start:
                updated = int(np.argmin(np.mean([loss for _, loss in history], axis=0)))
                changes += updated != chosen
                chosen = updated
    return output, changes


def main():
    original = ROOT / "results/experiments/local-defaults/whole-maintenance/london-field-reference"
    out = original.parent / "london-release-diagnosis"
    archive = ROOT / "data/public/london-air/stations.zip"
    with LocalStore(out).exclusive():
        if (out / "scores.json").exists():
            raise ValueError("不可覆盖开发诊断")
        old = json.loads((original / "scores.json").read_bytes())
        tools = prepare(archive)
        scores = {}
        details = {}
        for site in old["counts"]:
            truth = readings(archive, site).to_numpy()
            y, fit, windows, service, returned = roles(truth)
            valid = np.isfinite(truth[service])
            full = np.arange(len(y))
            parent = tools.train(tools.static, fit, y)
            base_parent = tools.issue(parent, full)
            parent_errors = tests(tools, parent, windows, y)
            # 开场前第三周的答案已经到达，可为后续再选择提供初值。
            clock = np.arange(np.flatnonzero(windows[2])[0], len(y))
            known = np.concatenate((y[clock[clock < service[0]]], returned))
            for arm in ("project", "paper"):
                decision = json.loads((original / site / (arm + "-decision.json")).read_bytes())
                trials = [t for t in decision["trials"] if "observed_rmse" in t]
                nominee = min(trials, key=lambda t: np.mean(np.square(t["observed_rmse"])))
                models = []
                columns = [base_parent[clock]]
                calibrations = []
                for trial in trials:
                    plan = trial["proposal"]
                    model = tools.train(Rule(**plan["recipe"]), fit, y)
                    base = tools.issue(model, full)
                    setting = setting_of(plan["feedback"])
                    models.append((base, setting))
                    columns.append(correct(base[clock, None], known, clock, setting)[:, 0])
                    calibrations.append(correct(base_parent[clock, None], known, clock, setting)[:, 0])
                winner = trials.index(nominee)
                base, setting = models[winner]
                direct = correct(base[service, None], returned, service, setting)[:, 0]
                fallback = base_parent[service]
                if decision["adopted"]:
                    fallback = direct
                else:
                    # 先在两个探索窗提名旧模型的反馈配置，最后确认一次。
                    settings = [value for _, value in models]
                    losses = [window_scores(base_parent[:, None], windows, y, s)[:, 0] for s in settings]
                    candidate = int(np.argmin([np.mean(v[:2] ** 2) for v in losses]))
                    if gate(parent_errors, losses[candidate]):
                        fallback = correct(base_parent[service, None], returned, service, settings[candidate])[:, 0]
                variants = {"direct": direct, "separate_feedback": fallback}
                choices = {}
                for days in (7, 28):
                    for include_parent_feedback in (False, True):
                        matrix = np.column_stack(columns + (calibrations if include_parent_feedback else []))
                        initial = winner + 1 if decision["adopted"] else 0
                        stream, count = reconsider(matrix, known, clock, service[0], initial, days)
                        key = f"reconsider-{days}-parent-feedback-{int(include_parent_feedback)}"
                        variants[key] = stream[clock >= service[0]]
                        choices[key] = count
                for variant, prediction in variants.items():
                    scores.setdefault(arm + "/" + variant, {})[site] = error(truth[service][valid], prediction[valid])
                details[site + "/" + arm] = choices
        means = {name: float(np.mean(list(values.values()))) for name, values in scores.items()}
        result = {"scope": "已见伦敦2023开发回放，保持原真实提案，检验发布决定与后续反馈工具",
                  "station_rmse": scores, "macro_rmse": means, "switches": details,
                  "original_project": old["macro_rmse"]["project"],
                  "original_best_fixed": old["macro_rmse"][old["best_fixed"]]}
        atomic_json(out / "scores.json", result)
        print(json.dumps(means, ensure_ascii=False))


if __name__ == "__main__":
    main()
