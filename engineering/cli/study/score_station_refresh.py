"""在两个已开分的伦敦批次中固定现场提案，只改变何时补训。"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow import london_field_driver as scene
from harness.models.sparse_reference import SETTINGS, choose_from_feedback, correct, name, setting_of, window_scores
from harness.models.station_residual import Rule, error, tests
from harness.models.station_refresh import prediction_bank, correct_versions
from harness.runtime.storage import LocalStore, atomic_json


def read(path):
    return json.loads(path.read_bytes())


def main():
    base = ROOT / "results/experiments/local-defaults/whole-maintenance"
    out = base / "london-training-recency"
    cases = (("2023", "london-field-reference", "london-air"),
             ("2024", "london-adaptive-recovery", "london-air-2024"))
    with LocalStore(out).exclusive():
        if (out / "scores.json").exists():
            raise ValueError("已见资料的配对实训不可覆盖")
        report = {}
        for year, experiment, source in cases:
            original = base / experiment
            old = read(original / "scores.json")
            scene.configure(old["start"])
            archive = ROOT / "data/public" / source / "stations.zip"
            tools = scene.prepare(archive, nonlinear=old.get("adaptive", False))
            rules = list(dict.fromkeys(Rule(**r) for r in old["recipes"] + old["extra"]))
            results = {}
            details = {}
            for site in old["counts"]:
                truth = scene.readings(archive, site).to_numpy()
                y, fit, windows, service, returned = scene.roles(truth)
                parent = tools.train(tools.static, fit, y)
                old_error = tests(tools, parent, windows, y)
                clock = np.arange(np.flatnonzero(windows[2])[0], len(y))
                service_rows = clock >= service[0]
                known = y.copy()
                known[service] = returned
                observed = known[clock]
                valid = np.isfinite(truth[service])
                decisions = {a: read(original / site / (a + "-decision.json")) for a in ("project", "paper")}
                proposed = {a: list(dict.fromkeys((Rule(**t["proposal"]["recipe"]), setting_of(t["proposal"]["feedback"]))
                            for t in d["trials"] if "observed_rmse" in t)) for a, d in decisions.items()}
                capture = list(dict.fromkeys(r for entries in proposed.values() for r, _ in entries))
                # 原有开场门槛只使用旧权重对三个已分离窗口的预测。
                old_matrix, _, _, _ = prediction_bank(tools, rules, fit, y, np.arange(len(y)), service[0], -1)
                accepted = {}
                for setting in SETTINGS:
                    losses = window_scores(old_matrix, windows, y, setting)
                    accepted[setting] = ((np.mean(losses[:2]**2, axis=0) < np.mean(old_error[:2]**2))
                                         & (losses[2] < old_error[2]))
                parent_base = tools.issue(parent, clock)
                parent_feedback = {s: correct(parent_base[:, None], observed, clock, s)[:, 0] for s in SETTINGS}
                for days in (-1, 0, 7, 28):
                    candidate_base, reset, journal, snapshots = prediction_bank(
                        tools, rules, fit, known, clock, service[0], days, capture)
                    streams = {s: correct_versions(candidate_base, observed, clock, s, reset) for s in SETTINGS}
                    predictions = {}
                    for arm, entries in proposed.items():
                        columns = [parent_base]
                        columns += [streams[s][:, rules.index(r)] for r, s in entries]
                        columns += [parent_feedback[s] for _, s in entries]
                        d = decisions[arm]
                        nominee = (Rule(**d["nominated"]), setting_of(d["feedback"])) if entries else None
                        initial = entries.index(nominee)+1 if d["adopted"] else 0
                        stream, choices = choose_from_feedback(np.column_stack(columns), observed, clock,
                                                              service[0], initial, 28)
                        predictions[arm] = stream[service_rows]
                        details[f"{site}/{arm}/{days}"] = {"switches": int(np.count_nonzero(np.diff(choices[service_rows])))}
                    key = f"refresh-{days}"
                    row = results.setdefault(key, {})
                    for arm, pred in predictions.items():
                        row.setdefault(arm, {})[site] = error(truth[service][valid], pred[valid])
                    for setting, stream in streams.items():
                        pred = stream[service_rows]
                        direct_errors = np.sqrt(np.mean((pred[valid] - truth[service][valid, None])**2, axis=0))
                        for i, value in enumerate(direct_errors):
                            suffix = f"{i}-{name(setting)}"
                            row.setdefault("direct-" + suffix, {})[site] = float(value)
                            row.setdefault("gated-" + suffix, {})[site] = float(value) if accepted[setting][i] else error(
                                truth[service][valid], parent_base[service_rows][valid])
                    dest = out / year / site / key
                    dest.mkdir(parents=True, exist_ok=False)
                    atomic_json(dest / "training.json", journal)
                    joblib.dump(snapshots, dest / "proposed-models.joblib")
                    np.savez_compressed(dest / "issued.npz", ids=service, **predictions)
            means = {mode: {k: float(np.mean(list(v.values()))) for k, v in rows.items()}
                     for mode, rows in results.items()}
            brief = {}
            for mode, mean in means.items():
                winner = min((k for k in mean if k.startswith(("direct-", "gated-"))), key=mean.get)
                brief[mode] = {"project": mean["project"], "paper": mean["paper"],
                               "fixed": mean[winner], "winner": winner}
            report[year] = {"summary": brief, "station_rmse": results, "macro_rmse": means,
                            "details": details, "rules": [r.__dict__ for r in rules]}
            print(json.dumps({year: brief}, ensure_ascii=False), flush=True)
        atomic_json(out / "scores.json", report)


if __name__ == "__main__":
    main()
