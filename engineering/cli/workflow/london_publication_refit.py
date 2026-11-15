"""选案结束后用已到期参考建立发布权重；旧交付不回写，所有固定对手同样补训。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow.london_midstream import identity, selected_rows
from engineering.cli.study.verify_generated_delivery import replay
from engineering.cli.study.verify_generated_probe import predict, same
from harness.integrations.generated_regressor import load_module
from harness.integrations.recipe_regressor import code_inputs
from harness.models.sparse_reference import correct, setting_of
from harness.models.station_residual import error, gate
from harness.runtime.storage import LocalStore, atomic_json


def read(path): return json.loads(path.read_bytes())


def publication_rows(labels, maintenance_hour):
    """时点恰为维护开始的答案仍未到期，不能加入发布训练。"""
    return (np.arange(len(labels)) < maintenance_hour) & np.isfinite(labels)


def main(argv):
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--prior-run", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    source = read(args.run / "scores.json"); prior = read(args.prior_run / "scores.json")
    for folder in (args.run, args.prior_run):
        if not read(folder / "audit.json")["all_fixed_and_scores_verified"]: raise ValueError("父交付未核验")
    if source["baseline"] != prior["baseline"] or source["start"] != prior["start"]:
        raise ValueError("来源不是同一交付起点")
    scene.configure(source["start"]); tools = scene.prepare(args.archive, True, True)
    features, names = code_inputs(tools)
    plans = read(args.prior_run / "plan-bank.json") | read(args.run / "plan-bank.json")
    baseline = Path(source["baseline"]); all_issued = {}; verified_models = 0; attempts = 0; failures_total = 0
    with LocalStore(args.output).exclusive():
        if (args.output / "plans.json").exists(): raise ValueError("不可覆盖发布实测")
        atomic_json(args.output / "plans.json", plans)
        for site in source["counts"]:
            dest = args.output / site; dest.mkdir()
            truth = scene.readings(args.archive, site).to_numpy()
            old_y, _, _, service, returned = scene.roles(truth)
            at = int(service[0]) + 28 * 24; labels = np.r_[old_y[:service[0]], returned]
            fit = publication_rows(labels, at)
            history = np.arange(at - 21 * 24 - 6, at); future = np.arange(at, len(labels))
            with np.load(baseline / site / "best-fixed.npz") as z: original = z["prediction"].copy()
            # 旧试训和原门槛只决定是否采用；发布补训的训练内误差不参与重新选案。
            historical = {}
            for old in (args.prior_run, args.run):
                for p in (old / site).rglob("training.json"):
                    record = read(p); historical[record["identity"]] = record["history_rmse"]
            parent_errors = read(args.run / site / "timing.json")["parent_errors"]
            cache = {}; cache_errors = {}; predictions = {}; plan_records = {}; failed = {}
            issued = {"baseline": original.copy()}
            for key, entry in plans.items():
                plan = entry["plan"]
                try:
                    selected = selected_rows(plan, fit, features, names)
                    total = np.zeros(len(future)); unit_ids = []
                    for component in plan["components"]:
                        code = component["python_source"]
                        digest = sha256(code.encode() + features[selected].tobytes() + labels[selected].tobytes()).hexdigest()
                        if digest in cache_errors: raise ValueError(cache_errors[digest])
                        if digest not in cache:
                            attempts += 1; folder = dest / "components" / digest; folder.mkdir(parents=True)
                            data = folder / "train-input.npz"
                            np.savez_compressed(data, train_x=features[selected], train_y=labels[selected],
                                                train_ids=np.flatnonzero(selected), predict_x=features[history], predict_ids=history)
                            limit = 1000. * max(1., float(np.max(np.abs(labels[selected]))))
                            try:
                                ids, hist = fit_source(code, data, folder / "fit")
                                if not np.array_equal(ids, history): raise ValueError("发布历史身份不同")
                                with np.load(folder / "fit/base.npz") as z:
                                    if np.max(np.abs(z["raw_prediction"])) > limit: raise ValueError("发布历史预测失稳")
                                future_input = folder / "future-input.npz"
                                np.savez_compressed(future_input, predict_x=features[future], predict_ids=future,
                                                    fallback=original[28*24:], prediction_limit=limit)
                                weights = folder / "fit/model.joblib"; before = sha256(weights.read_bytes()).hexdigest()
                                ids, values = fit_source(code, future_input, folder / "service", weights)
                                if not np.array_equal(ids, future) or before != sha256(weights.read_bytes()).hexdigest():
                                    raise ValueError("发布推理更换权重或时钟")
                                # 独立重载每份共享权重，验证缓存与逐行真实预测，而非重复训练。
                                load_module(folder / "fit/source.py"); model = joblib.load(weights)
                                _, checked_history, bad = predict(model, features[history], limit, np.zeros(len(history)))
                                if bad: raise ValueError("发布历史重载失败")
                                model = joblib.load(weights)
                                _, checked_future, _ = predict(model, features[future], limit, original[28*24:])
                                same(hist, checked_history, "发布历史重构不同"); same(values, checked_future, "发布逐行重构不同")
                                cache[digest] = (np.r_[hist, values], np.r_[checked_history, checked_future])
                                verified_models += 1
                                atomic_json(folder / "identity.json", {"source_sha256": sha256(code.encode()).hexdigest(),
                                    "weights_sha256": before, "training_rows": int(selected.sum()),
                                    "last_training_hour": int(np.flatnonzero(selected)[-1]), "maintenance_hour": at,
                                    "independently_reloaded": True})
                            except (ValueError, KeyError, TypeError, SyntaxError, subprocess.TimeoutExpired) as exc:
                                cache_errors[digest] = str(exc); raise
                        base, checked = cache[digest]; clock = np.r_[history, future]
                        known = labels[clock]; setting = setting_of(component["feedback"])
                        sequence = correct(base[:, None], known, clock, setting)[:, 0]
                        same(sequence, replay(checked, clock, known, setting), "发布反馈顺序不同")
                        total += component["weight"] * sequence[len(history):]; unit_ids.append(digest)
                    predictions[key] = total
                    direct = np.r_[original[:28*24], total]
                    accepted = key in historical and gate(np.asarray(parent_errors), np.asarray(historical[key]))
                    issued["refit-direct-" + key] = direct
                    issued["refit-gated-" + key] = direct if accepted else original.copy()
                    plan_records[key] = {"plan": plan, "component_ids": unit_ids, "gate_from_original_trial": bool(accepted)}
                except (ValueError, KeyError, TypeError, SyntaxError, subprocess.TimeoutExpired) as exc:
                    failed[key] = str(exc); failures_total += 1
                    predictions[key] = original[28*24:].copy()
                    issued["refit-direct-" + key] = original.copy(); issued["refit-gated-" + key] = original.copy()
            for arm in ("project", "paper"):
                decision = read(args.run / site / (arm + "-decision.json"))
                chosen = read(args.run / site / decision["nominated"] / "plan.json"); key = identity(chosen)
                direct = np.r_[original[:28*24], predictions[key]]
                issued[arm + "-refit-gated"] = direct if decision["adopted"] else original.copy()
                issued[arm + "-refit-direct"] = direct
                with np.load(args.run / site / "issued.npz") as z: issued[arm + "-original"] = z[arm].copy()
            for tag, folder in (("previous", args.prior_run), ("current", args.run)):
                with np.load(folder / site / "fixed-issued.npz") as z:
                    same(z["ids"], service, "既有固定时钟不同")
                    for key in z.files:
                        if key != "ids": issued[tag + "-" + key] = z[key].copy()
            for value in issued.values(): same(value[:28*24], original[:28*24], "回写了已经交出的前半段")
            atomic_json(dest / "publication-plans.json", plan_records); atomic_json(dest / "failures.json", failed)
            np.savez_compressed(dest / "issued.npz", ids=service, **issued)
            all_issued[site] = issued
            print(json.dumps({"station": site, "unique_component_fits": len(cache), "failed_plans": len(failed)}, ensure_ascii=False), flush=True)
        atomic_json(args.output / "predictions-complete.json", {"done": True})
        scores = {}
        for site, issued in all_issued.items():
            truth = scene.readings(args.archive, site).to_numpy(); _, _, _, service, _ = scene.roles(truth)
            valid = np.isfinite(truth[service])
            for key, value in issued.items(): scores.setdefault(key, {})[site] = error(truth[service][valid], value[valid])
        means = {key: float(np.mean(list(value.values()))) for key, value in scores.items()}
        fixed = [k for k in means if k.startswith(("refit-", "previous-", "current-")) or k == "baseline"]
        best = min(fixed, key=means.get)
        result = {"scope": "已见五站开发，提案与选案冻结后补入维护时点前全部合法参考；全部旧新固定工具同样补训",
                  "source_run": str(args.run), "prior_run": str(args.prior_run), "macro_rmse": means,
                  "station_rmse": scores, "best_fixed": best, "plan_count": len(plans),
                  "unique_fits_attempted": attempts, "models_independently_reloaded": verified_models,
                  "failed_plan_count": failures_total, "new_driver_requests": 0,
                  "fit_cutoff_verified": True, "feedback_and_first_half_verified": True}
        atomic_json(args.output / "scores.json", result)
        print(json.dumps({"项目确认后补训": means["project-refit-gated"], "成熟机制确认后补训": means["paper-refit-gated"],
                          "项目提名直接补训": means["project-refit-direct"], "成熟机制提名直接补训": means["paper-refit-direct"],
                          "同信息最强固定": means[best], "实际独立训练数": attempts, "独立重载数": verified_models}, ensure_ascii=False))
