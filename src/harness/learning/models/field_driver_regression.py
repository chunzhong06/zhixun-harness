"""从到期参考和真实试训误差产生下一次现场提案。确认资料只用于一次发布决定。"""
from dataclasses import asdict, dataclass
from itertools import product
import json

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness.adapters.providers.pytrio_planner import ChatRequest
from harness.adapters.storage.local import atomic_json


@dataclass(frozen=True)
class Recipe:
    learner: str = "ridge"
    days: int = 56
    extra: str = "none"
    use: str = "none"
    threshold: float = 0.0


def fixed_recipes(features):
    for learner, days, extra in product(("ridge", "trees"), (14, 28, 56, 365), ("none", *features)):
        yield Recipe(learner, days, extra)


def transform(values, means, scales, names, extra):
    if extra == "none":
        return values
    # 额外一列是选定传感器的平方，使用拟合行定义中心和尺度。
    i = names.index(extra)
    return np.column_stack((values, ((values[:, i] - means[i]) / scales[i]) ** 2))


def fit_recipe(recipe, x, y, days, now, names):
    # 以合法拟合池末日为窗口终点；探索与确认两周已从池中移出。
    keep = days >= min(now, int(np.max(days)) + 1) - recipe.days
    if recipe.use != "none":
        col, direction = recipe.use.split(":")
        values = x[:, names.index(col)]
        keep &= (values <= recipe.threshold if direction == "low" else values >= recipe.threshold)
    if keep.sum() < 24:
        raise ValueError("筛后可用的拟合行少于24")
    inputs = x[keep]
    means = np.nan_to_num(np.nanmean(inputs, axis=0))
    std = np.nan_to_num(np.nanstd(inputs, axis=0))
    scales = np.where(std > 1e-8, std, 1.0)
    model = (Ridge(alpha=3) if recipe.learner == "ridge" else
             ExtraTreesRegressor(n_estimators=48, min_samples_leaf=4, random_state=17, n_jobs=1))
    pipe = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), model)
    pipe.fit(transform(inputs, means, scales, names, recipe.extra), y[keep])
    return recipe.extra, means, scales, pipe


def predict(model, x, names):
    extra, means, scales, predictor = model
    return predictor.predict(transform(x, means, scales, names, extra))


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def diagnostics(x, y, pred, names):
    facts = {}
    for i, name in enumerate(names):
        vals = x[:, i]
        if np.isfinite(vals).sum() < 4:
            continue
        median = float(np.nanmedian(vals))
        groups = {}
        for key, mask in (("low", vals <= median), ("high", vals > median)):
            if mask.sum() >= 3:
                groups[key] = {"count": int(mask.sum()), "prediction_minus_answer":
                               round(float(np.mean(pred[mask] - y[mask])), 3),
                               "rmse": round(rmse(y[mask], pred[mask]), 3)}
        facts[name] = {"median": round(median, 4), **groups}
    return facts


def schema(names, paper):
    def choices(items):
        return {"type": "string", "enum": list(items)}
    common = {"hypothesis": {"type": "string"}, "learner": choices(["ridge", "trees"]),
              "history_days": {"type": "integer", "enum": [14, 28, 56, 365]},
              "extra": choices(["none"] if paper else ["none", *names]),
              "filter": choices(["none", *(f"{n}:{g}" for n in names for g in ("low", "high"))]),
              "threshold": {"type": "number"}}
    if paper:
        common["learner"] = choices(["ridge"])
        common["history_days"] = {"type": "integer", "enum": [365]}
    return {"type": "object", "properties": common, "required": list(common), "additionalProperties": False}


def decode(action, spec):
    if not isinstance(action, dict) or set(action) != set(spec["required"]):
        raise ValueError("提案缺少字段")
    for key, field in spec["properties"].items():
        if "enum" in field and action[key] not in field["enum"]:
            raise ValueError("提案越出工具范围")
    threshold = action["threshold"]
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not str(action["hypothesis"]).strip():
        raise ValueError("原因或阈值无效")
    return Recipe(action["learner"], action["history_days"], action["extra"], action["filter"],
                  float(threshold) if action["filter"] != "none" else 0.0)


def maintain(inbox, active, x, y, days, masks, now, names, folder, paper, history):
    fit, explore, confirm = masks
    parent = predict(active, x[explore], names)
    facts = {"task_id": "paper" if paper else "project",
             "task": "Predict CO concentration from eight instrument readings. Lower RMSE is better.",
             "elapsed_day": int(now), "fit_rows": int(fit.sum()), "explore_rows": int(explore.sum()),
             "parent_rmse": round(rmse(y[explore], parent), 4),
             "observed_errors": diagnostics(x[explore], y[explore], parent, names),
             "previous_decisions": history[-2:], "trials": []}
    prompt = ("You maintain a field regression model. Diagnose a concrete error from the supplied evidence "
              "and return one testable training recipe. In hypothesis use one short sentence. "
              "filter low/high retains rows below/above your threshold, none retains all. "
              "History uses recent days. Baseline and follow-up are compared on identical exploration rows. "
              "After a trial, use its measured outcome to propose a distinct follow-up. ")
    prompt += ("Use the Self-Healing ML strategy: propose a condition selecting training rows and test it "
               "on held-back recent observations; keep the learner, features and history fixed." if paper else
               "An extra feature adds the squared standardized reading. Choose any allowed combination "
               "of training rows, predictor and feature. Do not assume a specific fix is needed.")
    spec = schema(names, paper)
    facts["schema"] = spec
    trials = {}
    for turn in (1, 2):
        facts["batch_id"] = f"day-{now}-trial-{turn}"
        messages = [{"role":"system","content":prompt},
                    {"role":"user","content":json.dumps(facts, ensure_ascii=False, allow_nan=False)}]
        body = {"messages":messages, "model":"gpt-6-luna"}
        reply = inbox.send(ChatRequest("local-inbox", body, 120))
        action = json.loads(reply.body["choices"][0]["message"]["content"])
        event = {"trial": turn, "action": action}
        try:
            recipe = decode(action, spec)
            if recipe in trials:
                event["outcome"] = "same experiment"
            else:
                candidate = fit_recipe(recipe, x[fit], y[fit], days[fit], now, names)
                pred = predict(candidate, x[explore], names)
                loss = rmse(y[explore], pred)
                trials[recipe] = (loss, candidate)
                event.update(outcome="tested", rmse=round(loss, 5),
                             mean_prediction_minus_answer=round(float(np.mean(pred - y[explore])), 4))
        except ValueError as error:
            event.update(outcome="not built", reason=str(error))
        facts["trials"].append(event)
    decision = {"day": int(now), "trials": facts["trials"], "adopted": False}
    if trials:
        winner = min(trials, key=lambda key: trials[key][0])
        candidate = trials[winner][1]
        before = predict(active, x[confirm], names)
        after = predict(candidate, x[confirm], names)
        before_loss, after_loss = rmse(y[confirm], before), rmse(y[confirm], after)
        decision.update(nominated=asdict(winner), parent_confirm=before_loss, trial_confirm=after_loss)
        if trials[winner][0] < rmse(y[explore], parent) and after_loss < before_loss:
            active = candidate
            decision["adopted"] = True
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "decision.json", decision)
    joblib.dump(active, folder / "issued.joblib")
    return joblib.load(folder / "issued.joblib"), decision
