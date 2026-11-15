"""UCI 空气仪现场入口。先持久化每段预测，所有维护轮完成后才合并服务答案计分。"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import zipfile

import joblib
import numpy as np
import pandas as pd

from harness.integrations.proposal_inbox import ProposalInbox
from harness.models.field_driver_regression import Recipe, fit_recipe, fixed_recipes, maintain, predict, rmse
from harness.runtime.storage import LocalStore, atomic_json


FIELDS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
ARMS = ("project", "paper")
STARTS = (180, 236, 292)


def read_archive(archive):
    with zipfile.ZipFile(archive) as z, z.open("AirQualityUCI.csv") as f:
        table = pd.read_csv(f, sep=";", decimal=",").dropna(subset=["Date", "Time"])
    clocks = pd.to_datetime(table.Date + " " + table.Time, format="%d/%m/%Y %H.%M.%S")
    if not clocks.is_monotonic_increasing or clocks.duplicated().any():
        raise ValueError("原始时间轴有重叠或倒序")
    days = (clocks.dt.normalize() - clocks.min().normalize()).dt.days.to_numpy()
    x = table[FIELDS].replace(-200, np.nan).to_numpy(dtype=float)
    y = table["CO(GT)"].replace(-200, np.nan).to_numpy(dtype=float)
    feedback = ((days < 90) | (clocks.dt.hour.to_numpy() % 6 == 0)) & np.isfinite(y)
    return x, y, days, feedback, clocks.astype(str).to_numpy(dtype=str)


def windows(days, feedback, now):
    return (feedback & (days < now - 14),
            feedback & (days >= now - 14) & (days < now - 7),
            feedback & (days >= now - 7) & (days < now))


def fixed_step(recipe, active, x, y, days, masks, now):
    fit, explore, confirm = masks
    try:
        candidate = fit_recipe(recipe, x[fit], y[fit], days[fit], now, FIELDS)
    except ValueError:
        return active
    for mask in (explore, confirm):
        if rmse(y[mask], predict(candidate, x[mask], FIELDS)) >= rmse(y[mask], predict(active, x[mask], FIELDS)):
            return active
    return candidate


def issue(path, ids, predictions):
    if len(ids) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError("不能交付缺失的预测")
    np.savez_compressed(path, ids=ids, predictions=predictions)


def score(files, answers):
    pieces = []
    for f in files:
        with np.load(f, allow_pickle=False) as packet:
            ids = packet["ids"]
            p = packet["predictions"]
        valid = np.isfinite(answers[ids])
        pieces.append((p[valid] - answers[ids][valid]) ** 2)
    return {"rmse": float(np.sqrt(np.concatenate(pieces).mean())),
            "periods": [float(np.sqrt(v.mean())) for v in pieces],
            "scored": sum(len(v) for v in pieces)}


def run(archive, out, queue):
    x, y, days, feedback, times = read_archive(archive)
    source = (days < 90) & np.isfinite(y)
    # 源模型在公开历史上训练一次，三个现场和所有固定对照使用同一对象作为起点。
    initial = fit_recipe(Recipe(days=365), x[source], y[source], days[source], 90, FIELDS)
    joblib.dump(initial, out / "initial.joblib")
    active = {arm:initial for arm in ARMS}
    history = {arm:[] for arm in ARMS}
    fixed = {r:initial for r in fixed_recipes(FIELDS)}
    controls = {r:f"fixed-{i:02d}" for i, r in enumerate(fixed)}
    inbox = ProposalInbox(queue)
    outfiles = {}
    stage_info = []
    used = set()
    for now in STARTS:
        past = days < now
        xd, yd, dd = x[past], y[past].copy(), days[past]
        # 没有到期的参考值在维护侧不可见，即使它们已经存在于公开 CSV。
        yd[~feedback[past]] = np.nan
        masks = windows(dd, feedback[past], now)
        if any(mask.sum() < 4 for mask in masks):
            raise ValueError("缺少独立时段的参考测量")
        rows = np.flatnonzero((days >= now) & (days < now + 56))
        stage = out / f"period-{now}"
        stage.mkdir(exist_ok=False)
        def work(arm):
            return maintain(inbox, active[arm], xd, yd, dd, masks, now, FIELDS,
                            stage / arm, arm == "paper", history[arm])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {arm:pool.submit(work, arm) for arm in ARMS}
            for arm, future in futures.items():
                active[arm], dec = future.result()
                history[arm].append({k:v for k,v in dec.items() if k != "trials"})
                for trial in dec["trials"]:
                    if trial.get("outcome") == "tested":
                        a = trial["action"]
                        used.add(Recipe(a["learner"],a["history_days"],a["extra"],a["filter"],
                                        float(a["threshold"]) if a["filter"] != "none" else 0.0))
        models = dict(active, static=initial)
        for recipe in fixed:
            fixed[recipe] = fixed_step(recipe, fixed[recipe], xd, yd, dd, masks, now)
            models[controls[recipe]] = fixed[recipe]
        for name, model in models.items():
            f = stage / (name + ".npz")
            issue(f, rows, predict(model, x[rows], FIELDS))
            outfiles.setdefault(name, []).append(f)
        stage_info.append({"day":now, "first":times[rows[0]], "last":times[rows[-1]],
                           "issued":len(rows), "fit_explore_confirm":[int(m.sum()) for m in masks],
                           "feedback_after_issue":int(feedback[rows].sum())})
    # 把实际提过的新配方各冻结为全程相同的工具，单独审计“固定照做”是否也能取得分差。
    extra = {}
    for i, recipe in enumerate(sorted(used - set(fixed), key=repr)):
        name = f"frozen-proposal-{i:02d}"
        extra[recipe] = name
        model = initial
        for now in STARTS:
            fit, explore, confirm = windows(days, feedback, now)
            model = fixed_step(recipe, model, x, y, days, (fit, explore, confirm), now)
            rows = np.flatnonzero((days >= now) & (days < now + 56))
            f = out / f"period-{now}" / (name + ".npz")
            issue(f, rows, predict(model, x[rows], FIELDS))
            outfiles.setdefault(name, []).append(f)
    scores = {name:score(files, y) for name, files in outfiles.items()}
    fixed_names = ["static", *controls.values()]
    best = min(fixed_names, key=lambda n:scores[n]["rmse"])
    best_shared = min([*fixed_names, *extra.values()], key=lambda n:scores[n]["rmse"])
    result = {"driver":"gpt-6-luna", "stages":stage_info, "scores":scores,
              "decisions":history, "fixed_recipes":{v:asdict(k) for k,v in (controls | extra).items()},
              "best_fixed":best, "best_including_frozen_proposals":best_shared}
    atomic_json(out / "scores.json", result)
    summary = {key:scores[n]["rmse"] for key,n in (("现场","project"),("论文适配","paper"),
               ("最强固定",best),("含现场配方的最强固定",best_shared),("原版","static"))}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(argv):
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--queue", type=Path, help="复跑时沿用带请求摘要的原有代理队列，产物写入新目录")
    args = cli.parse_args(argv)
    with LocalStore(args.output).exclusive():
        if (args.output / "initial.joblib").exists():
            raise ValueError("已运行的交付目录不可覆盖")
        run(args.archive, args.output, args.queue or args.output / "queue")
