"""按采集日交付独立来源的分类预测，每个物理暴露只贡献一个样本。"""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
from river.drift.binary import DDM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from score_author_new_arrays import IssuedForest, ROOT, ObserverErrors, atomic_json, author_definition, digest
from harness.models.gas_feedback_bank import discriminate, fit_discriminant
from harness.models.vote_feedback import majority


DATA = ROOT / "data/public/uci-home-activity/uci-362.zip"
EXPECTED = "7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-home-forward.json"
OUT = ROOT / "results/experiments/local-defaults/whole-maintenance/home-forward"
CLASSES = [1, 2, 3]
ACTIONS = ("observer", "mixed", "recent")


def load_recordings():
    if sha256(DATA.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("UCI 原件身份不同")
    cols = [f"R{i}" for i in range(1, 9)]
    with ZipFile(DATA) as outer:
        meta = pd.read_csv(outer.open("HT_Sensor_metadata.dat"), sep=r"\s+").set_index("id")
        with ZipFile(BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            with inner.open("HT_Sensor_dataset.dat") as member:
                raw = pd.read_csv(member, sep=r"\s+", usecols=["id", "time", *cols])
    if not meta.index.is_unique or set(raw.id) - set(meta.index):
        raise ValueError("原件记录身份不唯一或无元数据")
    duration = raw.id.map(meta.dt)
    obs = raw.loc[(raw.time >= 0) & (raw.time < np.minimum(5/60, duration))]
    x = obs.groupby("id")[cols].mean()
    if not np.isfinite(x).all().all():
        raise ValueError("预定时间窗含非有限特征")
    all_ids = set(raw.id)
    excluded = {int(k): "no time series" if k not in all_ids else "no exposure in frozen window"
                for k in sorted(set(meta.index) - set(x.index))}
    log = meta.loc[x.index].copy()
    log["date"] = pd.to_datetime(log.date, format="%m-%d-%y")
    log.sort_values(["date", "t0"], kind="stable", inplace=True)
    y = log["class"].map({"background": 1, "banana": 2, "wine": 3})
    if y.isna().any():
        raise ValueError("未知原始标签")
    train, days = [], []
    for day, group in log.groupby("date", sort=True):
        ids = group.index.tolist()
        if len(train) < 30:
            train.extend(ids)
        else:
            days.append((str(day.date()), ids))
    return x, y.astype(int), train, days, excluded


def macro(y, p):
    return float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))


def main():
    x, y, train, days, excluded = load_recordings()
    if set(y.loc[train]) != set(CLASSES) or not days:
        raise ValueError("冻结的开发日期不能覆盖三个源类别或没有独立服务日")
    xs, ys = x.loc[train], y.loc[train]
    names = ("current", "logistic", "forest", "lda", "svm")
    source_tree = RandomForestClassifier(n_estimators=1, random_state=0, n_jobs=1).fit(xs, ys)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced",
                                 max_iter=1000, random_state=17)).fit(xs, ys)
    forest = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=17,
                                    class_weight="balanced", n_jobs=1).fit(xs, ys)
    lda = fit_discriminant(xs.to_numpy(), ys.to_numpy())
    svm = make_pipeline(StandardScaler(), SVC(C=1., class_weight="balanced")).fit(xs, ys)
    scaler = StandardScaler().fit(xs)
    recent_tree = None
    mixed = LogisticRegression(C=1., max_iter=1000, random_state=17)
    count_fits = 0
    if max(len(ids) for _, ids in days) > 32:
        raise ValueError("需另行适配作者 DDM 的整批回标接口")
    keys = [ids for _, ids in days]
    source_history = scaler.transform(xs)
    prior_weights = compute_sample_weight("balanced", ys)
    mixed.fit(source_history, ys, sample_weight=prior_weights)
    router = ObserverErrors(names, 3)
    active = "observer"
    once = None
    seen_ids = []
    previous_predictions = {a: [] for a in ACTIONS}
    files = []
    for j, (date, ids) in enumerate(days):
        batch = x.loc[ids]
        votes = {"current": source_tree.predict(batch), "logistic": logistic.predict(batch),
                 "forest": forest.predict(batch), "lda": discriminate(lda, batch.to_numpy()), "svm": svm.predict(batch)}
        pred = {**votes, "vote": majority(votes, 3), "observer": router.predict(votes, "uniform"),
                "mixed": mixed.predict(scaler.transform(batch))}
        pred["recent"] = pred["observer"] if recent_tree is None else recent_tree.predict(batch)
        past_scores = {}
        if j >= 2:
            past_scores = {a: macro(y.loc[seen_ids], previous_predictions[a]) for a in ACTIONS}
            best = max(ACTIONS, key=past_scores.get)
            if past_scores[best] >= past_scores[active] + .01:
                active = best
            if once is None:
                once = active
        pred["ongoing"] = pred[active]
        pred["once"] = pred[once or "observer"]
        returned = sorted(ids, key=lambda k: digest([17, k]))[:32] if j+1 < len(days) else []
        file = OUT / f"issued-{j+1:02}.json"
        atomic_json(file, {"date": date, "ids": ids, "labels_used": len(seen_ids), "choice": active,
                           "previous_scores": past_scores, "returned_after": returned,
                           "predictions": {a: v.tolist() for a, v in pred.items()}})
        files.append(file)
        if returned:
            positions = [ids.index(k) for k in returned]
            label = y.loc[returned]
            router.update(label, {a: v[positions] for a, v in votes.items()})
            for arm in ACTIONS:
                previous_predictions[arm].extend(pred[arm][positions].tolist())
            recent_tree = RandomForestClassifier(n_estimators=1, random_state=0, n_jobs=1).fit(x.loc[returned], label)
            seen_ids.extend(returned)
            target_labels = y.loc[seen_ids]
            target_weights = compute_sample_weight("balanced", target_labels) * (len(ys)/len(target_labels))
            mixed.fit(np.vstack([source_history, scaler.transform(x.loc[seen_ids])]), np.r_[ys, target_labels],
                      sample_weight=np.r_[prior_weights, target_weights])
            count_fits += 1
    control = IssuedForest(keys, OUT / "author-ddm")
    ddm = author_definition("study_2.py", "c56695ca92297f5b338af2d8c08dd9f49af71f5de474c925c8daa4ba10b70a70",
                            "adaptive_batch_with_drift_detection_ddm", {"accuracy_score": accuracy_score})
    batches = [x.loc[ids] for ids in keys]
    ddm(control, xs, ys, batches, [y.loc[ids] for ids in keys], batches[-1], DDM(drift_threshold=.8, warm_start=3))
    if [e["keys"] for e in control.events] != keys:
        raise ValueError("作者对照未交付相同的日期")
    saved = [json.loads(f.read_bytes()) for f in files]
    target_ids = [k for r in saved for k in r["ids"]]
    if len(set(target_ids)) != len(target_ids) or set(train) & set(target_ids):
        raise ValueError("划分或计分重复")
    by_day = {a: [row["predictions"][a] for row in saved] for a in saved[0]["predictions"]}
    by_day["author-ddm"] = [e["predictions"] for e in control.events]
    table = {}
    for a, p in by_day.items():
        joined = np.concatenate(p)
        table[a] = {"macro_f1": macro(y.loc[target_ids], joined),
                    "recall": recall_score(y.loc[target_ids], joined, labels=CLASSES, average=None,
                                           zero_division=0).tolist(),
                    "worst_day_macro": min(macro(y.loc[ids], one_day) for ids, one_day in zip(keys, p, strict=True))}
    fixed = max((a for a in table if a != "ongoing"), key=lambda a: table[a]["macro_f1"])
    passed = (table["ongoing"]["macro_f1"] >= table[fixed]["macro_f1"]+.01 and
              min(table["ongoing"]["recall"]) >= min(table[fixed]["recall"])-.02)
    result = {"source_hash": EXPECTED, "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
              "source_ids": train, "service_rows": len(target_ids), "service_days": len(days), "excluded": excluded,
              "feedback": len(seen_ids), "local_fits_per_target_family": count_fits,
              "author_ddm_fits_including_initial": control.fit_attempts, "best_control": fixed, "pass": passed,
              "choices": dict(Counter(r["choice"] for r in saved)), "scores": table,
              "issued_sha256": [sha256(f.read_bytes()).hexdigest() for f in files]}
    atomic_json(OUT / "scores.json", result)
    print(len(train), len(target_ids), len(days), len(seen_ids), result["choices"],
          {k: round(v["macro_f1"], 5) for k, v in table.items()}, flush=True)


if __name__ == "__main__":
    main()
