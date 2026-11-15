"""用既有工具和同一个累计损失选择器，逐批发放半导体质检预测，再开放本批模拟反馈。"""
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from time import process_time
from zipfile import ZipFile

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score

from score_local_defaults import BASE, ROOT, atomic_json, read
from harness.contracts import ActionPlan, Sample, TaskSpec, digest
from harness.models.feedback_choice import HistoricalToolChoice
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend


PROTOCOL = ROOT / "engineering/configs/experiments/secom-feedback-choice.json"
DEST = BASE / "secom-feedback-choice"


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def source(path):
    with ZipFile(path) as archive:
        x = np.loadtxt(BytesIO(archive.read("secom.data")))
        lines = archive.read("secom_labels.data").decode().splitlines()
    labels, times = [], []
    for line in lines:
        label, stamp = line.split(maxsplit=1)
        labels.append(int(label))
        times.append(datetime.strptime(stamp.strip('"'), "%d/%m/%Y %H:%M:%S"))
    if len(times) != len(x) or times != sorted(times) or set(labels) != {-1, 1}:
        raise ValueError("测量与标签未按原始时间对齐")
    return x, times, np.asarray(labels)


def boundaries(times):
    total = len(times)
    first = 2 * total // 5
    result = []
    for i in range(9):
        cut = first + (total - first) * i // 8
        while 0 < cut < total and times[cut] == times[cut - 1]:
            cut += 1
        result.append(cut)
    if any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError("时间相同的样本不能跨批次")
    return result


def fit_bank(x, y):
    mask = np.isfinite(x).any(axis=0)
    imputer = SimpleImputer(strategy="median").fit(x[:, mask])
    train = imputer.transform(x[:, mask])
    task = TaskSpec("secom", tuple(f"sensor{i}" for i in np.flatnonzero(mask)), (-1, 1), metric="macro_f1")
    rows = tuple(Sample(str(i), 1, tuple(v), int(target), "fit", "source", 0)
                 for i, (v, target) in enumerate(zip(train, y, strict=True)))
    backend = SklearnBackend(execution_enabled=True)
    models, receipts = {}, {}
    for name, family in (("balanced-logistic", "linear"), ("balanced-forest", "forest"), ("balanced-catboost", "catboost")):
        args = {} if family == "linear" else {"trees": 96, "max_depth": 6}
        recipe = PipelineRecipe(predictor=family, processor="standard" if family == "linear" else "none",
                                class_weight="balanced", **args)
        start = process_time()
        models[name] = backend.fit(task, "source", ActionPlan("balanced", pipeline=recipe), rows)
        receipts[name] = {"recipe": recipe.key, "cpu_seconds": process_time() - start}
    def predict(raw):
        clean = imputer.transform(raw[:, mask])
        probs = {}
        for name, model in models.items():
            if list(model.classes_) != [-1, 1]:
                raise ValueError("缺陷概率必须对应标签 1")
            opts = {"thread_count": 1} if name == "balanced-catboost" else {}
            probs[name] = model.predict_proba(clean, **opts)[:, 1]
        probs["mean-probability"] = np.mean(list(probs.values()), axis=0)
        probs["always-pass"] = np.zeros(len(raw))
        return {a: np.where(p >= .5, 1, -1).tolist() for a, p in probs.items()}
    return predict, {"models": receipts, "usable_columns": int(mask.sum()), "backend": backend.specification}


def macro(truth, values):
    return float(f1_score(truth, values, labels=[-1, 1], average="macro", zero_division=0))


def main():
    protocol = read(PROTOCOL)
    data = ROOT / protocol["file"]
    x, times, labels = source(data)
    cuts = boundaries(times)
    predict, bank = fit_bank(x[:cuts[0]], labels[:cuts[0]])
    selector = HistoricalToolChoice(protocol["tools"], protocol["default"])
    receipts, prior_losses = [], []
    for number, (start, stop) in enumerate(zip(cuts, cuts[1:]), 1):
        outputs = predict(x[start:stop])
        outputs["once"] = outputs[selector.first]
        outputs["ongoing"] = outputs[selector.active]
        path = DEST / f"batch-{number}.json"
        atomic_json(path, {"rows": [start, stop], "times": [times[start].isoformat(), times[stop-1].isoformat()],
                           "feedback_rows_before": start - cuts[0], "released_losses": list(prior_losses),
                           "once": selector.first, "ongoing": selector.active, "predictions": outputs})
        receipts.append(fingerprint(path))
        saved = read(path)
        if number < 8:
            # 仅预测已经封存的批次能够向下一批提供误差。
            losses = {a: 1 - macro(labels[start:stop], saved["predictions"][a]) for a in protocol["tools"]}
            selector.update([losses])
            prior_losses.append(losses)
    table = {}
    records = [read(DEST / f"batch-{i}.json") for i in range(1, 9)]
    for arm in records[0]["predictions"]:
        pred = np.concatenate([r["predictions"][arm] for r in records])
        per = [macro(labels[a:b], r["predictions"][arm]) for (a,b), r in zip(zip(cuts,cuts[1:]), records, strict=True)]
        _, fp, fn, tp = confusion_matrix(labels[cuts[0]:], pred, labels=[-1, 1]).ravel()
        table[arm] = {"macro_f1": macro(labels[cuts[0]:], pred), "failed_recall": float(tp / (fn + tp)),
                      "detected": int(tp), "missed": int(fn), "false_alarms": int(fp), "worst": min(per), "batches": per}
    fixed = max(protocol["tools"], key=lambda a: table[a]["macro_f1"])
    best, adaptive = table[fixed], table["ongoing"]
    accept = (adaptive["macro_f1"] >= max(best["macro_f1"], table["once"]["macro_f1"]) + .01
              and all(adaptive[m] >= best[m] - .01 for m in ("failed_recall", "worst")))
    result = {"protocol_sha256": fingerprint(PROTOCOL), "data_sha256": fingerprint(data), "receipt_sha256": receipts,
              "fit_rows": cuts[0], "score_rows": len(x) - cuts[0], "feedback_rows_used": cuts[-2] - cuts[0],
              "train_rows_sha256": digest(list(range(cuts[0]))), "bank": bank, "decisions": [r["ongoing"] for r in records],
              "best_fixed": fixed, "adopt": accept, "arms": table}
    atomic_json(DEST / "scores.json", result)
    print(result["score_rows"], result["decisions"], {k: round(v["macro_f1"], 5) for k,v in table.items()})


if __name__ == "__main__":
    main()
