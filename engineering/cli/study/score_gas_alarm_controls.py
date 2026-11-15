"""验证现场重训时，作者提出的筛选条件相对同一批全部回标的增量。"""
from hashlib import sha256
import json

import numpy as np
from river.drift.binary import DDM
from sklearn.metrics import accuracy_score

from score_author_new_arrays import (
    IssuedForest, ObserverErrors, OUT, ROOT, atomic_json, author_definition,
    metrics, read_arrays,
)


PROTOCOL = ROOT / "engineering/configs/experiments/gas-alarm-control.json"
METHOD_SHA = "c56695ca92297f5b338af2d8c08dd9f49af71f5de474c925c8daa4ba10b70a70"
DEST = OUT.parent / "new-arrays-controls"


def read(path):
    return json.loads(path.read_bytes())


def scheduled(keys, folder, source, batches, returned, refit_after):
    model = IssuedForest(keys, folder)
    model.fit(*source)
    for index, batch in enumerate(batches, 1):
        model.predict(batch)
        if index in refit_after:
            model.fit(batch, returned[index - 1])
    return model


def score_site(device, data):
    x, y, groups, cycles = data
    source_ids = groups[1, -1]
    source = (x.loc[source_ids], y.loc[source_ids])
    keys = [groups[device, i] for i in range(10)]
    service = [k for ids in keys for k in ids]
    batches = [x.loc[ids] for ids in keys]
    returned = [y.loc[ids] for ids in keys]
    name = f"array-{device}"
    folder = DEST / name
    base = OUT / name
    author_scores = read(base / "scores.json")
    if author_scores["status"] != "complete":
        raise ValueError("完整作者预测尚未交付")
    paper = [read(base / "paper" / f"round-{i:02}.json") for i in range(1, 11)]
    changed = [i for i in range(1, 10) if paper[i]["fits_before"] > paper[i-1]["fits_before"]]

    ddm = IssuedForest(keys, folder / "ddm")
    alarm = author_definition("study_2.py", METHOD_SHA, "adaptive_batch_with_drift_detection_ddm",
                              {"accuracy_score": accuracy_score})
    # 作者对照的最后一批单独传入，前九批才允许用于维护。
    alarm(ddm, *source, batches, returned, batches[-1], DDM(drift_threshold=.8, warm_start=3))
    every = scheduled(keys, folder / "every", source, batches, returned, range(1, 10))
    matched = scheduled(keys, folder / "matched", source, batches, returned, changed)
    runs = {"ddm": ddm, "every": every, "matched": matched}
    sequences = {arm: model.events for arm, model in runs.items()}
    sequences["author"] = paper
    names = ("current", "logistic", "forest", "lda", "svm")
    observers = {arm: ObserverErrors(names, 5) for arm in sequences}
    counts = {"paper": author_scores["author_fit_attempts"],
              **{k: v.fit_attempts for k, v in runs.items()}}
    chunks = []
    for j, ids in enumerate(keys):
        current = read(base / "project" / f"round-{j+1:02}.json")
        if current["keys"] != ids or current["known"] != 5*j:
            raise ValueError("现场分类器与冻结的服务行或回标时钟不符")
        pred = {key: current["predictions"][key] for key in ("project-transfer", "shared-fixed")}
        trial_votes = {}
        for arm, events in sequences.items():
            event = events[j]
            if len(events) != 10 or event["keys"] != ids:
                raise ValueError("模型没有逐轮交付同一批观测")
            votes = {k: np.asarray(current["predictions"][k]) for k in names}
            votes["current"] = np.asarray(event["predictions"])
            pred[arm] = votes["current"].tolist()
            pred[arm + "-calibrated"] = observers[arm].predict(votes, "uniform").tolist()
            trial_votes[arm] = votes
        receipt = folder / f"issued-{j+1:02}.json"
        atomic_json(receipt, {"keys": ids, "feedback_used": 5 * j, "predictions": pred})
        chunks.append(receipt)
        if j < 9:
            for arm, votes in trial_votes.items():
                observers[arm].update(returned[j], votes)

    output = [read(file)["predictions"] for file in chunks]
    predictions = {a: np.asarray([p for row in output for p in row[a]]) for a in output[0]}
    answers = y.loc[service]
    results = {k: metrics(answers, v, cycles) for k, v in predictions.items()}
    for new, old in (("author", "author-study2"), ("author-calibrated", "author-with-observer"),
                     ("project-transfer", "project-transfer")):
        if results[new] != author_scores["arms"][old]:
            raise ValueError("作者或项目对照没有复现")
    controls = {}
    for name_a, name_b in (("matched", "author"), ("ddm", "author"), ("every", "author")):
        correct_a = predictions[name_a] == answers.to_numpy()
        correct_b = predictions[name_b] == answers.to_numpy()
        controls[name_a] = {"different": int(np.count_nonzero(predictions[name_a] != predictions[name_b])),
                            "only_control_correct": int(np.sum(correct_a & ~correct_b)),
                            "only_author_correct": int(np.sum(correct_b & ~correct_a))}
    result = {"scores": results, "paired_with_author": controls, "author_refit_after": changed,
              "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
              "round_sha256": [sha256(p.read_bytes()).hexdigest() for p in chunks],
              "fits_including_source": counts, "author_requests": len(author_scores["all_author_requests"]),
              "feedback_rows": 45}
    atomic_json(folder / "scores.json", result)
    return result


if __name__ == "__main__":
    arrays = read_arrays()
    results = {f"array-{i}": score_site(i, arrays) for i in range(2, 6)}
    atomic_json(DEST / "scores.json", results)
    for site, result in results.items():
        print(site, {k: round(v["macro_f1"], 5) for k, v in result["scores"].items()},
              result["fits_including_source"], result["paired_with_author"], flush=True)
