"""将单树上批重训原样移到已有五条分类流，沿用已登记的回标时钟与强固定对手。"""
from hashlib import sha256
from time import process_time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from check_feedback_controls import scores
from harness.models.observer_errors import ObserverErrors
from score_feedback_refit import DEST as REFERENCE
from score_feedback_refit import ROOT, atomic_json, digest, read, sources


OUT = REFERENCE.parent / "last-batch-transfer"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-last-batch-transfer.json"


def evaluate(study):
    name, source, _, rows, stream, answers, budget, classes = study
    active = np.isfinite(source).any(axis=0)
    medians = np.nanmedian(np.where(np.isfinite(source[:, active]), source[:, active], np.nan), axis=0)

    def features(keys):
        data = np.asarray([rows[k] for k in keys])[:, active]
        return np.where(np.isfinite(data), data, medians)

    names = tuple(stream[0][1])
    observer = ObserverErrors(names, classes)
    transferred = ObserverErrors(names, classes)
    tree = None
    chunks = []
    fit_cpu = 0.
    history = []
    prior = read(REFERENCE / name / "scores.json")
    for j, (ids, votes, pool, pool_votes, returned_y) in enumerate(stream, 1):
        ref = REFERENCE / name / f"batch-{j:02}.json"
        if sha256(ref.read_bytes()).hexdigest() != prior["receipts"][j-1]:
            raise ValueError("原对照逐批记录已变")
        receipt = read(ref)
        ref_obs = observer.predict(votes, "uniform")
        if receipt["ids"] != list(ids) or ref_obs.tolist() != receipt["predictions"]["observer"]:
            raise ValueError("在相同样本上未复现当前交付预测")
        new_votes = dict(votes)
        if tree is not None:
            new_votes["current"] = tree.predict(features(ids))
        p = {"previous-observer": ref_obs.tolist(), "source-plus-site": receipt["predictions"]["refit"],
             "last-tree": (ref_obs if tree is None else new_votes["current"]).tolist(),
             "last-tree-calibrated": transferred.predict(new_votes, "uniform").tolist()}
        ix = sorted(range(len(pool)), key=lambda i: digest([17, pool[i]]))[:budget] if j < len(stream) else []
        returned = [pool[i] for i in ix]
        if receipt["returned_next"] != returned:
            raise ValueError("两种训练方法回标身份不一致")
        path = OUT / name / f"issued-{j:02}.json"
        atomic_json(path, {"ids": list(ids), "predictions": p, "previous_fit": history[-1] if history else [],
                           "returned_after": returned})
        chunks.append(path)
        if ix:
            labels = np.asarray([returned_y[i] for i in ix])
            sampled = {a: np.asarray(v)[ix] for a, v in pool_votes.items()}
            x = features(returned)
            observer.update(labels, sampled)
            if tree is not None:
                sampled["current"] = tree.predict(x)
            transferred.update(labels, sampled)
            start = process_time()
            tree = RandomForestClassifier(n_estimators=1, random_state=0, n_jobs=1).fit(x, labels)
            fit_cpu += process_time() - start
            history.append(returned)
    saved = [read(p) for p in chunks]
    ids = [k for row in saved for k in row["ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError("重复服务行")
    y = np.asarray([answers[k] for k in ids])
    blocks = np.split(np.arange(len(y)), np.cumsum([len(r["ids"]) for r in saved])[:-1])
    table = {a: scores(y, np.concatenate([r["predictions"][a] for r in saved]), blocks, classes)
             for a in saved[0]["predictions"]}
    for key, baseline in (("previous-observer", "observer"), ("source-plus-site", "refit")):
        if table[key] != prior["arms"][baseline]:
            raise ValueError("参考组得分未复现")
    result = {"scores": table, "feedback": sum(map(len, history)), "site_fits": len(history),
              "fit_cpu_seconds": fit_cpu, "rows": len(y),
              "tenure": "one returned batch", "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
              "issued_sha256": [sha256(p.read_bytes()).hexdigest() for p in chunks]}
    atomic_json(OUT / name / "scores.json", result)
    print(name, {a: round(s["pooled"], 5) for a, s in table.items()}, flush=True)
    return name, result


if __name__ == "__main__":
    atomic_json(OUT / "scores.json", dict(evaluate(study) for study in sources()))
