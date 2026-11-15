"""十轮真正交接后才查询，给覆盖、难例和历史邻域相同的训练器与反馈预算。"""
import json
import numpy as np

from score_gas_feedback_selection import DEST as SELECTED
from score_gas_feedback_quantity import source_stream
from score_gas_probability_fusion import (BASE, ROOT, ARCHIVE, SEEDS, SiteTraining, GasFeedbackBank, LocalAcceptance,
                                          LocalStore, read_official_gas, stream, read, metrics, atomic_json, digest)
from score_gas_served_scale import DEST as PRIOR
from harness.models.served_scale_gas import ServedScaleGas, MODE
from harness.models.probability_fusion import observer_probabilities
from harness.models.field_classification import model_scores
from harness.methods.feedback_coverage import coverage_queries

DEST = BASE / "whole-maintenance/gas-feedback-coverage"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-feedback-coverage.json"


def evaluate(pkg, batches, seed, policy):
    field = ServedScaleGas([r.values for r in pkg.train], [r.label for r in pkg.train], GasFeedbackBank.names, 6, MODE)
    folder = DEST / f"seed-{seed}" / policy
    endpoint = LocalAcceptance()
    endpoint.output = folder / "accepted"
    used, requests, feedback = set(), [], []
    service_ids = {k for batch in batches for k in batch[0]}
    protocol = digest(read(PROTOCOL))
    for j, (ids, x, votes, pool, fx, fv, pool_y, _) in enumerate(batches):
        active = field.current()
        pred = field.predict(x, votes)[active]
        record = {"round": j + 1, "ids": ids, "predictions": pred.tolist(), "returned_before": len(used)}
        key = digest([protocol, seed, policy, record])
        if not endpoint.submit(key, record)["accepted"]:
            raise ValueError("服务分类未先交付")
        requests.append(key)
        field.observe_served(x)
        if j + 1 == len(batches):
            continue
        if policy == "边界":
            selected = field.select(pool, fx, fv, 32, seed)
            ix = [pool.index(k) for k in selected]
        else:
            tool = SiteTraining.current(field)
            p = (observer_probabilities(field.observer, fv) if tool == "observer"
                 else model_scores(field.models()[tool], fx, 6))
            ix = coverage_queries(pool, fx, p, field.served, field.history, field.answers, 32, seed, policy)
            selected = [pool[i] for i in ix]
        if len(set(selected)) != 32 or set(selected) & (service_ids | used):
            raise ValueError("覆盖查询的额度、身份或授权池不符")
        vv = {k: v[ix] for k, v in fv.items()}
        before = field.predict(fx[ix], vv)
        answers = [pool_y[i] for i in ix]
        field.update(fx[ix], vv, answers, before)
        used.update(selected)
        feedback.append({"after_request": key, "ids": selected, "answers": answers})
    if len(used) != 288 or len(service_ids) != 414:
        raise ValueError("十轮发放或回标数异常")
    truths = {r.sample_id: r.label for r in pkg.validation}
    actual, answer = [], []
    for key in requests:
        record = read(endpoint.output / (key + ".json"))["payload"]
        if digest([protocol, seed, policy, record]) != key:
            raise ValueError("回读分类与实际交付不同")
        actual.extend(record["predictions"])
        answer.extend(truths[k] for k in record["ids"])
    score = metrics(answer, actual)
    if policy == "边界":
        prior = read(SELECTED / f"seed-{seed}/边界/run.json")
        if [v["ids"] for v in feedback] != [v["ids"] for v in prior["feedback"]]:
            raise ValueError("改前固定难例查询未复现")
        if score != read(PRIOR / f"seed-{seed}/边界/run.json")["scores"]["全部服务中途"]:
            raise ValueError("改前整批预测成绩未复现")
    atomic_json(folder / "run.json", {"requests": requests, "feedback": feedback, "scores": score})
    return score


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("覆盖开发比较已完成")
        pkg = read_official_gas(ARCHIVE)
        bank = GasFeedbackBank(pkg.train)
        batches = source_stream(stream(pkg, bank), bank)
        out = {}
        old = read(PRIOR / "scores.json")
        for policy in ("边界", "固定均匀覆盖", "固定难度覆盖", "历史邻域"):
            rows = [evaluate(pkg, batches, seed, policy) for seed in SEEDS]
            out[policy] = {"draws": rows, "macro": float(np.mean([r["macro"] for r in rows])),
                           "worst": float(np.mean([r["worst"] for r in rows]))}
            print(json.dumps({policy: {k: v for k, v in out[policy].items() if k != "draws"}}, ensure_ascii=False), flush=True)
        earlier = {k: r["means"]["全部服务中途"] for k, r in old["scores"].items()}
        atomic_json(DEST / "scores.json", {"seeds": list(SEEDS), "new": out, "previous": earlier})


if __name__ == "__main__":
    main()
