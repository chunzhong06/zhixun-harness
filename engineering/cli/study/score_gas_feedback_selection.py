"""各组仅查看自己挑出的当前回标，按原预算真实重训，最终只计已下发预测。"""
import json

import numpy as np

from score_gas_feedback_quantity import DEST as EARLIER, source_stream
from score_gas_probability_fusion import (
    ROOT, BASE, ARCHIVE, SEEDS, SiteTraining, GasFeedbackBank, LocalAcceptance, LocalStore,
    read_official_gas, stream, read, metrics, atomic_json, digest,
)
from harness.methods.feedback_sampling import select_feedback
from harness.models.probability_fusion import observer_probabilities
from harness.models.field_classification import model_scores

DEST = BASE / "whole-maintenance/gas-feedback-selection"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-feedback-selection.json"
POLICIES = ("随机", "边界", "类别均衡", "均衡边界", "固定半随机", "历史错误")


def evaluate(pkg, batches, seed, policy):
    field = SiteTraining([r.values for r in pkg.train], [r.label for r in pkg.train],
                         GasFeedbackBank.names, 6, "pooled-midway-cold")
    counts = np.zeros((6, 2), dtype=int)
    folder = DEST / f"seed-{seed}" / policy
    receiver = LocalAcceptance()
    receiver.output = folder / "accepted"
    requests, feedback = [], []
    received = set()
    service = {k for row in batches for k in row[0]}
    protocol = digest(read(PROTOCOL))
    for j, (ids, x, votes, pool, fx, fv, pool_y, base) in enumerate(batches):
        served = field.predict(x, votes)
        observed = served["observer"]
        source, site = served["pooled-source"], served["pooled-separate"]
        predictions = {"中途换尺度": served[field.current()], "原观察者": observed,
                       "固定源尺度": source if field.history else observed,
                       "固定现场尺度": site if field.history else observed}
        snapshot = {"models": field.models(), "counts": field.observer.counts.tolist(),
                    "current": field.current(), "base": base}
        identity = digest(snapshot)
        atomic_json(DEST / "snapshots" / (identity + ".json"), snapshot)
        payload = {"ids": ids, "round": j, "snapshot": identity, "past_labels": len(received),
                   "predictions": {k: v.tolist() for k, v in predictions.items()}}
        key = digest([protocol, seed, policy, payload])
        ack = receiver.submit(key, payload)
        if not ack.get("accepted") or ack["request_id"] != key:
            raise RuntimeError("气体预测未交接")
        requests.append(key)
        if j == len(batches) - 1:
            continue
        current = field.current()
        p = (observer_probabilities(field.observer, fv) if current == "observer"
             else model_scores(field.models()[current], fx, 6))
        chosen, probes = select_feedback(pool, p, 32, seed, policy, counts)
        returned = [pool[i] for i in chosen]
        if len(set(returned)) != 32 or set(returned) & (received | service):
            raise ValueError("回标重复、超出额度或查到服务答案")
        # 查询清单固定后再索引答案，仅已返回的随机探针可以更新类别历史。
        pred = field.predict(fx[chosen], {k: v[chosen] for k, v in fv.items()})
        answers = [pool_y[i] for i in chosen]
        answer_by_index = dict(zip(chosen, answers, strict=True))
        for i in probes:
            label = int(p[i].argmax())
            counts[label, int(label + 1 != answer_by_index[i])] += 1
        field.update(fx[chosen], {k: v[chosen] for k, v in fv.items()}, answers, pred)
        received.update(returned)
        feedback.append({"after_request": key, "ids": returned, "answers": answers,
                         "random_probes": [pool[i] for i in probes]})
    if len(received) != 288 or len(service) != 414:
        raise ValueError("提交或反馈总数改变")
    records = []
    for key in requests:
        payload = read(receiver.output / (key + ".json"))["payload"]
        if digest([protocol, seed, policy, payload]) != key:
            raise ValueError("回读的预测与发放时不同")
        records.append(payload)
    truths = {r.sample_id: r.label for r in pkg.validation}
    y = [truths[k] for record in records for k in record["ids"]]
    scores = {arm: metrics(y, np.concatenate([r["predictions"][arm] for r in records]))
              for arm in records[0]["predictions"]}
    if policy == "随机":
        prior = read(EARLIER / "source/cap-32" / f"seed-{seed}" / "scores.json")
        for new, old in (("中途换尺度","current"), ("原观察者","observer"),
                         ("固定源尺度","source-cold"), ("固定现场尺度","separate-cold")):
            if scores[new] != prior["scores"][old]:
                raise ValueError("固定参照未能复现")
    atomic_json(folder / "run.json", {"requests": requests, "feedback": feedback, "scores": scores})
    return scores


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已完成本协议")
        pkg = read_official_gas(ARCHIVE)
        bank = GasFeedbackBank(pkg.train)
        batches = source_stream(stream(pkg, bank), bank)
        result = {}
        for policy in POLICIES:
            draws = [evaluate(pkg, batches, seed, policy) for seed in SEEDS]
            mean = {arm: {metric: float(np.mean([row[arm][metric] for row in draws]))
                          for metric in ("macro", "worst")} for arm in draws[0]}
            result[policy] = {"means": mean, "seed17": draws[0], "draws": draws}
            print(json.dumps({policy: mean}, ensure_ascii=False), flush=True)
        atomic_json(DEST / "scores.json", {"seeds": list(SEEDS), "development": True, "scores": result})


if __name__ == "__main__":
    main()
