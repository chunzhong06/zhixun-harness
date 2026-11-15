"""旧选择器原样提取有限答案，另一个发布模型只从已经交接的普通读数估计现场尺度。"""
import json
from hashlib import sha256

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from score_gas_feedback_selection import DEST as PRIOR, POLICIES, PROTOCOL as PRIOR_PROTOCOL
from score_gas_feedback_quantity import source_stream
from score_gas_probability_fusion import (
    ROOT, BASE, ARCHIVE, SEEDS, GasFeedbackBank, read_official_gas, stream, read,
    LocalStore, LocalAcceptance, atomic_json, digest, metrics,
)
from harness.models.field_classification import fixed_snapshot, model_scores, weights

DEST = BASE / "whole-maintenance/gas-served-scale"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-served-scale.json"


def classifier(scale, source, source_y, history, labels, mode):
    z = np.vstack((source, scale.transform(history)))
    # 沿用已登记的每类权重上限，少数已返回的标签不能独占一类的训练分量。
    target_w = weights(labels,len(source_y)) if mode == "domain" else (
        len(source_y) / 6 / np.maximum(np.bincount(labels,minlength=7)[labels],32/6))
    model = LogisticRegression(C=1., random_state=17, max_iter=1000).fit(
        z, np.r_[source_y, labels], sample_weight=np.r_[weights(source_y, len(source_y)), target_w])
    return fixed_snapshot("mixed", (scale, model))


def evaluate(seed, policy, pkg, batches, source, source_y):
    prior = PRIOR / f"seed-{seed}" / policy
    old = read(prior / "run.json")
    folder = DEST / f"seed-{seed}" / policy
    endpoint = LocalAcceptance()
    endpoint.output = folder / "accepted"
    history, answers, observed, requests = [], [], [], []
    reported = set()
    protocol, previous_identity = digest(read(PROTOCOL)), digest(read(PRIOR_PROTOCOL))
    for j, (ids, x, _, keys, fx, _, eligible_y, _) in enumerate(batches):
        request = old["requests"][j]
        previous = read(prior / "accepted" / (request + ".json"))["payload"]
        if digest([previous_identity, seed, policy, previous]) != request or previous["ids"] != ids:
            raise ValueError("旧取样器和本次服务测量不一致")
        published = previous["predictions"]
        scores = {"原规则": published["中途换尺度"], "固定源尺度": published["固定源尺度"],
                  "原观察者": published["原观察者"]}
        snapshots = {}
        for window in ("全部服务", "最近服务", "服务加回标"):
            for mode in (("domain","per-class") if window == "全部服务" else ("domain",)):
                name = window + ("缓增" if mode == "per-class" else "")
                pred = published["原观察者"]
                if history:
                    pool = (observed[-1] if window == "最近服务" else np.vstack(observed))
                    labeled = np.asfortranarray(history)
                    if window == "服务加回标":
                        pool = np.vstack((pool, labeled))
                    scale = StandardScaler().fit(np.asfortranarray(pool))
                    record = classifier(scale, source, source_y, labeled, np.asarray(answers), mode)
                    identity = digest(record)
                    atomic_json(DEST / "models" / (identity + ".json"), record)
                    snapshots[name] = identity
                    pred = (model_scores(record, x, 6).argmax(axis=1) + 1).tolist()
                scores[name] = pred
                scores[name + "中途"] = pred if j >= 5 else published["固定源尺度"]
        payload = {"ids": ids, "round": j, "previous": request, "scale_rows_seen": sum(map(len,observed)),
                   "labels_seen": len(answers), "snapshots": snapshots, "predictions": scores}
        key = digest([protocol,seed,policy,payload])
        ack = endpoint.submit(key,payload)
        if not ack["accepted"] or ack["request_id"] != key:
            raise RuntimeError("新预测未交接")
        requests.append(key)
        observed.append(x)
        if j < len(batches)-1:
            feedback = old["feedback"][j]
            if feedback["after_request"] != request:
                raise ValueError("回标与上一项实际已发预测不匹配")
            lookup = {k:i for i,k in enumerate(keys)}
            chosen = feedback["ids"]
            if len(set(chosen)) != 32 or set(chosen) & (reported | set(ids)):
                raise ValueError("反馈再利用或命中服务行")
            ix = [lookup[k] for k in chosen]
            y = [eligible_y[i] for i in ix]
            if y != feedback["answers"]:
                raise ValueError("冻结查询的实际标签改变")
            history.extend(fx[ix].tolist())
            answers.extend(y)
            reported.update(chosen)
    logs = [read(endpoint.output / (k + ".json"))["payload"] for k in requests]
    if any(digest([protocol,seed,policy,p]) != k for k,p in zip(requests,logs,strict=True)):
        raise ValueError("交接后预测改变")
    truth = {r.sample_id:r.label for r in pkg.validation}
    ids = [k for row in logs for k in row["ids"]]
    if len(set(ids)) != 414 or set(ids)&reported or len(reported)!=288:
        raise ValueError("服务或回标数量不符")
    y = [truth[k] for k in ids]
    table = {k:metrics(y,np.concatenate([p["predictions"][k] for p in logs])) for k in logs[0]["predictions"]}
    if table["原规则"] != old["scores"]["中途换尺度"]:
        raise ValueError("原规则分数不一致")
    atomic_json(folder/"run.json",{"requests":requests,"scores":table})
    return table


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已完成本协议")
        pkg = read_official_gas(ARCHIVE)
        bank = GasFeedbackBank(pkg.train)
        batches = source_stream(stream(pkg,bank),bank)
        sx = np.asfortranarray([r.values for r in pkg.train])
        source = StandardScaler().fit_transform(sx)
        sy = np.array([r.label for r in pkg.train])
        result = {}
        for policy in POLICIES:
            runs = [evaluate(seed,policy,pkg,batches,source,sy) for seed in SEEDS]
            mean = {arm:{k:float(np.mean([r[arm][k] for r in runs])) for k in ("macro","worst")} for arm in runs[0]}
            result[policy] = {"means":mean,"seed17":runs[0],"draws":runs}
            print(json.dumps({policy:mean},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"protocol":sha256(PROTOCOL.read_bytes()).hexdigest(),"scores":result})


if __name__=="__main__":
    main()
