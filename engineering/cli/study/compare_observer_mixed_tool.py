"""用真实已发模型的预测作为新增投票，同一反馈历史上检验整套交付与固定、只选一次的差别。"""
from collections import Counter
from hashlib import sha256

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import ROOT, atomic_json, digest, read
from harness.models.field_classification import macro
from harness.models.observer_errors import ObserverErrors
from score_feedback_refit import DEST as PREVIOUS, SEEDS, sources
from score_whole_maintenance import INTEGRATED, integrated_predictions


OUT = PREVIOUS.parent / "observer-with-mixed"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-observer-mixed-tool.json"
CHOICES = ("observer", "mixed", "add")


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def integrated_mixed():
    folder = INTEGRATED.parents[3] / "whole-maintenance/field-delivery/integrated-fixed/gas"
    trajectory = read(folder / "run.json")
    values = {}
    for event in trajectory["accepted"]:
        payload = read(folder / "accepted" / (event["request"] + ".json"))["payload"]
        if digest([trajectory["identity"], payload]) != event["request"] or payload["selected_tool"] != "mixed":
            raise ValueError("固定混合模型交付身份不符")
        values.update(payload["predictions"])
    return values


def run(study, seed):
    site, _, _, _, stream, truth, budget, classes = study
    folder = (PREVIOUS if seed == 17 else PREVIOUS / "feedback-seeds" / str(seed)) / site
    refs = read(folder / "scores.json")
    bank = tuple(stream[0][1])
    current_key = "current" if "current" in bank else bank[0]
    add = ObserverErrors((*bank, "mixed"), classes)
    replace = ObserverErrors(bank, classes)
    original = ObserverErrors(bank, classes)
    history = np.zeros((len(CHOICES), classes, classes), dtype=np.int64)
    active, once = "observer", None
    deliveries = []
    for j, (keys, votes, pool, fp, pool_y) in enumerate(stream):
        path = folder / f"batch-{j+1:02}.json"
        if hash_file(path) != refs["receipts"][j]:
            raise ValueError("被复用的混合训练输出变动")
        prior = read(path)
        if list(keys) != prior["ids"]:
            raise ValueError("评价行没有对齐")
        mixed = np.asarray(prior["predictions"]["refit"], dtype=int)
        p = {"observer": original.predict(votes, "uniform"), "mixed": mixed,
             "add": add.predict({**votes, "mixed": mixed}, "uniform"),
             "replace": replace.predict({**votes, current_key: mixed}, "uniform")}
        if p["observer"].tolist() != prior["predictions"]["observer"]:
            raise ValueError("改前校准没有逐条复现")
        p.update(ongoing=p[active], once=p[once or "observer"])
        ix = sorted(range(len(pool)), key=lambda i: digest([seed, pool[i]]))[:budget] if j+1 < len(stream) else []
        sampled_ids = [pool[i] for i in ix]
        if sampled_ids != prior["returned_next"]:
            raise ValueError("更换工具改变了反馈样本")
        deliveries.append({"ids": list(keys), "selected": active, "first": once, "returned_after": sampled_ids,
                           "predictions": {k: v.tolist() for k, v in p.items()}})
        if ix:
            sampled = {k: np.asarray(v)[ix] for k, v in fp.items()}
            # 混合模型先做现场预测，再回标和重训；此处只使用当时产生的预测及随后开放的同一标签。
            new_out = np.asarray(prior["feedback_predictions"]["refit"], dtype=int)
            add_votes = {**sampled, "mixed": new_out}
            old = original.predict(sampled, "uniform")
            if old.tolist() != prior["feedback_predictions"]["observer"]:
                raise ValueError("旧工具反馈输出不同")
            before = (old, new_out, add.predict(add_votes, "uniform"))
            y = np.asarray([pool_y[i] for i in ix], dtype=int)
            for k, values in enumerate(before):
                np.add.at(history[k], (y-1, values-1), 1)
            rank = {a: macro(h) for a, h in zip(CHOICES, history, strict=True)}
            best = max(CHOICES, key=rank.get)
            if rank[best] >= rank[active] + .01:
                active = best
            if once is None and any(np.any(v != old) for v in before[1:]):
                once = active
            original.update(y, sampled)
            add.update(y, add_votes)
            replace.update(y, {**sampled, current_key: new_out})
    path = OUT / site / f"seed-{seed}.json"
    atomic_json(path, {"protocol": hash_file(PROTOCOL), "draw": seed, "replaced": current_key, "batches": deliveries})
    issued = read(path)["batches"]
    keys = [k for b in issued for k in b["ids"]]
    if len(keys) != len(set(keys)):
        raise ValueError("服务行有重叠")
    truth_y = np.array([truth[k] for k in keys])
    blocks = np.split(np.arange(len(keys)), np.cumsum([len(b["ids"]) for b in issued])[:-1])
    table = {}
    for arm in issued[0]["predictions"]:
        pred = np.concatenate([r["predictions"][arm] for r in issued])
        table[arm] = scores(truth_y, pred, blocks, classes)
        if site == "public" and seed == 17 and arm in ("observer", "mixed"):
            actual = integrated_predictions("gas") if arm == "observer" else integrated_mixed()
            if dict(zip(keys, pred.tolist(), strict=True)) != actual:
                raise ValueError("未逐条复现正式公开入口")
    return {"seed": seed, "receipt": hash_file(path), "scores": table, "issued_choices": dict(Counter(r["selected"] for r in issued))}


def evaluate(study):
    runs = [run(study, seed) for seed in SEEDS]
    means = {a: {"macro": float(np.mean([r["scores"][a]["pooled"] for r in runs])),
                 "worst": float(np.mean([min(r["scores"][a]["recall"]) for r in runs]))} for a in runs[0]["scores"]}
    result = {"means": means, "runs": runs, "new_fits": 0, "protocol": hash_file(PROTOCOL)}
    atomic_json(OUT / study[0] / "scores.json", result)
    print(study[0], {a: round(v["macro"], 5) for a, v in means.items()}, flush=True)
    return means


if __name__ == "__main__":
    means = {s[0]: evaluate(s) for s in sources()}
    passed = {}
    for arm in ("mixed", "add", "replace", "ongoing"):
        controls = ("observer",) if arm == "mixed" else ("observer", "mixed")
        if arm == "ongoing":
            controls += ("add", "replace", "once")
        main = all(t[arm]["macro"] >= t[c]["macro"] + .01 and t[arm]["worst"] >= t["observer"]["worst"] - .02
                   for site, t in means.items() if site in ("public", "late") for c in controls)
        elsewhere = all(t[arm]["macro"] >= t["observer"]["macro"] - .01 and t[arm]["worst"] >= t["observer"]["worst"] - .02
                        for site, t in means.items() if site not in ("public", "late"))
        passed[arm] = main and elsewhere
    atomic_json(OUT / "scores.json", {"mean_by_site": means, "gate": passed, "protocol": hash_file(PROTOCOL)})
