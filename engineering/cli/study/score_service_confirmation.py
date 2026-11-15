"""比较发布前查看原分类器或实际交付结果。每份确认数据用到达前的校准计数计分。"""
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from compare_native_release_path import DEST as PRIOR, identity, predict
from compare_group_objectives import _trainer
from gas_complementary_feedback import OUT, ROOT, atomic_json, digest, read
from harness.models.observer_errors import ObserverErrors
from score_feedback_refit import SEEDS, sources
from score_native_feedback import balanced_draw


PROTOCOL = ROOT / "engineering/configs/experiments/gas-service-confirmation.json"
DEST = OUT / "service-confirmation"
MODES = ("recommended", "root", "latest", "alternate", "native-strict", "service-strict", "service-tie")


def run_seed(study, trainer, parent, seed, tmp):
    site, source, source_y, rows, stream, truth, budget, classes = study
    incumbent = dict.fromkeys(MODES, 0)
    names = tuple(stream[0][1])
    routers = {k: ObserverErrors(names, classes) for k in MODES}
    models = [trainer.Model.load(parent)]
    history, labels, held_ids, held_labels = [], [], [], []
    previous_counts, held_votes = {}, {}
    receipt_paths, training_seconds, fit_cpu, steps = [], 0., 0., 0
    folder = DEST / site / str(seed)
    original = read(PRIOR / site / "scores.json") if seed == 17 else None
    protocol = identity(PROTOCOL)
    for j, (keys, votes, pool, fp, pool_y) in enumerate(stream):
        decisions = {}
        if j:
            rng = np.random.default_rng(np.random.SeedSequence([17, j]))
            src_ix = balanced_draw(source_y, len(source), rng)
            balanced_draw(source_y, len(source), rng)  # 按原控制组顺序消耗第二次源抽样。
            hist_ix = balanced_draw(np.array(labels), len(source), rng)
            x = np.vstack((source[src_ix], np.array([rows[history[i]] for i in hist_ix])))
            y = np.r_[source_y[src_ix], np.asarray(labels)[hist_ix]] - 1
            path = tmp / f"{site}-{seed}-{j}.npz"
            models[-1].save(path)
            model = trainer.Model.load(path)
            start, cpu = perf_counter(), process_time()
            model.fit(x, y, 1)
            training_seconds += perf_counter() - start
            fit_cpu += process_time() - cpu
            steps += model.meta["steps"] - models[-1].meta["steps"]
            models.append(model)
            held_x = np.asarray([rows[k] for k in held_ids])
            native = {i: predict(models[i], held_x) for i in {*incumbent.values(), j}}
            incumbent["latest"] = j
            if j % 2 == 0:
                incumbent["alternate"] = j
            for arm in ("native-strict", "service-strict", "service-tie"):
                prior = incumbent[arm]
                router = ObserverErrors(names, classes, previous_counts[arm])
                old_service = router.predict({**held_votes, "current": native[prior]}, "uniform")
                new_service = router.predict({**held_votes, "current": native[j]}, "uniform")
                old, new = (native[prior], native[j]) if arm == "native-strict" else (old_service, new_service)
                correct = np.asarray(held_labels)
                gain = int(np.sum(new == correct) - np.sum(old == correct))
                take = gain > 0 or (arm == "service-tie" and gain == 0)
                if take:
                    incumbent[arm] = j
                decisions[arm] = {"old": prior, "candidate": j, "gain_on_holdout": gain, "publish": take,
                                  "native_disagree": int(np.sum(native[prior] != native[j])),
                                  "service_disagree": int(np.sum(old_service != new_service))}
        values = np.asarray([rows[k] for k in keys])
        candidate_raw = {i: predict(models[i], values) for i in set(incumbent.values())}
        predictions = {a: router.predict(votes if a == "recommended" else {**votes, "current": candidate_raw[incumbent[a]]},
                                         "uniform").tolist() for a, router in routers.items()}
        predictions["latest-raw"] = candidate_raw[j].tolist()
        ix = sorted(range(len(pool)), key=lambda i: digest([seed, pool[i]]))[:budget] if j + 1 < len(stream) else []
        selected = [pool[i] for i in ix]
        if original:
            baseline = read(PRIOR / site / "accepted" / (original["requests"][j] + ".json"))["payload"]
            if list(keys) != baseline["ids"] or selected != baseline["decision"]["returned_after"]:
                raise ValueError("资格或回标抽样变动")
            matches = {"recommended": "service/recommended", "root": "service/original", "latest": "service/mixed",
                       "native-strict": "service/detached", "latest-raw": "raw/mixed"}
            if any(predictions[a] != baseline["predictions"][b] for a, b in matches.items()):
                raise ValueError("重训产生的原校准、候选或换版未逐条复现")
        payload = {"keys": list(keys), "round": j+1, "returned_after": selected, "holdout_ids": held_ids,
                   "fit_history": digest(history), "versions": incumbent.copy(), "decisions": decisions,
                   "predictions": predictions}
        request = digest([protocol, seed, site, payload])
        path = folder / f"round-{j+1:02}.json"
        atomic_json(path, {"request": request, "payload": payload})
        receipt_paths.append(path)
        if ix:
            # 回执落盘后才开放本批标签。下轮模型比较保留当前的计数，排除确认行自我校准。
            visible_y = np.asarray([pool_y[i] for i in ix])
            mid = len(ix) // 2
            history.extend(selected[:mid])
            labels.extend(visible_y[:mid].tolist())
            held_ids, held_labels = selected[mid:], visible_y[mid:].tolist()
            if set(held_ids) & set(history):
                raise ValueError("确认样本用于训练")
            observed = {k: np.asarray(v)[ix] for k, v in fp.items()}
            held_votes = {k: v[mid:] for k, v in observed.items()}
            seen_x = np.asarray([rows[k] for k in selected])
            native = {i: predict(models[i], seen_x) for i in set(incumbent.values())}
            for arm, router in routers.items():
                previous_counts[arm] = router.counts.copy()
                used = observed if arm == "recommended" else {**observed, "current": native[incumbent[arm]]}
                router.update(visible_y, used)
    batches = []
    for path in receipt_paths:
        record = read(path)
        if record["request"] != digest([protocol, seed, site, record["payload"]]):
            raise ValueError("接收内容变动")
        batches.append(record["payload"])
    keys = [k for batch in batches for k in batch["keys"]]
    if len(keys) != len(set(keys)):
        raise ValueError("重复发放")
    y = np.asarray([truth[k] for k in keys])
    blocks = np.split(np.arange(len(y)), np.cumsum([len(batch["keys"]) for batch in batches])[:-1])
    table = {}
    for arm in batches[0]["predictions"]:
        issued = np.concatenate([r["predictions"][arm] for r in batches])
        table[arm] = scores(y, issued, blocks, classes)
    refdir = OUT / "feedback-refit"
    ref = read((refdir if seed == 17 else refdir / "feedback-seeds" / str(seed)) / site / "scores.json")
    table["independent-mixed"] = ref["arms"]["refit"]
    if abs(table["recommended"]["pooled"] - ref["arms"]["observer"]["pooled"]) > 1e-12:
        raise ValueError("现行校准的抽样对照未复现")
    result = {"scores": table, "requests": [identity(path) for path in receipt_paths], "protocol": protocol,
              "weight_updates": len(models)-1, "gradient_steps": steps, "training_rows": 2*len(source)*(len(models)-1),
              "returned": 2*len(history), "shared_fit_wall_s": training_seconds, "shared_fit_cpu_s": fit_cpu}
    atomic_json(folder / "scores.json", result)
    return result


def evaluate(study, trainer, parent, tmp):
    site = study[0]
    runs = {seed: run_seed(study, trainer, parent, seed, tmp) for seed in SEEDS}
    means = {}
    for arm in next(iter(runs.values()))["scores"]:
        means[arm] = {"macro": float(np.mean([r["scores"][arm]["pooled"] for r in runs.values()])),
                      "worst_recall": float(np.mean([min(r["scores"][arm]["recall"]) for r in runs.values()]))}
    fixed = ("root", "latest", "alternate", "latest-raw", "recommended")
    gate = {a: all(means[a]["macro"] >= means[k]["macro"] + .01 and
                   means[a]["worst_recall"] >= means[k]["worst_recall"] - .02 for k in fixed)
            for a in ("service-strict", "service-tie")}
    result = {"means": means, "gate": gate, "protocol": identity(PROTOCOL),
              "draws": {str(s): identity(DEST / site / str(s) / "scores.json") for s in runs}}
    atomic_json(DEST / site / "scores.json", result)
    print(site, {k: round(v["macro"], 5) for k, v in means.items()}, gate, flush=True)


if __name__ == "__main__":
    archive, = (ROOT / "data/official/gas").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "A") as trainer:
        parent = Path(tmp) / "root.npz"
        parent.write_bytes(z.read("checkpoint.npz"))
        for study in islice(sources(), 2):
            evaluate(study, trainer, parent, Path(tmp))
