"""相同回标与相同特征下比较直接续拟合、现场校准，以及根据此前交付分数决定是否交接。"""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import islice
from time import process_time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from check_feedback_controls import scores
from compare_fixed_tree_references import paths
from gas_complementary_feedback import OUT, ROOT, REGISTERED_SHA256, atomic_json, digest, official, read, read_gas_batches
from harness.models.observer_errors import ObserverErrors
from harness.models.vote_feedback import majority
from score_new_electronic_nose import source_frame
from score_observer_errors import datasets
from score_guarded_gas import SEEDS
from sgp40_transfer import prepared, streams


DEST = OUT / "feedback-refit"
PROTOCOL = ROOT / "engineering/configs/experiments/feedback-refit-transfer.json"


def sources():
    package, _, batches = official("gas")
    fit = [r for r in batches[0].samples if r.role == "fit"]
    train = np.asarray([r.values for r in fit])
    y = np.asarray([r.label for r in fit])
    samples = read(paths("gas", "gas")[0] / "state.json")["samples"]
    for name, stream, truth, budget, classes in islice(datasets(), 3):
        if name == "public":
            rows = {k: r["values"] for k,r in samples.items() if r["role"] in ("explore","adopt","serve")}
            if {r.sample_id for r in fit} & rows.keys():
                raise ValueError("源训练和现场样本重叠")
            yield name, train, y, rows, stream, truth, budget, classes
        elif name == "late":
            batches = read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip", expected_sha256=REGISTERED_SHA256)
            rows = {r.sample_id: r.values for b in batches if b.batch_number >= 3 for r in b.records}
            yield name, train, y, rows, stream, truth, budget, classes
        else:
            df, columns, _, fit_ix, _, test, _, labels, _ = source_frame()
            raw = df[columns].to_numpy(dtype=float)
            raw[~np.isfinite(raw)] = np.nan
            rows = {f"row-{i+2}": raw[i] for i in test}
            yield name, raw[fit_ix], labels[fit_ix], rows, stream, truth, budget, classes
    data = prepared()
    voc = data["A_y"]
    y = 1 + (voc > np.median(voc)).astype(int)
    for name, stream, truth, meta in streams():
        unit = meta["device"]
        new_ids, rows = data[unit+"_ids"], data[unit+"_x"]
        if set(new_ids) & set(data["A_ids"]):
            raise ValueError("源混合物重复")
        yield name, data["A_x"], y, dict(zip(map(str, new_ids), rows, strict=True)), stream, truth, 8, 2


def weights(y, mass):
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return mass / (len(counts) * counts[inverse])


def macro(matrix):
    den = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2*matrix.diagonal(), den, out=np.zeros(len(den)), where=den>0).mean())


def evaluate(name, source, labels, rows, stream, answers, budget, classes, *, seed=17):
    folder = DEST/name if seed == 17 else DEST/"feedback-seeds"/str(seed)/name
    use = np.isfinite(source).any(axis=0)
    med = np.nanmedian(source[:, use], axis=0)
    def clean(x):
        v = np.asarray(x)[:, use]
        return np.where(np.isfinite(v), v, med)
    scaler = StandardScaler().fit(clean(source))
    src = scaler.transform(clean(source))
    keys = list(rows)
    row_id = {k:i for i,k in enumerate(keys)}
    x = scaler.transform(clean([rows[k] for k in keys]))
    source_weights = weights(labels, len(labels))
    def fit(x, y, w):
        return LogisticRegression(C=1., random_state=17, max_iter=1000).fit(x, y, sample_weight=w)
    steady = fit(src, labels, source_weights)
    names = tuple(stream[0][1])
    observer = ObserverErrors(names, classes)
    confusion = {arm: np.zeros((classes,classes), dtype=int) for arm in ("refit", "observer")}
    history, returned_y = [], []
    used = set()
    spends = []
    receipts = []
    for n, (ids, votes, pool, fp, ys) in enumerate(stream):
        choose = n >= 2 and macro(confusion["refit"]) >= macro(confusion["observer"]) + .01
        model = steady
        if history:
            start = process_time()
            new_y = np.asarray(returned_y)
            model = fit(np.vstack((src, x[history])), np.concatenate((labels,new_y)),
                        np.concatenate((source_weights, weights(new_y,len(labels)))))
            spends.append(process_time()-start)
        features = x[[row_id[k] for k in ids]]
        calibration = observer.predict(votes,"uniform")
        candidate = model.predict(features)
        p = {**votes, "vote": majority(votes,classes), "source-logistic": steady.predict(features),
             "refit": candidate, "observer": calibration, "adaptive": candidate if choose else calibration}
        ix = sorted(range(len(pool)), key=lambda i:digest([seed,pool[i]]))[:budget] if n+1<len(stream) else []
        return_ids = [pool[i] for i in ix]
        sampled = {k:np.asarray(v)[ix] for k,v in fp.items()}
        feedback = {}
        if ix:
            feedback = {"refit": model.predict(x[[row_id[k] for k in return_ids]]).tolist(),
                        "observer": observer.predict(sampled,"uniform").tolist()}
        path = folder/f"batch-{n+1:02}.json"
        atomic_json(path, {"ids": list(ids), "training_target_rows": len(history), "use_refit": bool(choose),
                           "previous_macro": {a: macro(v) for a,v in confusion.items()},
                           "predictions": {k: v.tolist() for k,v in p.items()}, "returned_next": return_ids,
                           "feedback_predictions": feedback})
        saved = read(path)
        receipts.append(sha256(path.read_bytes()).hexdigest())
        if ix:
            if used.intersection(return_ids):
                raise ValueError("回标重复")
            used.update(return_ids)
            truth = np.asarray([ys[i] for i in ix],dtype=int)
            for a,v in saved["feedback_predictions"].items():
                np.add.at(confusion[a], (truth-1,np.asarray(v)-1),1)
            observer.update(truth,sampled)
            history.extend(row_id[k] for k in return_ids)
            returned_y.extend(truth.tolist())
    logs = [read(folder/f"batch-{n+1:02}.json") for n in range(len(stream))]
    keys = [k for r in logs for k in r["ids"]]
    if len(keys) != len(set(keys)):
        raise ValueError("服务样本重复")
    y = np.array([answers[k] for k in keys])
    blocks = np.split(np.arange(len(y)),np.cumsum([len(r["ids"]) for r in logs])[:-1])
    pred = {a: np.concatenate([r["predictions"][a] for r in logs]) for a in logs[0]["predictions"]}
    table = {a: scores(y,v,blocks,classes) for a,v in pred.items()}
    best = max((*names,"vote","source-logistic"),key=lambda a:table[a]["pooled"])
    changed = pred["adaptive"] != pred["observer"]
    result = {"arms":table,"feedback_seed":seed,"best_fixed":best,"target_fit_calls":len(spends),"target_fit_cpu_seconds":sum(spends),
              "selected_refit_batches":[i+1 for i,r in enumerate(logs) if r["use_refit"]], "used_labels":len(used),
              "train_rows":len(source),"train_digest":digest([src.tolist(),labels.tolist()]),"rows":len(y),
              "rescued":int(np.sum(changed & (pred["adaptive"]==y))),
              "spoiled":int(np.sum(changed & (pred["observer"]==y))), "receipts":receipts,
              "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest()}
    atomic_json(folder/"scores.json",result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-seeds", action="store_true", help="仅改变回标抽取，复用项目已有的十三个种子")
    args = parser.parse_args()
    overview = {}
    for study in sources():
        name = study[0]
        runs = [evaluate(*study, seed=s) for s in (SEEDS if args.feedback_seeds else (17,))]
        cols = ("observer","refit","adaptive")
        overview[name] = {
            "seeds": [r["feedback_seed"] for r in runs],
            "macro_f1": {a:float(np.mean([r["arms"][a]["pooled"] for r in runs])) for a in cols},
            "worst_recall": {a:float(np.mean([min(r["arms"][a]["recall"]) for r in runs])) for a in cols},
            "refit_minus_observer": [r["arms"]["refit"]["pooled"]-r["arms"]["observer"]["pooled"] for r in runs],
            "adaptive_minus_best_of_both": [r["arms"]["adaptive"]["pooled"]-
                max(r["arms"][a]["pooled"] for a in ("observer","refit")) for r in runs]}
        print(name, overview[name]["macro_f1"], flush=True)
    if args.feedback_seeds:
        atomic_json(DEST/"feedback-seeds/scores.json",overview)
