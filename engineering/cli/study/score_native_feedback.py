"""原包模型与 Adam 原样恢复；在同一回标分割上比较直接交接、只看源数据及现场确认后交接。"""
from collections import defaultdict
from hashlib import sha256
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from compare_group_objectives import _trainer
from gas_complementary_feedback import OUT, ROOT, atomic_json, digest, read
from score_feedback_refit import sources


DEST = OUT / "native-feedback"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-native-feedback.json"


def balanced_draw(y, size, rng):
    groups = [np.flatnonzero(y == g) for g in np.unique(y)]
    return np.asarray([rng.choice(groups[i % len(groups)]) for i in range(size)])


def run(trainer, parent, name, source, labels, rows, stream, answers, budget, classes):
    folder = DEST / name
    store = folder / "weights"
    store.mkdir(parents=True, exist_ok=True)
    def load(path):
        return trainer.Model.load(path)
    def classify(model, x):
        return model.predict(x).argmax(1)+1
    fixed = load(parent)
    deployed = dict.fromkeys(("source-only","mixed","confirmed"), parent)
    fit, fit_y, confirm, confirm_y = [], [], [], []
    trace = []
    pred = defaultdict(list)
    all_ids, batches = [], []
    for n,(ids, _, pool, _, y_pool) in enumerate(stream):
        event = {"step": n+1, "fit_target": len(fit), "confirm": len(confirm), "updates": {}}
        if n:
            rng = np.random.default_rng(np.random.SeedSequence([17,n]))
            first = balanced_draw(labels,len(source),rng)
            source_second = balanced_draw(labels,len(source),rng)
            chosen = balanced_draw(np.asarray(fit_y),len(source),rng)
            x1, y1 = source[first], labels[first]
            recipe = {"source-only": (source[source_second],labels[source_second]),
                      "mixed": (np.asarray([rows[fit[i]] for i in chosen]),np.asarray(fit_y)[chosen])}
            for arm, path in deployed.copy().items():
                x2, y2 = recipe["source-only" if arm=="source-only" else "mixed"]
                candidate = load(path)
                start = process_time()
                steps = candidate.meta["steps"]
                candidate.fit(np.vstack((x1,x2)),np.concatenate((y1,y2))-1,1)
                elapsed = process_time()-start
                candidate_path = store/f"{arm}-{n:02}.npz"
                candidate.save(candidate_path)
                instance = load(candidate_path)
                use = True
                info = {"parent_sha256": sha256(path.read_bytes()).hexdigest(),
                        "candidate_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
                        "steps": instance.meta["steps"]-steps, "cpu": elapsed}
                if arm == "confirmed":
                    v = np.asarray([rows[i] for i in confirm])
                    old, new = classify(load(path),v), classify(instance,v)
                    before, after = [int(np.sum(q==confirm_y)) for q in (old,new)]
                    use = after > before
                    info.update(previous_correct=before,candidate_correct=after,confirm_ids=confirm)
                if use:
                    deployed[arm] = candidate_path
                info["deployed"] = use
                event["updates"][arm] = info
        values = np.asarray([rows[i] for i in ids])
        issued = {"original":classify(fixed,values), **{a:classify(load(p),values) for a,p in deployed.items()}}
        path = folder/f"batch-{n+1:02}.json"
        atomic_json(path, {"ids":ids,"event":event,"weights":{a:sha256(p.read_bytes()).hexdigest() for a,p in deployed.items()},
                           "predictions":{a:v.tolist() for a,v in issued.items()}})
        record = read(path)
        trace.append(sha256(path.read_bytes()).hexdigest())
        for a,v in record["predictions"].items():
            pred[a].extend(v)
        batches.append(np.arange(len(all_ids),len(all_ids)+len(ids)))
        all_ids.extend(ids)
        if n+1<len(stream):
            ranks = sorted(range(len(pool)), key=lambda i:digest([17,pool[i]]))[:budget]
            previous = read(OUT/"feedback-refit"/name/f"batch-{n+1:02}.json")["returned_next"]
            if [pool[i] for i in ranks] != previous:
                raise ValueError("回标与实验原始上限、身份或顺序不同")
            train_ix, check_ix = ranks[:len(ranks)//2], ranks[len(ranks)//2:]
            fit.extend(pool[i] for i in train_ix)
            fit_y.extend(int(y_pool[i]) for i in train_ix)
            confirm = [pool[i] for i in check_ix]
            confirm_y = np.asarray([y_pool[i] for i in check_ix])
            if set(fit)&set(confirm):
                raise ValueError("回标拟合与采用资料发生重合")
    y = np.array([answers[k] for k in all_ids])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("服务行重复")
    table = {a:scores(y,np.asarray(v),batches,classes) for a,v in pred.items()}
    best = max(("original","source-only","mixed"),key=lambda a:table[a]["pooled"])
    gate = (table["confirmed"]["pooled"]>=table[best]["pooled"]+.01 and
            min(table["confirmed"]["recall"])>=min(table[best]["recall"])-.02)
    files = [read(folder/f"batch-{n:02}.json") for n in range(2,len(stream)+1)]
    accepted = [r["event"]["step"] for r in files if r["event"]["updates"]["confirmed"]["deployed"]]
    cpu = {a:sum(r["event"]["updates"][a]["cpu"] for r in files) for a in deployed}
    steps = {a:sum(r["event"]["updates"][a]["steps"] for r in files) for a in deployed}
    atomic_json(folder/"scores.json", {"arms":table,"accepted_batches":accepted,"receipts":trace,
                "training_cpu_seconds":cpu,"updates":steps,"training_rows_per_attempt":2*len(source),
                "best_fixed":best,"meets_gate":gate,"original_sha256":sha256(parent.read_bytes()).hexdigest(),
                "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest()})
    print(name, {a:round(v["pooled"],5) for a,v in table.items()},accepted,flush=True)


if __name__ == "__main__":
    archive, = (ROOT/"data/official/gas").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"A") as trainer:
        parent = Path(tmp)/"root.npz"
        parent.write_bytes(z.read("checkpoint.npz"))
        for study in islice(sources(),2):
            run(trainer,parent,*study)
