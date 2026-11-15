"""由官方原权重产生基础票，同组比较训练样本分量，并从已交付文件评分。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from score_gas_probability_fusion import (
    ROOT, BASE, ARCHIVE, SOURCE, SEEDS, SiteTraining, GasFeedbackBank, LocalAcceptance,
    LocalStore, read_official_gas, atomic_json, read, digest, metrics, _trainer,
)
from harness.models.field_classification import weights, fixed_snapshot, model_scores

DEST = BASE / "whole-maintenance/gas-label-mass"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-label-mass.json"
MODES = ("domain", "ramp", "per-class", "per-row")


def measurements(package):
    samples = read(SOURCE / "state.json")["samples"].values()
    bank = GasFeedbackBank(package.train)
    batches = []
    with TemporaryDirectory(dir=DEST) as temp, ZipFile(ARCHIVE) as archive:
        folder = Path(temp)
        official = archive.read("checkpoint.npz")
        weight_id = sha256(official).hexdigest()
        with _trainer(archive, folder, "A") as trainer:
            path = folder / "unmodified.npz"
            path.write_bytes(official)
            model = trainer.Model.load(path)
            for j in range(10):
                rows = [r for r in samples if r["available_at"] == j]
                roles = {role: sorted((r for r in rows if r["role"] == role),key=lambda r:r["sample_id"])
                         for role in ("serve","adopt","explore")}
                service = roles["serve"]
                if any(r["label"] is not None for r in service):
                    raise ValueError("服务行带有答案")
                feedback = roles["explore"] + roles["adopt"]
                x = np.asarray([r["values"] for r in service])
                fx = np.asarray([r["values"] for r in feedback])
                batches.append(([r["sample_id"] for r in service], x, bank.predict(model,x),
                                [r["sample_id"] for r in feedback], fx, bank.predict(model,fx),
                                [r["label"] for r in feedback]))
    return batches, weight_id


def adjusted_predictions(field, which_scale, x, snapshot_dir=None):
    history = np.asfortranarray(field.history)
    labels = np.asarray(field.answers)
    scale = field.source_scale if which_scale == "source" else StandardScaler().fit(history)
    z = np.vstack((field.source_z, scale.transform(history)))
    targets = np.r_[field.labels, labels]
    source_n = len(field.labels)
    by_class = np.bincount(labels, minlength=field.classes + 1)
    capped = source_n / field.classes / np.maximum(by_class[labels],32 / field.classes)
    weighting = {
        "domain":weights(labels,source_n),
        "ramp":weights(labels,source_n * min(1.,len(labels)/32)),
        "per-row":weights(labels,len(labels)),
        "per-class":capped,
    }
    predictions = {}
    snapshots = {}
    for name, target_weights in weighting.items():
        model = LogisticRegression(C=1.,random_state=17,max_iter=1000).fit(
            z,targets,sample_weight=np.r_[field.source_weights,target_weights])
        record = fixed_snapshot("mixed",(scale,model))
        identity = digest(record)
        file = (snapshot_dir if snapshot_dir is not None else DEST / "snapshots") / (identity+".json")
        if not file.exists():
            atomic_json(file,record)
        predictions[name] = model_scores(read(file),x,field.classes).argmax(1)+1
        snapshots[name] = identity
    return predictions,snapshots


def score(seed,cap,package,batches,parent):
    field = SiteTraining([r.values for r in package.train],[r.label for r in package.train],
                         GasFeedbackBank.names,6,"pooled-midway-cold")
    folder = DEST / f"cap-{cap}" / f"seed-{seed}"
    receiver = LocalAcceptance()
    receiver.output = folder / "accepted"
    requests = []
    feedback_ids = set()
    for j,(ids,x,votes,pool,fx,fv,answers) in enumerate(batches):
        selected = sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:cap] if j<9 else []
        seen = len(field.answers)
        observer = field.observer.predict(votes,"uniform")
        preds = {"observer":observer}
        snapshots = {}
        if seen:
            source, snapshots["source"] = adjusted_predictions(field,"source",x)
            if seen>=160 or field.rounds>=5:
                other,snapshots["site"] = adjusted_predictions(field,"site",x)
            else:
                other = source
        else:
            source = other = dict.fromkeys(MODES,observer)
        for name in MODES:
            preds[name+"-source"] = source[name]
            preds[name+"-160"] = (other if seen>=160 else source)[name]
        preds["current"] = (other if field.rounds>=5 else source)["domain"]
        preds["wait32"] = preds["domain-160"] if seen>=32 else observer
        payload = {"ids":ids,"parent":parent,"round":j+1,"prior_answers":seen,"snapshots":snapshots,
                   "observer_counts":field.observer.counts.tolist(),"predictions":{k:v.tolist() for k,v in preds.items()}}
        request = digest([digest(read(PROTOCOL)),seed,cap,payload])
        ack = receiver.submit(request,payload)
        if ack["request_id"]!=request or not ack["accepted"]:
            raise RuntimeError("交付未成功")
        requests.append(request)
        if selected:
            returned = [pool[i] for i in selected]
            labels = [answers[i] for i in selected]
            if feedback_ids.intersection(returned):
                raise ValueError("反馈身份重复")
            feedback_ids.update(returned)
            field.observer.update(labels,{k:v[selected] for k,v in fv.items()})
            field.history.extend(fx[selected].tolist())
            field.answers.extend(labels)
            field.rounds += 1
            atomic_json(folder/"feedback"/(request+".json"),{"ids":returned,"labels":labels})
    logs = [read(receiver.output/(key+".json"))["payload"] for key in requests]
    ids = [i for row in logs for i in row["ids"]]
    if len(set(ids))!=len(ids) or len(ids)!=414 or len(feedback_ids)!=9*cap or feedback_ids.intersection(ids):
        raise ValueError("服务与反馈身份或数量不符合协议")
    truth = {r.sample_id:r.label for r in package.validation}
    y = [truth[i] for i in ids]
    scores = {k:metrics(y,np.concatenate([row["predictions"][k] for row in logs])) for k in preds}
    atomic_json(folder/"scores.json",{"requests":requests,"scores":scores})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此冻结协议已有结果")
        package = read_official_gas(ARCHIVE)
        batches,parent = measurements(package)
        result = {}
        for cap in (1,4,8,16,32):
            draws = [score(seed,cap,package,batches,parent) for seed in SEEDS]
            means = {name:{k:float(np.mean([d[name][k] for d in draws])) for k in ("macro","worst")}
                     for name in draws[0]}
            result[cap] = {"means":means,"default":draws[0]}
            print(json.dumps({f"额度{cap}":means},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),
                    "original_sha256":parent,"budgets":result,"seeds":list(SEEDS)})


if __name__ == "__main__":
    main()
