"""逐个采集批发放全部气体预测；本批被抽中的真实答案仅在交接后开放。"""
import json
from hashlib import sha256

import numpy as np

import score_gas_label_mass as training
from score_gas_probability_fusion import (
    ARCHIVE, BASE, ROOT, SEEDS, SiteTraining, GasFeedbackBank, LocalAcceptance, LocalStore,
    atomic_json, digest, read, metrics, read_official_gas,
)
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches

DEST = BASE / "whole-maintenance/gas-label-mass-late"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-label-mass-late.json"
FROZEN = BASE / "gas-complementary-feedback/fixed.json"


def evaluate(seed, training_rows, batches, frozen):
    field = SiteTraining([r.values for r in training_rows],[r.label for r in training_rows],
                         GasFeedbackBank.names,6,"pooled-midway-cold")
    folder = DEST / f"seed-{seed}"
    receiver = LocalAcceptance()
    receiver.output = folder / "accepted"
    requests = []
    ids_seen = set()
    start = 0
    for j,(batch,keys) in enumerate(zip(batches,frozen["keys"],strict=True)):
        ids = [r.sample_id for r in batch.records]
        if ids!=keys or ids_seen.intersection(ids):
            raise ValueError("实际采集批与封存基础预测的样本不符")
        ids_seen.update(ids)
        stop = start+len(ids)
        supplied = {k:np.asarray(v[start:stop]) for k,v in frozen["predictions"].items()}
        votes = {k:supplied[k] for k in GasFeedbackBank.names}
        x = np.asarray([r.values for r in batch.records])
        observer = field.observer.predict(votes,"uniform")
        output = {"observer":observer,**{"single-"+k:v for k,v in supplied.items()}}
        snapshots = {}
        if field.history:
            src,snapshots["source"] = training.adjusted_predictions(field,"source",x,DEST/"snapshots")
            site,snapshots["site"] = training.adjusted_predictions(field,"site",x,DEST/"snapshots")
        else:
            src = site = dict.fromkeys(training.MODES,observer)
        for name in training.MODES:
            output[name+"-source"] = src[name]
            output[name+"-site"] = site[name]
            output[name+"-160"] = (site if len(field.answers)>=160 else src)[name]
        output["current"] = (site if field.rounds>=5 else src)["domain"]
        payload = {"ids":ids,"batch":batch.batch_number,"prior_answers":len(field.answers),
                   "observer_counts":field.observer.counts.tolist(),"snapshots":snapshots,
                   "predictions":{k:v.tolist() for k,v in output.items()}}
        request = digest([digest(read(PROTOCOL)),seed,payload])
        ack = receiver.submit(request,payload)
        if not ack["accepted"] or ack["request_id"]!=request:
            raise RuntimeError("批次未交接")
        requests.append(request)
        if j<len(batches)-1:
            ix = sorted(range(len(ids)),key=lambda i:digest([seed,ids[i]]))[:32]
            labels = [batch.records[i].label for i in ix]
            field.observer.update(labels,{k:v[ix] for k,v in votes.items()})
            field.history.extend(x[ix].tolist())
            field.answers.extend(labels)
            field.rounds += 1
            atomic_json(folder/"feedback"/(request+".json"),{"ids":[ids[i] for i in ix],"labels":labels})
        start = stop
    if len(ids_seen)!=12221 or len(field.answers)!=224:
        raise ValueError("服务或反馈数量与协议不符")
    logs = [read(receiver.output/(r+".json"))["payload"] for r in requests]
    rows = {r.sample_id:r.label for batch in batches for r in batch.records}
    truths = [np.asarray([rows[k] for k in row["ids"]]) for row in logs]
    scores = {}
    for name in logs[0]["predictions"]:
        predictions = [row["predictions"][name] for row in logs]
        pooled = metrics(np.concatenate(truths),np.concatenate(predictions))
        equal = np.mean([metrics(y,p)["macro"] for y,p in zip(truths,predictions,strict=True)])
        scores[name] = {**pooled,"equal_batches":float(equal)}
    atomic_json(folder/"scores.json",{"requests":requests,"scores":scores})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此协议已执行")
        package = read_official_gas(ARCHIVE)
        batches = [b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",
                    expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
        frozen = read(FROZEN)
        draws = [evaluate(seed,package.train,batches,frozen) for seed in SEEDS]
        means = {k:{m:float(np.mean([r[k][m] for r in draws])) for m in ("macro","worst","equal_batches")}
                 for k in draws[0]}
        atomic_json(DEST/"scores.json",{"means":means,"default":draws[0],"seeds":list(SEEDS),
                    "fixed_sha256":sha256(FROZEN.read_bytes()).hexdigest(),"protocol":sha256(PROTOCOL.read_bytes()).hexdigest()})
        print(json.dumps({"后八批均分":means},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
