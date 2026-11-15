"""按原采集批发放已冻结气体输出，回标在当批交付以后才用于下批选择。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, metric, read
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts


def fixed_outputs(batches):
    signatures = {}
    def issued(name):
        folder = BASE/name
        path = folder/"issued.json"
        signatures[name] = sha256(path.read_bytes()).hexdigest()
        if signatures[name]!=read(folder/"scores.json")["issued_sha256"]:
            raise ValueError("已封存的源预测身份不符")
        return read(path)
    old = issued("gas-concentration")["predictions"]
    svm = issued("gas-sensor-ranks")["outputs"]
    arms = {name:old[k] for name,k in (("current","project"),("paper","paper"),("original","original"))}
    folder = BASE/"selected"/"gas"
    first = read(folder/"run.json")["rounds"][0]
    header,blob = ModelArtifacts(folder/"models").read(first["model_artifact"])
    signatures["single"] = header["sha256"]
    archive, = (ROOT/"data/official/gas").glob("*.zip")
    rows = [r for b in batches for r in b.records]
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"A") as trainer:
        path = Path(tmp)/"first.npz"
        path.write_bytes(blob)
        model = pair_model(trainer.Model).load(path)
        y = model.predict(np.array([r.values for r in rows])).argmax(axis=1)+1
        arms["single"] = dict(zip((r.sample_id for r in rows),y.tolist(),strict=True))
    arms.update({k:v["candidate"] for k,v in svm.items()})
    return {a:{r.sample_id:p[r.sample_id] for r in rows} for a,p in arms.items()},signatures


def pick(candidates,answers):
    if not answers:
        return "current"
    return max(candidates,key=lambda a:metric("gas",answers,{i:candidates[a][i] for i in answers}))


def evaluate():
    folder = BASE/"gas-returned-labels"
    batches = [b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
               if b.batch_number>=3]
    candidates,signatures = fixed_outputs(batches)
    issued, traces = dict(candidates), {}
    for seed in (17,47,89):
        last, cumulative = {}, {}
        for batch in batches:
            keys = [r.sample_id for r in batch.records]
            sampled = sorted(keys,key=lambda i:digest([seed,i]))[:32]
            for mode,available in (("last",last),("cumulative",cumulative)):
                key = f"{mode}-{seed}"
                choice = pick(candidates,available)
                issued.setdefault(key,{}).update({i:candidates[choice][i] for i in keys})
                traces.setdefault(key,[]).append({"batch":batch.batch_number,"choice":choice,
                    "feedback_seen":len(available),"returned_after_service":sampled})
            selected = set(sampled)
            last = {r.sample_id:r.label for r in batch.records if r.sample_id in selected}
            cumulative.update(last)
    path = folder/"issued.json"
    atomic_json(path,{"frozen_sources":signatures,"seeds":[17,47,89],"predictions":issued,"traces":traces})
    delivered = read(path)["predictions"]
    truth = {r.sample_id:r.label for b in batches for r in b.records}
    scores, oracle, oracle_trace = {}, {}, []
    for b in batches:
        available = {r.sample_id:r.label for r in b.records}
        choice = pick(candidates,available)
        oracle_trace.append(choice)
        oracle.update({i:candidates[choice][i] for i in available})
    for arm,stream in {**delivered,"posthoc-batch-best":oracle}.items():
        if stream.keys()!=truth.keys():
            raise ValueError("缺少同批交付的测量")
        values = [metric("gas",truth,{r.sample_id:stream[r.sample_id] for r in b.records}) for b in batches]
        scores[arm] = {"pooled":metric("gas",truth,stream),"equal_batches":float(np.mean(values)),
                       "worst_batch":min(values),"by_batch":values}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":scores,
                                      "oracle_by_batch":oracle_trace,"fixed":list(candidates)})
    for arm,row in scores.items():
        print(arm,*(round(row[k],5) for k in ("pooled","equal_batches","worst_batch")))


if __name__=="__main__":
    evaluate()
