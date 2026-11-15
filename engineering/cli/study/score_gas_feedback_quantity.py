"""在完全相同的已返回标签上替换两个启用时机，评分只使用已交接预测。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from score_gas_probability_fusion import (
    ROOT, BASE, ARCHIVE, SEEDS, stream, read, metrics, read_official_gas,
    GasFeedbackBank, SiteTraining, LocalAcceptance, LocalStore, atomic_json, digest,
    ModelArtifacts, SOURCE, pair_model, _trainer,
)

DEST = BASE / "whole-maintenance/gas-feedback-quantity"
PRIOR = BASE / "whole-maintenance/gas-cold-start"
PROTOCOL = ROOT / "engineering/configs/experiments/gas-feedback-quantity.json"
BUDGETS = (1, 4, 8, 16, 32)


def source_stream(batches, bank):
    """小额度比较统一重载首轮权重，排除原交付后来基于额外回标的权重更新。"""
    first = read(SOURCE / "run.json")["rounds"][0]["model_artifact"]
    header, blob = ModelArtifacts(SOURCE / "models").read(first)
    with TemporaryDirectory(dir=DEST) as temporary, ZipFile(ARCHIVE) as archive:
        path = Path(temporary)
        with _trainer(archive, path, "A") as trainer:
            weights = path / "first.npz"
            weights.write_bytes(blob)
            model = pair_model(trainer.Model).load(weights)
            return [(ids, x, bank.predict(model, x), pool, fx, bank.predict(model, fx), y, header["sha256"])
                    for ids, x, _, pool, fx, _, y, _ in batches]


def evaluate(seed, cap, batches, package, original=False):
    field = SiteTraining([r.values for r in package.train], [r.label for r in package.train],
                         GasFeedbackBank.names, 6, "pooled-midway-cold")
    folder = DEST / ("original" if original else "source") / f"cap-{cap}" / f"seed-{seed}"
    endpoint = LocalAcceptance()
    endpoint.output = folder / "accepted"
    previous = PRIOR / f"seed-{seed}"
    reference = read(previous/"scores.json")["requests"] if original else []
    history = set()
    requests = []
    for j,(ids,x,votes,pool,fx,fv,pool_y,parent) in enumerate(batches):
        take = sorted(range(len(pool)),key=lambda i:digest([seed,pool[i]]))[:cap] if j+1<len(batches) else []
        returned = [pool[i] for i in take]
        if history.intersection(ids) or set(returned).intersection(history.union(ids)):
            raise ValueError("服务或反馈身份重叠")
        tools = field.predict(x,votes)
        observed = tools["observer"]
        source = tools["pooled-source"]
        separate = tools["pooled-separate"]
        clock = source if field.rounds<5 else separate
        count = source if len(history)<160 else separate
        predictions = {
            "observer":observed, "linear-midway":clock, "current":tools[field.current()],
            "delay-start":clock if len(history)>=32 else observed,
            "delay-scale":count if history else observed,
            "quantity":count if len(history)>=32 else observed,
            "start32-source":source if len(history)>=32 else observed,
            "start32-separate":separate if len(history)>=32 else observed,
            "source-cold":source if history else observed, "separate-cold":separate if history else observed,
        }
        if reference:
            old = read(previous/"accepted"/(reference[j]+".json"))["payload"]
            if old["ids"]!=ids:raise ValueError("参考行序不同")
            for arm in ("current","quantity"):
                np.testing.assert_array_equal(predictions[arm],old["predictions"]["fixed-midway"])
        snapshot = {"models":{k:field.models()[k] for k in ("pooled-source","pooled-separate")},
                    "observer":field.observer.counts.tolist(),"source_weight":parent}
        identity = digest(snapshot)
        atomic_json(DEST/"snapshots"/(identity+".json"),snapshot)
        payload = {"round":j+1,"ids":ids,"snapshot":identity,"history":len(history),
                   "predictions":{k:v.tolist() for k,v in predictions.items()}}
        request = digest([digest(read(PROTOCOL)),seed,cap,payload])
        ack = endpoint.submit(request,payload)
        if ack["request_id"]!=request or not ack["accepted"]:raise RuntimeError("交接未成功")
        requests.append(request)
        if take:
            votes_before = {k:v[take] for k,v in fv.items()}
            before = field.predict(fx[take],votes_before)
            revealed = [pool_y[i] for i in take]
            field.update(fx[take],votes_before,revealed,before)
            atomic_json(folder/"feedback"/(request+".json"),{"ids":returned,"labels":revealed})
            history.update(returned)
    logs = [read(endpoint.output/(r+".json"))["payload"] for r in requests]
    keys = [k for row in logs for k in row["ids"]]
    if len(set(keys))!=414 or len(keys)!=414 or len(history)!=9*cap or history.intersection(keys):
        raise ValueError("样本与额度不符")
    labels = {r.sample_id:r.label for r in package.validation}
    y = [labels[k] for k in keys]
    scores = {k:metrics(y,np.concatenate([row["predictions"][k] for row in logs]))
              for k in logs[0]["predictions"]}
    atomic_json(folder/"scores.json",{"scores":scores,"requests":requests})
    return scores


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成本协议")
        pkg = read_official_gas(ARCHIVE)
        bank = GasFeedbackBank(pkg.train)
        original = stream(pkg,bank)
        batches = source_stream(original,bank)
        result = {}
        for cap in BUDGETS:
            draws = [evaluate(s,cap,batches,pkg) for s in SEEDS]
            means = {a:{k:float(np.mean([d[a][k] for d in draws])) for k in ("macro","worst")} for a in draws[0]}
            result[cap] = {"means":means,"default":draws[0]}
            print(json.dumps({f"额度{cap}":means},ensure_ascii=False),flush=True)
        replay = [evaluate(seed,32,original,pkg,original=True) for seed in SEEDS]
        mean_replay = float(np.mean([draw["quantity"]["macro"] for draw in replay]))
        print(json.dumps({"原完整交付复现":mean_replay},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"protocol":sha256(PROTOCOL.read_bytes()).hexdigest(),"budgets":result,
                    "original_full_feedback_macro":mean_replay,
                    "seeds":list(SEEDS),"development_only":True})


if __name__=="__main__":
    main()
