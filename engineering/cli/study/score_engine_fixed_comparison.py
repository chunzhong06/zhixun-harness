"""在相同源训练行上交叉比较工况处理与预测器，交接预测后再计分。"""
import json
from hashlib import sha256
from time import process_time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from score_engine_early_reference import DEST as EARLIER, official, read, frozen, tree_values, cohorts, summary
from harness.contracts import digest
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = EARLIER.parent / "engine-fixed-comparison"


def train_models(rows, source):
    x = np.asarray([r.values for r in rows])
    y = np.asarray([r.label for r in rows])
    base = frozen("engine")[0]
    training = {}
    arms = {}
    for name in ("raw", "uniform", "early"):
        scaler = None
        tree = base["catboost"]
        if name != "raw":
            path = EARLIER / (name + ".joblib")
            if sha256(path.read_bytes()).hexdigest() != source["models"][name]["sha256"]:
                raise ValueError("工况参照与先前交接不符")
            prior = joblib.load(path)
            scaler = prior.named_steps["condition"]
            tree = prior.named_steps["tree"]
        z = scaler.transform(x) if scaler is not None else x
        learners = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)), "forest": clone(base["forest"])}
        for kind, estimator in learners.items():
            start = process_time()
            estimator.fit(z, y)
            seconds = process_time() - start
            file = DEST / (name + "-" + kind + ".joblib")
            joblib.dump(estimator, file)
            training[file.name] = {"sha256":sha256(file.read_bytes()).hexdigest(),
                                   "cpu_seconds":seconds,"bytes":file.stat().st_size}
            learners[kind] = joblib.load(file)
        arms[name] = (scaler, tree, learners)
    return arms, training


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此固定比较已有完整结果")
        source = read(EARLIER/"training.json")
        package, _, batches = official("engine")
        ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        rows = [r for r in package.train if r.sample_id in ids]
        if [r.sample_id for r in rows] != source["fit_ids"]:
            raise ValueError("寿命训练行顺序变化")
        repair = SourceCombinations([r.values for r in rows])
        arms, training = train_models(rows, source)
        atomic_json(DEST/"training.json",{"source_ids":source["fit_ids"],"repair":repair.identity,"fit":training})
        prior_scores = read(EARLIER/"scores.json")
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        results = {}
        for site, keys, features, groups, answers, _ in cohorts("engine",package):
            previous = prior_scores[site]["request"]
            expected = read(EARLIER/"accepted"/(previous+".json"))["payload"]
            if digest([site,expected]) != previous or list(keys) != expected["ids"]:
                raise ValueError("旧交付与本次服务测量不同")
            x = repair.transform(np.asarray(features))
            pred = {name:expected["predictions"][name] for name in ("current-auto","paper-auto")}
            negatives = {}
            for name,(scaler,tree,learners) in arms.items():
                z = scaler.transform(x) if scaler is not None else x
                tree_pred = tree_values(tree,"engine",z)
                old = expected["predictions"]["tree-auto" if name=="raw" else name]
                np.testing.assert_allclose(tree_pred,[old[k] for k in keys],rtol=0,atol=1e-7)
                forest = learners["forest"].predict(z)
                ridge = learners["ridge"].predict(z)
                values = {"tree":tree_pred,"forest":forest,"ridge":ridge,
                          "forest-half":(forest+ridge)/2,"tree-half":(tree_pred+ridge)/2}
                for kind,v in values.items():
                    arm = name+"-"+kind
                    if not np.isfinite(v).all():
                        raise ValueError("预测含非有限值")
                    negatives[arm] = int(np.count_nonzero(v<0))
                    pred[arm] = dict(zip(keys,np.maximum(v,0).tolist(),strict=True))
            payload = {"ids":list(keys),"parent_request":previous,"training":digest(training),
                       "clipped":negatives,"predictions":pred}
            request = digest([site,payload])
            ack = receiver.submit(request,payload)
            if not ack["accepted"] or ack["request_id"]!=request:
                raise RuntimeError("发动机预测未交接")
            sealed = read(receiver.output/(request+".json"))["payload"]["predictions"]
            scores = summary("engine",answers(),{"parent":sealed["current-auto"],**sealed},groups)
            results[site] = {"request":request,"scores":scores,"clipped":negatives}
            print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in scores.items()}},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",results)


if __name__=="__main__":
    main()
