"""单条只给传感器；预测本批时机队组成只能由上一次已交出的读数推断。"""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

from score_engine_stage_profiles import DEST as PREVIOUS, PREVIOUS as WEIGHTS
from score_engine_overestimates import public_stream, failure_stream
from score_completed_residual_transfer import engines, official, summary
from score_engine_full_condition_forward import ROOT, STUDY, released, read, digest, _trainer
from harness.models.full_condition_engine import FullConditionEngine
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent / "engine-stage-readings"


def composition(posteriors, source):
    """在先前读数上做混合比例估计，保留16个源观测的权重以免小批直接消去某阶段。"""
    if posteriors is None:
        return source
    prior = source.copy()
    likelihood = posteriors / source
    for _ in range(20):
        weights = likelihood * prior
        weights /= weights.sum(axis=1, keepdims=True)
        prior = (weights.sum(axis=0) + 16 * source) / (len(weights) + 16)
    return prior


def issue(site, stream, truth, predict, source, cuts, ages):
    endpoint = LocalAcceptance()
    endpoint.output = DEST / site / "accepted"
    old = read(WEIGHTS / "scores.json")[site]["request"]
    old_payload = read(WEIGHTS / "accepted" / (old + ".json"))["payload"]
    if digest([site, old_payload]) != old:
        raise ValueError("原参照发放身份不符")
    issued_before = old_payload["predictions"]["epoch-60"]
    previous = None
    requests, owners = [], {}
    for clock, _, _, rows in stream:
        keys = [r[0] for r in rows]
        if len(set(keys)) != len(keys) or set(keys).intersection(owners):
            raise ValueError("服务测点重复")
        prior = composition(previous, source)
        base, direct, stages, probabilities = predict([r[2] for r in rows])
        np.testing.assert_allclose(base, [issued_before[k] for k in keys], atol=1e-8, rtol=0)
        if site == "engine":
            in_window = ages[ages > 20 * clock]
            if clock < 9:
                in_window = in_window[in_window <= 20 * (clock + 1)]
            scheduled = np.searchsorted(cuts, np.median(in_window), side="right")
        else:
            scheduled = np.searchsorted(cuts, clock, side="right")
        updated = probabilities * (prior / source)
        updated /= updated.sum(axis=1, keepdims=True)
        arms = {"直接估计": direct, "按单条辨认": (stages * probabilities).sum(axis=1),
                "按上批读数": (stages * updated).sum(axis=1), "固定源比例": stages @ source,
                "按时钟": stages[:, scheduled]}
        arms.update({f"始终阶段{i+1}": stages[:, i] for i in range(4)})
        arms.update({name + "各半": (base + p) / 2 for name, p in list(arms.items())})
        arms["原版"] = base
        payload = {"ids": keys, "predictions": {k: np.maximum(0, p).tolist() for k, p in arms.items()},
                   "previous_request": requests[-1] if requests else None, "stage_prior": prior.tolist()}
        key = digest([site, payload])
        ack = endpoint.submit(key, payload)
        if not ack["accepted"] or ack["request_id"] != key:
            raise RuntimeError("发放失败")
        previous = probabilities
        owners.update((r[0], r[1]) for r in rows)
        requests.append(key)
    if owners.keys() != issued_before.keys():
        raise ValueError("覆盖与原版不一致")
    issued = defaultdict(dict)
    for key in requests:
        payload = read(endpoint.output / (key + ".json"))["payload"]
        if digest([site, payload]) != key:
            raise ValueError("已交预测被修改")
        for name, values in payload["predictions"].items():
            issued[name].update(zip(payload["ids"], values, strict=True))
    scores = summary("engine", truth(), {"parent": issued["原版"], **issued}, owners)
    del scores["parent"]
    print(json.dumps({site: {k: {"误差": v["score"], "最差台": v["worst_group"]} for k, v in scores.items()}},
                     ensure_ascii=False), flush=True)
    return {"requests": requests, "scores": scores}


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有交付结果")
        pkg, checksum, _ = official("engine")
        report, repair, blob = released(STUDY.parent / "engine-continued-integrated/engine", "native_artifact")
        training = read(PREVIOUS / "training.json")
        cuts = np.array(training["age_cuts"])
        archive, = (ROOT / "data/official/engine").glob("*.zip")
        with TemporaryDirectory(dir=DEST) as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "C") as trainer:
            cond, center, model, weight = FullConditionEngine.load_bundle(blob, trainer)
            if checksum != training["package"] or weight != training["weight"] or report["identity"] != training["source"]:
                raise ValueError("训练条件或权重改变")
            if any(r.role != "fit" or r.label is None for r in pkg.train):
                raise ValueError("源模型只允许训练行")
            x = center.transform(cond.transform([r.values for r in pkg.train]))
            y = np.array([r.label for r in pkg.train])
            ages = np.array([r.time_end for r in pkg.train])
            stage = np.searchsorted(cuts, ages, side="right")
            source = np.bincount(stage) / len(stage)
            settings = dict(iterations=160, depth=6, random_seed=17, thread_count=1,
                            learning_rate=.1, verbose=False, allow_writing_files=False)
            direct = CatBoostRegressor(**settings).fit(x, y)
            expert = CatBoostRegressor(**settings).fit(np.column_stack((x, stage)), y)
            gate = CatBoostClassifier(**settings).fit(x, stage)
            for name, tool in (("direct", direct), ("expert", expert), ("gate", gate)):
                tool.save_model(str(DEST / (name + ".cbm")))
            atomic_json(DEST / "training.json", {"package": checksum, "weight": weight, "cuts": cuts.tolist(),
                                                 "source_prior": source.tolist()})

            def predict(values):
                z = center.transform(cond.transform(repair.transform(values)))
                base = np.maximum(0, model.predict(z * model.scale + model.mean))
                staged = [expert.predict(np.column_stack((z, np.full(len(z), i))), thread_count=1) for i in range(4)]
                return base, direct.predict(z, thread_count=1), np.column_stack(staged), gate.predict_proba(z, thread_count=1)

            results = {"engine": issue("engine", public_stream(), lambda: {r.sample_id: r.label for r in pkg.validation},
                                       predict, source, cuts, ages)}
            data = engines()
            results["phm08"] = issue("phm08", failure_stream(data), data[3], predict, source, cuts, ages)
            atomic_json(DEST / "scores.json", results)


if __name__ == "__main__":
    main()
