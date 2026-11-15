"""源训练时准备分阶段估计，用已返回的其他设备答案决定是否替换实际预测。"""
from pathlib import Path
from collections import defaultdict
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

import numpy as np

from score_engine_overestimates import public_stream, failure_stream
from score_completed_residual_transfer import engines, official, summary
from score_engine_full_condition_forward import ROOT, STUDY, released, read, digest, _trainer
from harness.models.full_condition_engine import FullConditionEngine
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent / "engine-stage-profiles"
PREVIOUS = STUDY.parent / "engine-weight-average"
ORIGINAL = "原版"


class Profiles:
    def __init__(self, prediction, truth, age):
        self.ages = np.asarray(age)
        self.cuts = np.quantile(self.ages, [.25, .5, .75])
        edges = np.unique(np.quantile(prediction, np.linspace(0, 1, 9)))
        bins = np.searchsorted(edges[1:-1], prediction, side="right")
        self.curves = {}
        stage = np.searchsorted(self.cuts, self.ages, side="right")
        subsets = {"全程": np.ones(len(stage), dtype=bool)}
        subsets.update({f"阶段{i + 1}": stage == i for i in range(4)})
        for name, subset in subsets.items():
            centers, shifts = [], []
            for i in range(len(edges) - 1):
                keep = subset & (bins == i)
                if keep.any():
                    centers.append(float(np.mean(prediction[keep])))
                    shifts.append(float(np.sum(truth[keep] - prediction[keep]) / (32 + keep.sum())))
            self.curves[name] = (centers, shifts)

    def estimates(self, p):
        return {ORIGINAL: p, **{k: np.maximum(0, p + np.interp(p, *curve))
                                for k, curve in self.curves.items()}}

    def scheduled(self, site, clock):
        if site == "engine":
            subset = self.ages[self.ages > 20 * clock]
            if clock < 9:
                subset = subset[subset <= 20 * (clock + 1)]
            age = np.median(subset)
        else:
            age = clock
        return f"阶段{1 + np.searchsorted(self.cuts, age, side='right')}"


class Feedback:
    """设备是独立回标单位；PHM08只有其他机器已结束才能累积数据。"""
    def __init__(self):
        self.devices = set()
        self.keys = set()
        self.errors = defaultdict(dict)

    def add(self, rows, forecast, profiles):
        if not rows:
            return
        units = np.array([r[0] for r in rows])
        keys = [r[1] for r in rows]
        if len(set(keys)) != len(keys) or self.keys.intersection(keys):
            raise ValueError("同一回标被重复使用")
        self.keys.update(keys)
        self.devices.update(units)
        true = np.array([r[3] for r in rows], dtype=float)
        for name, pred in profiles.estimates(forecast([r[2] for r in rows])).items():
            for unit in np.unique(units):
                keep = units == unit
                old, count = self.errors[name].get(unit, (0., 0))
                self.errors[name][unit] = (old + np.sum((pred[keep] - true[keep])**2), count + keep.sum())

    def scores(self):
        return {k: np.mean([err / n for err, n in values.values()]) for k, values in self.errors.items()}


def issue(site, stream, truth, forecast, profiles):
    previous = read(PREVIOUS / "scores.json")[site]
    snapshot = read(PREVIOUS / "accepted" / (previous["request"] + ".json"))["payload"]
    if digest([site, snapshot]) != previous["request"]:
        raise ValueError("原版交接身份改变")
    expected = snapshot["predictions"]["epoch-60"]
    receiver = LocalAcceptance()
    receiver.output = DEST / site / "accepted"
    fit, confirm = Feedback(), Feedback()
    owners, requests, choices = {}, [], defaultdict(int)
    fit_devices, confirm_devices = set(), set()
    for clock, explore, adopt, targets in stream:
        if site == "engine":
            fit, confirm = Feedback(), Feedback()
        fit.add(explore, forecast, profiles)
        confirm.add(adopt, forecast, profiles)
        fit_devices.update(fit.devices)
        confirm_devices.update(confirm.devices)
        fit_error, confirm_error = fit.scores(), confirm.scores()
        proposed = min(fit_error, key=fit_error.get) if fit_error else ORIGINAL
        chosen = proposed if confirm_error and confirm_error[proposed] < confirm_error[ORIGINAL] else ORIGINAL
        fixed = profiles.scheduled(site, clock)
        keys = [k for k, _, _ in targets]
        if len(set(keys)) != len(keys) or set(keys).intersection(owners):
            raise ValueError("服务测点重复")
        arms = profiles.estimates(forecast([r[2] for r in targets]))
        np.testing.assert_allclose(arms[ORIGINAL], [expected[k] for k in keys], rtol=0, atol=1e-8)
        arms.update({"按时钟": arms[fixed], "按探索": arms[proposed], "经确认": arms[chosen]})
        owners.update((k, u) for k, u, _ in targets)
        choices[chosen] += len(keys)
        payload = {"ids": keys, "clock": clock, "explore": [r[1] for r in explore],
                   "adopt": [r[1] for r in adopt], "proposed": proposed, "chosen": chosen,
                   "scheduled": fixed, "predictions": {k: pred.tolist() for k, pred in arms.items()}}
        key = digest([site, payload])
        ack = receiver.submit(key, payload)
        if not ack.get("accepted") or ack["request_id"] != key:
            raise RuntimeError("阶段估计发放失败")
        requests.append(key)
    if fit_devices & confirm_devices or (fit_devices | confirm_devices) & set(owners.values()) or set(expected) != set(owners):
        raise ValueError("服务与回标的设备或交付清单不符")
    issued = defaultdict(dict)
    for key in requests:
        record = read(receiver.output / (key + ".json"))["payload"]
        if digest([site, record]) != key:
            raise ValueError("封存后预测改变")
        for arm, pred in record["predictions"].items():
            issued[arm].update(zip(record["ids"], pred, strict=True))
    scores = summary("engine", truth(), {"parent": issued[ORIGINAL], **issued}, owners)
    del scores["parent"]
    print(json.dumps({site: {arm: {"误差": val["score"], "最差台": val["worst_group"]}
                                    for arm, val in scores.items()}}, ensure_ascii=False), flush=True)
    return {"requests": requests, "prior": previous["request"], "scores": scores, "choices": choices}


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有阶段估计交付")
        pkg, checksum, _ = official("engine")
        report, repair, blob = released(STUDY.parent / "engine-continued-integrated/engine", "native_artifact")
        archive, = (ROOT / "data/official/engine").glob("*.zip")
        with TemporaryDirectory(dir=DEST) as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "C") as trainer:
            cond, center, model, weight = FullConditionEngine.load_bundle(blob, trainer)
            if weight != read(PREVIOUS / "training.json")["weights"]["epoch-60"] or checksum != read(PREVIOUS / "training.json")["source"]:
                raise ValueError("模型或训练包与封存身份不一致")
            def forecast(rows):
                x = FullConditionEngine.inputs(cond, center, model, repair.transform(np.asarray(rows)))
                return np.maximum(0, model.predict(x))

            if any(r.role != "fit" or r.label is None for r in pkg.train):
                raise ValueError("源训练行的用途不符")
            profiles = Profiles(forecast([r.values for r in pkg.train]), np.array([r.label for r in pkg.train]),
                                [r.time_end for r in pkg.train])
            atomic_json(DEST / "training.json", {"weight": weight, "package": checksum, "source": report["identity"],
                                                 "age_cuts": profiles.cuts.tolist(), "profiles": profiles.curves})
            results = {"engine": issue("engine", public_stream(), lambda: {r.sample_id: r.label for r in pkg.validation},
                                       forecast, profiles)}
            data = engines()
            results["phm08"] = issue("phm08", failure_stream(data), data[3], forecast, profiles)
            atomic_json(DEST / "scores.json", results)


if __name__ == "__main__":
    main()
