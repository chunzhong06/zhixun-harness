"""把逐飞行的工况参照迁至赛题单条输入，判断两种采样条件下的收益是否相同。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from score_engine_full_condition import cohorts
from score_engine_stage_profiles import PREVIOUS
from score_engine_full_condition_forward import ROOT, STUDY, released, read, digest, summary, _trainer
from score_completed_residual_transfer import official
from harness.models.full_condition_engine import FullConditionEngine
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent / "engine-load-snapshot"


def transform(healthy, x):
    return np.column_stack((x[:, :21] - healthy.predict(x[:, 21:]), x[:, 21:]))


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有单条工况参照预测")
        pkg, _, batches = official("engine")
        _, repair, blob = released(STUDY.parent / "engine-continued-integrated/engine", "native_artifact")
        archive, = (ROOT / "data/official/engine").glob("*.zip")
        with TemporaryDirectory(dir=DEST) as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "C") as trainer:
            cond, center, model, weight = FullConditionEngine.load_bundle(blob, trainer)
            if weight != read(PREVIOUS / "training.json")["weights"]["epoch-60"]:
                raise ValueError("冻结起点不符")
            if any(r.role != "fit" or r.label is None for r in pkg.train):
                raise ValueError("训练池中出现其他用途的数据")
            x = np.asarray([r.values for r in pkg.train])
            early = x[[r.time_end <= 8 for r in pkg.train]]
            if len(early) != 8 * len({r.unit_id for r in pkg.train}):
                raise ValueError("源参考行数与开工八次不符")
            y = np.array([r.label for r in pkg.train])
            healthy = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.))
            healthy.fit(early[:, 21:], early[:, :21])
            def features(values):
                return {"原始森林": values, "连续偏差森林": transform(healthy, values), "离散工况森林": cond.transform(values)}
            bank = {k: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=17, n_jobs=1).fit(z, y)
                    for k, z in features(x).items()}
            joblib.dump((healthy, bank), DEST / "tools.joblib")
            def forecast(values):
                v = repair.transform(values)
                base = np.maximum(0, model.predict(FullConditionEngine.inputs(cond, center, model, v)))
                pred = {k: bank[k].predict(z) for k, z in features(v).items()}
                pred.update({k + "各半": (p + base) * .5 for k, p in list(pred.items())})
                return {"原版": base, **pred}
            def feedback(role):
                rows = [r for r in batches[0].samples if r.role == role and r.available_at == 0]
                ids = np.array([r.unit_id for r in rows])
                targets = np.array([r.label for r in rows])
                losses = {}
                for k, pred in forecast([r.values for r in rows]).items():
                    per = [np.mean((pred[ids == u] - targets[ids == u]) ** 2) for u in np.unique(ids)]
                    losses[k] = (float(np.mean(per)), float(np.sqrt(max(per))))
                return losses, set(ids)
            explore, fit_units = feedback("explore")
            adopt, adopt_units = feedback("adopt")
            if fit_units & adopt_units:
                raise ValueError("提名和采用使用了相同设备")
            nominated = min(explore, key=lambda k: explore[k][0])
            chosen = nominated if (adopt[nominated][0] < adopt["原版"][0]
                                    and adopt[nominated][1] <= 1.01 * adopt["原版"][1]) else "原版"
            previous = read(PREVIOUS / "scores.json")
            receiver = LocalAcceptance()
            receiver.output = DEST / "accepted"
            requests, pending = {}, {}
            for site, keys, values, owners, answers, _ in cohorts("engine", pkg):
                req = previous[site]["request"]
                prior = read(PREVIOUS / "accepted" / (req + ".json"))["payload"]
                if digest([site, prior]) != req or list(keys) != prior["ids"]:
                    raise ValueError("服务输入身份或顺序不一致")
                pred = forecast(values)
                np.testing.assert_allclose(pred["原版"], [prior["predictions"]["epoch-60"][k] for k in keys], atol=1e-8, rtol=0)
                pred["首批选用"] = pred[chosen]
                payload = {"ids": list(keys), "predictions": {k: p.tolist() for k, p in pred.items()}, "choice": chosen}
                key = digest([site, payload])
                ack = receiver.submit(key, payload)
                if not ack["accepted"] or ack["request_id"] != key:
                    raise RuntimeError("发动机单条预测未交接")
                requests[site], pending[site] = key, (owners, answers)
            results = {}
            for site, key in requests.items():
                p = read(receiver.output / (key + ".json"))["payload"]
                if digest([site, p]) != key:
                    raise ValueError("已交付预测身份不符")
                issued = {k: dict(zip(p["ids"], v, strict=True)) for k, v in p["predictions"].items()}
                owners, answers = pending[site]
                table = summary("engine", answers(), {"parent": issued["原版"], **issued}, owners)
                del table["parent"]
                print(json.dumps({site: {k: {"误差": v["score"], "最差台": v["worst_group"]} for k, v in table.items()}},
                                 ensure_ascii=False), flush=True)
                results[site] = {"request": key, "scores": table}
            atomic_json(DEST / "scores.json", {"choice": chosen, "nomination": nominated, "explore": explore,
                                               "confirmation": adopt, "sites": results})


if __name__ == "__main__":
    main()
