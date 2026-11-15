"""从干净源轨迹构造专家；部署时不输入仿真子集、单位号、运行年龄或标签。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

from score_engine_full_condition import cohorts
from score_engine_stage_profiles import PREVIOUS
from score_engine_full_condition_forward import ROOT, STUDY, released, read, digest, summary, _trainer
from score_completed_residual_transfer import official
from harness.models.full_condition_engine import FullConditionEngine
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent / "engine-fault-families"


def fit_bank(x, y, units):
    names, inverse = np.unique(units, return_inverse=True)
    flags = np.array([int(u.startswith(("FD003:", "FD004:"))) for u in names])
    if {u.split(":")[0] for u in names} != {"FD001","FD002","FD003","FD004"}:
        raise ValueError("源机队身份缺失")
    settings = dict(iterations=160, depth=6, learning_rate=.1, random_seed=17, thread_count=1,
                    allow_writing_files=False, verbose=False)
    pooled = CatBoostRegressor(**settings).fit(x, y)
    pooled.save_model(str(DEST / "pooled.cbm"))
    banks = {}
    for name, device_flags in (("故障族", flags), ("乱组", np.random.default_rng(17).permutation(flags))):
        group = device_flags[inverse]
        gate = CatBoostClassifier(**settings).fit(x, group)
        gate.save_model(str(DEST / (name + "-gate.cbm")))
        experts = []
        for g in range(2):
            expert = CatBoostRegressor(**settings).fit(x[group == g], y[group == g])
            expert.save_model(str(DEST / f"{name}-{g}.cbm"))
            experts.append(expert)
        banks[name] = (gate, experts)
    return pooled, banks


def main():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("本次故障族回放已有结果")
        pkg, _, _ = official("engine")
        if any(r.role != "fit" or r.label is None for r in pkg.train):
            raise ValueError("源端训练行不合法")
        _, repair, blob = released(STUDY.parent / "engine-continued-integrated/engine", "native_artifact")
        archive, = (ROOT / "data/official/engine").glob("*.zip")
        with TemporaryDirectory(dir=DEST) as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "C") as trainer:
            cond, center, model, weight = FullConditionEngine.load_bundle(blob, trainer)
            if weight != read(PREVIOUS / "training.json")["weights"]["epoch-60"]:
                raise ValueError("原版身份改变")
            x = center.transform(cond.transform([r.values for r in pkg.train]))
            y = np.array([r.label for r in pkg.train])
            pooled, banks = fit_bank(x, y, [r.unit_id for r in pkg.train])
            receiver = LocalAcceptance()
            receiver.output = DEST / "accepted"
            previous = read(PREVIOUS / "scores.json")
            pending = {}
            for site, keys, values, owners, answers, _ in cohorts("engine", pkg):
                z = center.transform(cond.transform(repair.transform(values)))
                base = np.maximum(0, model.predict(z * model.scale + model.mean))
                req = previous[site]["request"]
                old = read(PREVIOUS / "accepted" / (req + ".json"))["payload"]
                if digest([site, old]) != req or list(keys) != old["ids"]:
                    raise ValueError("旧参照样本发生变化")
                np.testing.assert_allclose(base, [old["predictions"]["epoch-60"][k] for k in keys], rtol=0, atol=1e-8)
                pred = {"不分组": pooled.predict(z, thread_count=1)}
                for name, (gate, experts) in banks.items():
                    prob = gate.predict_proba(z, thread_count=1)[:, 1]
                    a, b = (m.predict(z, thread_count=1) for m in experts)
                    pred.update({name+"一": a, name+"二": b, name+"固定各半": (a+b)/2,
                                 name+"按读数": a*(1-prob)+b*prob})
                pred.update({name+"与原版各半": .5*(base+p) for name,p in list(pred.items())})
                pred["原版"] = base
                payload = {"ids": list(keys), "predictions": {k: np.maximum(0,p).tolist() for k,p in pred.items()}}
                key = digest([site,payload])
                ack = receiver.submit(key,payload)
                if not ack["accepted"] or ack["request_id"] != key:
                    raise RuntimeError("故障族服务预测未交接")
                pending[site] = key, owners, answers
            results = {}
            for site,(key,owners,answers) in pending.items():
                payload = read(receiver.output/(key+".json"))["payload"]
                if digest([site,payload]) != key:
                    raise ValueError("已交预测身份不符")
                out = {k:dict(zip(payload["ids"],v,strict=True)) for k,v in payload["predictions"].items()}
                table = summary("engine", answers(), {"parent":out["原版"],**out}, owners)
                del table["parent"]
                results[site] = {"request":key,"scores":table}
                print(json.dumps({site:{k:{"误差":v["score"],"最差台":v["worst_group"]} for k,v in table.items()}},
                                 ensure_ascii=False),flush=True)
            atomic_json(DEST/"scores.json",results)


if __name__ == "__main__":
    main()
