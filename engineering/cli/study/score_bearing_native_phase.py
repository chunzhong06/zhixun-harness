"""把阶段树的实际输出迁入赛方轴承权重，对照固定时长、真实比例和打乱的教师预测。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, tree_values
from compare_group_objectives import ROOT, _trainer, official, read
from harness.contracts import digest
from harness.models.training_tools import SklearnBackend
from harness.runtime.storage import atomic_json
from score_life_training_weights import cohorts
from score_native_bearing_choices import DEST as POOL, fingerprint, publish
from score_tree_hidden_units import choose_borders


DEST = POOL.parent / "bearing-native-phase"
PROTOCOL = ROOT / "engineering/configs/experiments/bearing-native-phase-teacher.json"


def train(tag, rows, parent, trainer, teacher, reference):
    x = np.array([r.values for r in rows])
    remaining = np.array([r.label for r in rows])
    total = np.array([r.time_end - 1 + r.label for r in rows])
    if np.any(total <= 0) or any(r.role != "fit" for r in rows):
        raise ValueError("教师学习仅接收已完成源设备的合法训练记录")
    base = trainer.Model.load(parent)
    z = base.normalized(x)
    scale = reference["scale"]
    target = scale * np.clip(tree_values(teacher, "bearing", x), 0, 1)
    source = base.predict(x)
    width = base.params["w1"].shape[1]
    started = perf_counter()
    selected, _ = choose_borders(z, target, teacher.named_steps["predictor"], base, width)
    if len(selected) != width:
        raise ValueError("源阶段树中有效边界不足")
    options = {"native-phase": (selected, target), "original-slots": ([], target),
               "true-phase": (selected, scale*remaining/total),
               "shuffled": (selected, np.random.default_rng(17).permutation(target)),
               "half-parent": ([], (source + target)/2)}
    folder = DEST/tag/"models"
    folder.mkdir(parents=True, exist_ok=True)
    models, stats = {}, {}
    for name, (splits, label) in options.items():
        model = trainer.Model.load(parent)
        for i, (axis, cut) in enumerate(splits):
            model.params["w1"][:,i] = 0
            model.params["w1"][axis,i] = 4.
            model.params["b1"][i] = -4*cut
        h, _ = model.forward(z)
        a = np.column_stack((h, np.ones(len(rows))))
        gram = a.T @ a + .001*len(rows)*np.eye(a.shape[1])
        norm = (label-model.meta["target_mean"])/model.meta["target_scale"]
        w = np.linalg.solve(gram, a.T@norm)
        model.params["w2"][:,0] = w[:-1]
        model.params["b2"][0] = w[-1]
        with TemporaryDirectory(dir=folder) as tmp:
            path = Path(tmp)/(name+".npz")
            model.save(path)
            final = folder/path.name
            path.replace(final)
        models[name] = trainer.Model.load(final)
        p = models[name].predict(x)
        if not np.allclose(p, model.predict(x), rtol=0, atol=1e-9):
            raise ValueError("新权重未能逐条恢复")
        stats[name] = {"sha256":fingerprint(final), "source_teacher_rmse": float(np.sqrt(np.mean((p-target)**2))),
                       "source_fit_rmse":float(np.sqrt(np.mean((p-remaining)**2)))}
    record = {"original_sha256": fingerprint(parent), "source_digest":digest([r.sample_id for r in rows]),
              "rows":len(rows), "teacher":reference, "models":stats, "source_wall_seconds":perf_counter()-started,
              "selected":selected, "ridge_solves":len(options), "adam_steps":0}
    atomic_json(DEST/tag/"training.json", record)
    return models, record


def main():
    pkg, _, batches = official("bearing")
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    phase = OUT/"phase/trained/bearing"
    ref = read(phase/"receipt.json")
    if fingerprint(phase/"phase.joblib") != ref["sha256"]:
        raise ValueError("冻结教师已变")
    teacher = SklearnBackend(execution_enabled=True).load((phase/"phase.joblib").read_bytes())
    small = [r for r in batches[0].samples if r.role == "fit"]
    reports = {}
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "B") as trainer:
        parent = Path(tmp)/"root.npz"
        parent.write_bytes(pkg.checkpoint)
        for tag, rows in (("small", small), ("full", pkg.train)):
            models, training = train(tag, rows, parent, trainer, teacher, ref)
            tables = {}
            for site, ids, x, groups, answers, _ in cohorts("bearing", pkg):
                src = POOL/"pool"/site
                if not src.is_dir():
                    continue
                path = src/"issued.json"
                if fingerprint(path) != read(src/"scores.json")["issued_sha256"]:
                    raise ValueError("部件对照已变")
                old = read(path)
                if list(ids) != old["ids"]:
                    raise ValueError("对照服务行有别")
                features = np.asarray(x)
                teacher_output = np.clip(tree_values(teacher, "bearing", features), 0, 1)*ref["scale"]
                if not np.allclose(teacher_output, old["predictions"]["phase"], atol=1e-8, rtol=0):
                    raise ValueError("输入与已验证的阶段树不符")
                p = {k: m.predict(features).tolist() for k, m in models.items()}
                p["half-life"] = np.full(len(ids), .5*ref["scale"]).tolist()
                contents = {"ids":ids, "devices":[groups[k] for k in ids],
                            "predictions":{**old["predictions"], **p}, "added_weights":training["models"],
                            "parent":pkg.checkpoint_sha, "protocol":fingerprint(PROTOCOL)}
                tables[site] = publish(DEST/tag/site, contents, answers)
            reports[tag] = tables
            print(tag, {k:[round(r["scores"][a]["rmse"],2) for a in
                           ("half-life","phase","native-phase","original-slots","true-phase","shuffled","half-parent")]
                        for k,r in tables.items()},flush=True)
    atomic_json(DEST/"scores.json", {"protocol":fingerprint(PROTOCOL), "runs":reports})


if __name__ == "__main__":
    main()
