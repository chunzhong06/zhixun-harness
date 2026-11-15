"""在同一已交接模型上改变续训输入，分别计分同父版候选及各分支真正使用的模型。"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import archived, source_window, unlabelled
from compare_group_objectives import (
    BASE, ROOT, PROFILES, LocalStore, _trainer, atomic_json, delivered, digest, metric, official,
    read, reference_folders, remaining_cycles, remaining_minutes, summary,
)
from compare_proposal_frequency import TASKS
from harness.runtime.handoff import LocalAcceptance

OUT = BASE / "sensor-pairs"
SEEDS = (17, 47, 89)
BLOCKS = {"gas": (16, 8), "bearing": (2, 17), "engine": (21, 1)}


class SensorPairs:
    """供体完全取自本轮合法 fit 清单。发动机只移动测量值，保留三个控制量。"""

    def __init__(self, task, rows):
        self.blocks, self.width = BLOCKS[task]
        self.x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows])
        self.y = y - 1 if task == "gas" else y
        self.band = 0. if task == "gas" else max(1., .05 * np.subtract(*np.quantile(y, [.75, .25])))
        units = np.asarray([getattr(r, "unit_id", "") for r in rows])
        groups = np.asarray([getattr(r, "group", "") for r in rows])
        self.donors = []
        for i, label in enumerate(y):
            eligible = np.abs(y - label) <= self.band
            if task != "gas":
                eligible &= units != units[i]
            if task == "engine":
                eligible &= groups == groups[i]
            eligible[i] = False
            self.donors.append(np.flatnonzero(eligible))

    def inputs(self, model, seed):
        """两臂共用锚点及强度。每行都核对原模型实际看见的标准化位移长度。"""
        rng = np.random.default_rng(np.random.SeedSequence([seed, model.meta["epochs"]]))
        cols = np.arange(self.blocks * self.width)
        z = model.normalized(self.x)[:, cols]
        joint, split = self.x.copy(), self.x.copy()
        touched = 0
        for i in rng.permutation(len(z))[:len(z)//2]:
            if len(self.donors[i]) < 2:
                continue
            a, b = rng.choice(self.donors[i], 2, replace=False)
            use_a = rng.integers(0, 2, self.blocks).astype(bool)
            k, l = rng.choice(self.blocks, 2, replace=False)
            use_a[k], use_a[l] = True, False
            whole_delta = z[a] - z[i]
            split_delta = np.where(use_a.repeat(self.width), z[a], z[b]) - z[i]
            n_joint, n_split = np.linalg.norm(whole_delta), np.linalg.norm(split_delta)
            length = .25 * min(n_joint, n_split)
            if length == 0:
                continue
            for inputs, delta, norm in ((joint, whole_delta, n_joint), (split, split_delta, n_split)):
                inputs[i, cols] = (z[i] + delta * (length / norm)) * model.scale[cols] + model.mean[cols]
            touched += 1
        norms = [np.linalg.norm(model.normalized(x)[:, cols] - z, axis=1) for x in (joint, split)]
        if not np.allclose(*norms, atol=1e-9, rtol=1e-9):
            raise ValueError("两种扰动在官方模型实际坐标中的强度不等")
        return {"joint": joint, "split": split}, touched


def predict(task, model, rows):
    values = model.predict(np.asarray([r["values"] for r in rows]))
    if not np.isfinite(values).all():
        raise ValueError("模型预测非有限值")
    if task == "gas":
        values = values.argmax(axis=1) + 1
    return dict(zip([r["sample_id"] for r in rows], values.tolist(), strict=True))


def candidate(parent, pairs, seed, recipe, work):
    trial = deepcopy(parent)
    x, touched = pairs.x, 0
    start = perf_counter()
    if recipe != "ordinary":
        options, touched = pairs.inputs(parent, seed)
        x = options[recipe]
    trial.fit(x, pairs.y, 1)
    path = work / "candidate.npz"
    trial.save(path)
    trial = type(parent).load(path)
    identity = sha256(path.read_bytes()).hexdigest()
    path.unlink()
    steps = trial.meta["steps"] - parent.meta["steps"]
    if (steps != ceil(len(x)/parent.meta["batch_size"])
            or trial.meta["epochs"] != parent.meta["epochs"] + 1
            or not np.array_equal(trial.mean, parent.mean) or not np.array_equal(trial.scale, parent.scale)):
        raise ValueError("训练状态、样本数或优化步数不符")
    return trial, identity, {
        "steps": steps, "changed_inputs": touched, "fit_wall_seconds": perf_counter() - start,
    }


def run_branch(task, parent, identity, pairs, seed, recipe, windows, work, endpoint, external):
    current = parent
    reuse = None
    first = None
    stream, history = {}, []
    for ordinal, (explore, confirm, serve) in enumerate(windows, start=5):
        if any(r["label"] is not None for r in serve):
            raise ValueError("服务答案提前进入交接")
        if reuse is None:
            trial, trial_id, build = candidate(current, pairs, seed, recipe, work)
        else:
            trial, trial_id = reuse
            build = None
        if first is None:
            first = trial
        event = {"round": ordinal, "parent": identity, "trial": trial_id, "build": build}
        passed = False
        for role, rows in (("explore", explore), ("confirm", confirm)):
            if not rows:
                break
            labels = {r["sample_id"]: r["label"] for r in rows}
            before, after = [metric(task, labels, predict(task, m, rows)) for m in (current, trial)]
            event[role] = [before, after]
            gain = after - before if task == "gas" else before - after
            if gain <= 1e-12:
                break
            passed = role == "confirm"
        if passed:
            current, identity, reuse = trial, trial_id, None
        else:
            reuse = (trial, trial_id)
        pred = predict(task, current, serve)
        if stream.keys() & pred.keys():
            raise ValueError("不同轮次交接了相同样本")
        payload = {"round": ordinal, "model": identity, "predictions": pred}
        request = digest(payload)
        if endpoint.submit(request, payload)["accepted"] is not True:
            raise ValueError("本地交接未确认")
        history.append({**event, "adopted": passed, "issued": request})
        stream.update(pred)
    serving = [r for _, _, rows in windows for r in rows]
    return stream, history, predict(task, first, serving), predict(task, first, external), predict(task, current, external)


def compare(cohort, *, class_directions=False):
    task = TASKS[cohort]
    package, checksum, _ = official(task)
    fixed, paper = reference_folders(cohort, "mature")
    old = BASE/"selected"/task if cohort == task else fixed.parent/"project"
    original = BASE/"original"/task if cohort == task else fixed.parent/"original"
    samples = read(fixed/"state.json")["samples"]
    fit_ids = {i for i, row in samples.items() if row["role"] == "fit"}
    rows = [r for r in package.train if r.sample_id in fit_ids]
    if len(rows) != len(fit_ids):
        raise ValueError("训练行不完全来自官方 fit 清单")
    if class_directions:
        if task!="gas":
            raise ValueError("同类方向只适用于气体分类")
        from gas_class_directions import ClassDirections
        pairs = ClassDirections(rows)
    else:
        pairs = SensorPairs(task, rows)
    windows = [source_window(samples, i) for i in range(4, 10)]
    service_ids = {r["sample_id"] for _, _, serving in windows for r in serving}
    ext_blocks, ext_truth = unlabelled(task)
    external = [r for block in ext_blocks for r in block]
    if (any(r["label"] is not None for r in external)
            or {r["sample_id"] for r in external} & set(samples)):
        raise ValueError("移用样本答案或来源隔离有误")
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    dest = BASE/"gas-class-directions" if class_directions else OUT/cohort
    with LocalStore(dest).exclusive(), TemporaryDirectory(prefix="sensorpairs-") as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), PROFILES[task]) as trainer:
        parent, identity, _ = archived(old, trainer, Path(tmp), task)
        refs = {k: delivered(f) for k, f in (("parent", old), ("fixed", fixed), ("paper", paper), ("original", original))}
        prefix = {i: v for i, v in refs["parent"].items() if i not in service_ids}
        round4 = source_window(samples, 3)[2]
        check = predict(task, parent, round4)
        if not np.allclose(list(check.values()), [prefix[i] for i in check], rtol=1e-12, atol=1e-8):
            raise ValueError("未复现第4轮在用交接")
        first = {"parent": predict(task, parent, [r for _, _, s in windows for r in s])}
        external_first = {"parent": predict(task, parent, external)}
        external_final = external_first.copy()
        histories = {}
        endpoint = LocalAcceptance()
        endpoint.output = dest/"accepted"
        recipes = ("ordinary","principal","random") if class_directions else ("ordinary","split","joint")
        for recipe in recipes:
            for seed in ((17,) if recipe == "ordinary" else SEEDS):
                name = recipe if recipe == "ordinary" else f"{recipe}-{seed}"
                pred, histories[name], first[name], external_first[name], external_final[name] = run_branch(
                    task, parent, identity, pairs, seed, recipe, windows, Path(tmp), endpoint, external)
                refs[name] = {**prefix, **pred}
        issued = dest/"issued.json"
        record = {"whole": refs, "first_candidate": first, "external_first": external_first,
                  "external_terminal": external_final, "start": identity, "fit_ids": digest([r.sample_id for r in rows]),
                  "package_sha256": checksum}
        if issued.exists() and digest(read(issued)) != digest(record):
            raise ValueError("重跑未复现已封存的预测")
        atomic_json(issued, record)
        # 全部预测已写出；独立评分只通过服务编号关联答案。
        truth = {r.sample_id: r.label for r in package.validation}
        if cohort == "pronostia":
            truth = remaining_minutes("full-test-manifest.json")
        elif cohort == "phm08":
            truth = remaining_cycles()
        groups = {i: r["unit_id"] for i, r in samples.items() if r["role"] == "serve"}
        ext_groups = {r["sample_id"]: r["unit_id"] for r in external}
        result = {"whole": summary(task, truth, refs, groups), "first_candidate": summary(task, truth, first, groups),
                  "external_first": summary(task, ext_truth, external_first, ext_groups),
                  "external_terminal": summary(task, ext_truth, external_final, ext_groups), "history": histories,
                  "fit_rows": len(rows),
                  "issued_sha256": sha256(issued.read_bytes()).hexdigest(),
                  "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
        if class_directions:
            result["source_variation"] = pairs.last_diagnostics
            result["by_batch"] = {}
            for name,streams in (("candidate",external_first),("delivered",external_final)):
                result["by_batch"][name] = {arm:[metric(task,ext_truth,{r["sample_id"]:p[r["sample_id"]] for r in b})
                                                     for b in ext_blocks] for arm,p in streams.items()}
        else:
            result.update(eligible=sum(len(d)>=2 for d in pairs.donors),band=pairs.band)
        atomic_json(dest/"scores.json", result)
        print(cohort, {k: round(v["score"], 5) for k, v in result["whole"].items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=TASKS)
    parser.add_argument("--class-directions",action="store_true")
    args=parser.parse_args()
    compare(args.cohort,class_directions=args.class_directions)
