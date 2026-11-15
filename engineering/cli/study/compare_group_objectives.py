"""从同一官方权重比较一次续训的分组目标，用同一冻结候选预测各来源的服务样本。"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from math import ceil, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time_ns, perf_counter_ns
from zipfile import ZipFile

import numpy as np

from compare_proposal_frequency import (
    BASE, ROOT, LocalStore, atomic_json, digest, metric, official, read,
    reference_folders, remaining_cycles, remaining_minutes,
)
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.models.official_rul_cpu import _trainer

DEST = BASE / "group-objectives"
PROFILES = {"gas": "A", "bearing": "B", "engine": "C"}
RATE = 0.01


def weighted_epoch(model, x, y, groups, rule):
    """保留随包洗牌、梯度裁剪及 Adam 历史，仅替换小批中各行在目标函数的权重。"""
    if rule not in ("equal", "dro"):
        raise ValueError("未知训练目标")
    meta = model.meta
    x = model.normalized(x)
    classifier = meta["task"] == "classification"
    targets = y if classifier else (y-meta["target_mean"])/meta["target_scale"]
    log_weights = np.zeros(int(groups.max())+1)
    order = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epochs"]])).permutation(len(x))
    for start in range(0, len(x), meta["batch_size"]):
        ids = order[start:start+meta["batch_size"]]
        batch, target, group = x[ids], targets[ids], groups[ids]
        hidden, logits = model.forward(batch)
        if classifier:
            exp = np.exp(logits-logits.max(axis=1, keepdims=True))
            delta = exp/exp.sum(axis=1, keepdims=True)
            loss = -np.log(np.maximum(delta[np.arange(len(ids)), target], 1e-300))
            delta[np.arange(len(ids)), target] -= 1
        else:
            error = logits[:, 0]-target
            loss, delta = error**2, (2*error)[:, None]
        counts = np.bincount(group, minlength=len(log_weights))
        if rule == "dro":
            loss_sum = np.bincount(group, weights=loss, minlength=len(counts))
            log_weights += RATE*loss_sum/np.maximum(counts, 1)
        q = np.exp(log_weights-log_weights.max())
        q /= q.sum()
        # 作者实现对缺席组给零梯度。总权重按全部组归一，等权对照使用相同分母。
        delta *= (q[group]/counts[group])[:, None]
        hidden_delta = (delta @ model.params["w2"].T) * (1-hidden**2)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                     "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
        norm = sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
        meta["steps"] += 1
        for key, gradient in gradients.items():
            gradient *= min(1., 5./max(norm, 1e-12))
            model.first[key] = .9*model.first[key] + .1*gradient
            model.second[key] = .999*model.second[key] + .001*gradient**2
            first = model.first[key]/(1-.9**meta["steps"])
            second = model.second[key]/(1-.999**meta["steps"])
            model.params[key] -= meta["learning_rate"]*first/(np.sqrt(second)+1e-8)
    meta["epochs"] += 1
    model.validate()
    return q.tolist()


def train_candidates(task, trainer, parent, rows, work):
    x = np.array([r.values for r in rows])
    y = np.array([r.label for r in rows])
    group_names, groups = np.unique([str(r.label) if task=="gas" else r.group for r in rows], return_inverse=True)
    if task == "gas":
        y = y-1
    model = trainer.Model.load(parent)
    models, receipts = {"parent":model}, {}
    steps = ceil(len(x)/model.meta["batch_size"])
    for rule in ("ordinary", "equal", "dro"):
        child = trainer.Model.load(parent)
        cpu, wall = process_time_ns(), perf_counter_ns()
        if rule == "ordinary":
            child.fit(x, y, 1)
            weights = None
        else:
            weights = weighted_epoch(child, x, y, groups, rule)
        seconds, elapsed = (process_time_ns()-cpu)/1e9, (perf_counter_ns()-wall)/1e9
        path = work/(rule+".npz")
        child.save(path)
        models[rule] = child = trainer.Model.load(path)
        if (child.meta["steps"] != model.meta["steps"]+steps or child.meta["epochs"] != model.meta["epochs"]+1
                or not np.array_equal(child.mean,model.mean) or not np.array_equal(child.scale,model.scale)):
            raise ValueError("训练没有从原权重接续相同遍数或标准化已变")
        receipts[rule] = {"weights_sha256":sha256(path.read_bytes()).hexdigest(), "steps":steps,
                          "final_group_weights":weights, "cpu_seconds":seconds, "wall_seconds":elapsed}
    return models, {"parent_sha256":sha256(parent.read_bytes()).hexdigest(), "fit_ids": [r.sample_id for r in rows],
                    "groups":group_names.tolist(), "group_counts":np.bincount(groups).tolist(), "models":receipts}


def coverage_rows(task, rows, pool):
    """匹配每遍的分组数和 Adam 步数。轮换只访问官方训练行，每次按原包顺序送入训练器。"""
    key = (lambda r: str(r.label)) if task=="gas" else (lambda r:r.group)
    quotas = Counter(key(r) for r in rows)
    used = {r.sample_id for r in rows}
    pools = {g: sorted([r for r in pool if key(r)==g and r.sample_id not in used],
                        key=lambda r:digest([17,r.sample_id])) for g in quotas}
    epochs = [rows]
    if task != "gas":
        for step in range(3):
            ids = {r.sample_id for g,n in quotas.items() for r in pools[g][step*n:(step+1)*n]}
            chosen = [r for r in pool if r.sample_id in ids]
            if Counter(key(r) for r in chosen)!=quotas or used & ids:
                raise ValueError("可用训练行不满足跨遍配额")
            used.update(ids)
            epochs.append(chosen)
    return epochs


def train_coverage(task, trainer, parent, rows, pool, work):
    epochs = coverage_rows(task,rows,pool)
    recipes = {"ordinary":[rows], "repeat":[rows]*4}
    if task != "gas":
        recipes["rotate"] = epochs
    models = {"parent":trainer.Model.load(parent)}
    receipts = {}
    for name, sequence in recipes.items():
        child = trainer.Model.load(parent)
        elapsed, cpu = 0., 0.
        for epoch, sample in enumerate(sequence):
            x = np.array([r.values for r in sample])
            y = np.array([r.label for r in sample])-(1 if task=="gas" else 0)
            start, wall = process_time_ns(), perf_counter_ns()
            child.fit(x, y, 1)
            cpu += (process_time_ns()-start)/1e9
            elapsed += (perf_counter_ns()-wall)/1e9
            if epoch==0 and name!="ordinary":
                for tensors in ("params","first","second"):
                    actual, expected = getattr(child,tensors), getattr(models["ordinary"],tensors)
                    if any(not np.array_equal(actual[k],expected[k]) for k in actual):
                        raise ValueError("两条四遍轨迹第一遍的权重或 Adam 状态不一致")
        path = work/(name+".npz")
        child.save(path)
        models[name] = trainer.Model.load(path)
        receipts[name] = {"weights_sha256":sha256(path.read_bytes()).hexdigest(),
                          "unique_rows":len({r.sample_id for s in sequence for r in s}),
                          "row_visits":sum(map(len,sequence)), "epochs":len(sequence),
                          "steps":child.meta["steps"]-models["parent"].meta["steps"],
                          "fit_ids_sha256":[digest([r.sample_id for r in s]) for s in sequence],
                          "cpu_seconds":cpu, "wall_seconds":elapsed}
    return models, {"parent_sha256":sha256(parent.read_bytes()).hexdigest(), "pool":len(pool), "models":receipts}


def predict_all(task, models, ids, x):
    outputs, timing = {}, {}
    for name, model in models.items():
        start, wall = process_time_ns(), perf_counter_ns()
        values = model.predict(x)
        if len(values) != len(ids) or not np.isfinite(values).all():
            raise ValueError("候选预测缺少有效输出")
        if task == "gas":
            values = values.argmax(axis=1)+1
        timing[name] = {"cpu_seconds":(process_time_ns()-start)/1e9, "wall_seconds":(perf_counter_ns()-wall)/1e9}
        outputs[name] = dict(zip(ids, values.tolist(), strict=True))
    return outputs, timing


def delivered(folder):
    stream = {}
    report = read(folder/"run.json")
    if len(report["rounds"]) != 10:
        raise ValueError("历史参照缺少十轮交接")
    for batch in report["rounds"]:
        payload = read(folder/batch["submitted"])["payload"]
        pred = payload["predictions"]
        if stream.keys() & pred.keys() or payload["model_id"] != batch["selected"]:
            raise ValueError("交接中出现重复测量或错误的采用权重")
        stream.update(pred)
    return stream


def summary(task, answers, predictions, devices):
    keys = list(predictions["parent"])
    y = np.array([answers[i] for i in keys])
    membership = y if task=="gas" else np.array([devices[i] for i in keys])
    groups = {str(g): membership==g for g in np.unique(membership)}
    def errors(p):
        return np.asarray(p != y, dtype=float) if task=="gas" else np.abs(p-y)
    base = errors(np.array([predictions["parent"][i] for i in keys]))
    result = {}
    for name, pred in predictions.items():
        if pred.keys() != predictions["parent"].keys():
            raise ValueError("参照方法与候选使用的服务测点不同")
        p = np.array([pred[i] for i in keys])
        loss = errors(p)
        scores = {g:float(1-loss[mask].mean() if task=="gas" else np.sqrt((loss[mask]**2).mean()))
                  for g,mask in groups.items()}
        result[name] = {"score":metric(task, answers, pred), "rows":len(p),
                        "better_than_parent":int((loss<base).sum()), "worse_than_parent":int((loss>base).sum()),
                        "worst_group":(min if task=="gas" else max)(scores.values()), "group_scores":scores}
    return result


def evaluate(task, coverage=False):
    package, checksum, batches = official(task)
    fit_ids = {r.sample_id for b in batches for r in b.samples if r.role=="fit"}
    rows = [r for r in package.train if r.sample_id in fit_ids]
    if len(rows)!=len(fit_ids) or fit_ids & {r.sample_id for r in package.validation}:
        raise ValueError("官方训练池与评价划分不符")
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    output = (BASE/"training-volume" if coverage else DEST)/task
    with LocalStore(output).exclusive(), TemporaryDirectory(prefix="group-objective-") as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), PROFILES[task]) as trainer:
        work = Path(tmp)
        parent = work/"parent.npz"
        parent.write_bytes(package.checkpoint)
        if coverage:
            models, training = train_coverage(task, trainer, parent, rows, package.train, work)
        else:
            models, training = train_candidates(task, trainer, parent, rows, work)
            training["group_step_size"] = RATE
        training.update(package_sha256=checksum, implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
        cohorts = [task] if task=="gas" else [task, "pronostia" if task=="bearing" else "phm08"]
        results, timings = {}, {}
        for cohort in cohorts:
            fixed, paper = reference_folders(cohort,"mature")
            state = read(fixed/"state.json")
            samples = state["samples"]
            if {i for i,s in samples.items() if s["role"]=="fit"} != fit_ids:
                raise ValueError("跨来源对照的训练行发生变化")
            ids = sorted(i for i,s in samples.items() if s["role"]=="serve")
            if any(samples[i]["label"] is not None for i in ids):
                raise ValueError("服务行带有目标答案")
            predictions, timings[cohort] = predict_all(task, models, ids, np.array([samples[i]["values"] for i in ids]))
            path = output/(cohort+"-issued.json")
            atomic_json(path, {"input_sha256":digest([(i,samples[i]["values"]) for i in ids]), "predictions":predictions})
            original = BASE/"original"/task if task==cohort else fixed.parent/"original"
            static = delivered(original)
            if set(static)!=set(ids) or not np.allclose([static[i] for i in ids], list(predictions["parent"].values()), rtol=1e-12, atol=1e-8):
                raise ValueError("未复现同源、同起点的原版预测")
            old = BASE/"selected"/task if task==cohort else fixed.parent/"project"
            predictions.update(old=delivered(old), mature=delivered(paper))
            answers = {r.sample_id:r.label for r in package.validation}
            if cohort=="phm08":
                answers = remaining_cycles()
            if cohort=="pronostia":
                answers = remaining_minutes("full-test-manifest.json")
            results[cohort] = summary(task, answers, predictions, {i:samples[i]["unit_id"] for i in ids})
            results[cohort]["issued_sha256"] = sha256(path.read_bytes()).hexdigest()
        if task=="gas":
            for batch in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256):
                if batch.batch_number < 3:
                    continue
                cohort = f"gas-batch-{batch.batch_number:02}"
                ids = [r.sample_id for r in batch.records]
                data = np.array([r.values for r in batch.records])
                predictions, timings[cohort] = predict_all(task, models, ids, data)
                path = output/(cohort+"-issued.json")
                atomic_json(path, {"archive":REGISTERED_SHA256,"input_sha256":digest(ids),"predictions":predictions})
                results[cohort] = summary(task, {r.sample_id:r.label for r in batch.records},predictions,{})
                results[cohort]["issued_sha256"] = sha256(path.read_bytes()).hexdigest()
        atomic_json(output/"scores.json",{"training":training,"prediction_times":timings,"cohorts":results})
        for cohort, arms in results.items():
            print(cohort,{arm:round(data["score"],5) for arm,data in arms.items() if isinstance(data,dict)})


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=PROFILES)
    parser.add_argument("--coverage",action="store_true")
    args = parser.parse_args()
    evaluate(args.task,args.coverage)
