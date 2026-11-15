"""从随包 Adam 接续，比较直接回归和预测分段存活时长的辅助训练。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, official
from score_life_training_weights import cohorts, score


DEST = OUT / "interval-training"


def auxiliary_epoch(model, z, normalized_life, fractions):
    """沿用原包的洗牌、裁剪和 Adam；辅助梯度只进入计算概率的隐藏单元。"""
    meta = model.meta
    order = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epochs"]])).permutation(len(z))
    count = fractions.shape[1]
    for start in range(0, len(order), meta["batch_size"]):
        ids = order[start:start + meta["batch_size"]]
        batch = z[ids]
        hidden, logits = model.forward(batch)
        delta = (2 * (logits[:, 0] - normalized_life[ids]) / len(ids))[:, None]
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        # p=(1+tanh(h))/2=sigmoid(2h)，交叉熵对 h 的导数等于 2(p-y)。
        probabilities = (1 + hidden[:, :count]) / 2
        hidden_delta[:, :count] += 2 * (probabilities - fractions[ids]) / (len(ids) * count)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                     "w1": batch.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
        norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients.values()))
        meta["steps"] += 1
        for name, gradient in gradients.items():
            gradient *= min(1., 5. / max(norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** meta["steps"])
            second = model.second[name] / (1 - .999 ** meta["steps"])
            model.params[name] -= meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
    meta["epochs"] += 1
    model.validate()


def save_readout(model_class, path, widths, x):
    """从已保存的同一候选导出固定读数头，明确与继续训练的分支分开。"""
    derived = model_class.load(path)
    hidden, _ = derived.forward(derived.normalized(x))
    reference = ((1 + hidden[:, :len(widths)]) / 2) @ widths
    derived.params["w2"].fill(0)
    derived.params["w2"][:len(widths), 0] = widths / (2 * derived.meta["target_scale"])
    derived.params["b2"][0] = (widths.sum() / 2 - derived.meta["target_mean"]) / derived.meta["target_scale"]
    derived.save(path.with_stem(path.stem + "-sum"))
    reloaded = model_class.load(path.with_stem(path.stem + "-sum"))
    if not np.allclose(reference, reloaded.predict(x), atol=1e-8, rtol=0):
        raise ValueError("赛方预测器的读数和独立分段求和不一致")
    return reloaded


def train(pkg, trainer, folder):
    if any(r.role != "fit" for r in pkg.train):
        raise ValueError("训练池混入非训练观测")
    x = np.asarray([r.values for r in pkg.train])
    y = np.asarray([r.label for r in pkg.train], dtype=float)
    edges = np.r_[0, np.quantile(y, np.arange(1, 9) / 8)]
    widths = np.diff(edges)
    if np.any(widths <= 0) or np.any(y < 0):
        raise ValueError("训练寿命需要八个有效的非负区间")
    interval_fraction = np.clip((y[:, None] - edges[:-1]) / widths, 0, 1)
    if not np.allclose(interval_fraction @ widths, y, rtol=0, atol=1e-8):
        raise ValueError("区间目标无法还原训练寿命")
    direct_fraction = np.broadcast_to(y[:, None] / edges[-1], interval_fraction.shape)
    parent_path = folder / "parent.npz"
    parent_path.write_bytes(pkg.checkpoint)
    if sha256(parent_path.read_bytes()).hexdigest() != pkg.checkpoint_sha:
        raise ValueError("源权重与登记不一致")
    models, receipts = {}, {}
    for name in ("direct", "direct-aux", "interval"):
        model = trainer.Model.load(parent_path)
        z = model.normalized(x)
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
        start_steps = model.meta["steps"]
        wall = cpu = 0.
        for epoch in range(1, 6):
            start, cpu_start = perf_counter(), process_time()
            if name == "direct":
                model.fit(x, y, 1)
            else:
                auxiliary_epoch(model, z, target, direct_fraction if name == "direct-aux" else interval_fraction)
            wall += perf_counter() - start
            cpu += process_time() - cpu_start
            if epoch not in (1, 5):
                continue
            arm = f"{name}-{epoch}"
            path = folder / (arm + ".npz")
            model.save(path)
            models[arm] = trainer.Model.load(path)
            if model.meta["steps"] - start_steps != epoch * ((len(x) + model.meta["batch_size"] - 1) // model.meta["batch_size"]):
                raise ValueError("候选的实际更新步数不一致")
            names = [arm]
            if name != "direct":
                readout = widths if name == "interval" else np.full(8, edges[-1] / 8)
                models[arm + "-sum"] = save_readout(trainer.Model, path, readout, x)
                names.append(arm + "-sum")
            for key in names:
                weights = folder / (key + ".npz")
                receipts[key] = {"sha256": sha256(weights.read_bytes()).hexdigest(), "cpu_seconds": cpu,
                                 "wall_seconds": wall, "updates": model.meta["steps"] - start_steps,
                                 "source_rmse": float(np.sqrt(np.mean((models[key].predict(x) - y) ** 2)))}
    atomic_json(folder / "training.json", {"parent": pkg.checkpoint_sha, "rows": len(y),
                "fit_ids": digest([r.sample_id for r in pkg.train]), "edges": edges.tolist(), "arms": receipts})
    return models


def main(task):
    pkg, _, batches = official(task)
    fit_ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    small = [r.label for r in pkg.train if r.sample_id in fit_ids]
    trees, _ = frozen(task)
    folder = DEST / task / "models"
    folder.mkdir(parents=True, exist_ok=True)
    archive, = (ROOT / "data/official" / task).glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "B" if task == "bearing" else "C") as trainer:
        models = train(pkg, trainer, folder)
        def forecast(x):
            return {"small-tree": tree_values(trees["catboost"], task, x),
                    "small-mean": np.full(len(x), np.mean(small)), "small-median": np.full(len(x), np.median(small)),
                    **{k: m.predict(x) for k, m in models.items()}}
        result = {c[0]: score(task, c, forecast, DEST, "small-tree") for c in cohorts(task, pkg)}
    atomic_json(DEST / task / "scores.json", result)
    for name, table in result.items():
        print(name, {k: round(table[k]["score"], 2) for k in ("current", "small-tree", "direct-5", "direct-aux-5", "interval-5", "interval-5-sum")})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("engine", "bearing"))
    main(parser.parse_args().task)
