"""复用作者的双向选样函数。先发布决定与逐条预测，再读未来段真值计分。"""
import argparse
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from compare_gas_paper_modules import (
    HLLM, PACKAGE, PROTOCOL, ROOT, SPLIT, OfficialFit, accuracy_score, atomic_json, choose,
    conditions, feature_schema, np, pd, predicted, published_method, read_gas_batches,
    REGISTERED_SHA256, repair, score, sha,
)

PLAN = ROOT / "engineering/configs/experiments/gas-subset-direction.json"
OLD = ROOT / "results/experiments/official-gas/module-smoke/measured/issued.jsonl"
PATCH = ROOT / "results/experiments/official-gas/first-proposal/patch/issued.jsonl"


def require_identity(path, expected):
    if sha(path) != expected:
        raise ValueError(f"冻结输入发生变化 {path.name}")


def compare(output):
    plan = json.loads(PLAN.read_bytes())
    frozen = json.loads(PROTOCOL.read_bytes())
    for path, expected in ((PACKAGE, frozen["package_sha256"]), (SPLIT, frozen["split_sha256"]),
                           (OLD, plan["old_issued_sha256"]), (PATCH, plan["project_issued_sha256"])):
        require_identity(path, expected)
    output.mkdir(parents=True, exist_ok=False)
    split = json.loads(SPLIT.read_bytes())["rounds"]
    with ZipFile(PACKAGE) as zipfile:
        train = list(map(json.loads, zipfile.read("public/train.jsonl").splitlines()))
        valid = {r["id"]: r for r in map(json.loads, zipfile.read("public/validation.jsonl").splitlines())}
        checkpoint = zipfile.read("checkpoint.npz")
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model

    with TemporaryDirectory(prefix="subset-fit-") as tmp:
        parent = Path(tmp) / "parent.npz"
        parent.write_bytes(checkpoint)
        require_identity(parent, frozen["parent_sha256"])
        base = Model.load(parent)
        label = base.meta["classes"].index
        train_ids = [r["id"] for r in train]
        x = np.asarray([r["inputs"]["features"] for r in train])
        y = np.asarray([label(r["answer"]) for r in train])
        exp, conf, nxt = split["round-1"]["explore"], split["round-1"]["adopt"], split["round-2"]["explore"]
        ids = train_ids + exp + conf + nxt
        if len(ids) != len(set(ids)):
            raise ValueError("训练、探索、确认和下一段资料重叠")
        def measurements(keys):
            return np.asarray([valid[i]["inputs"]["features"] for i in keys])
        def answers(keys):
            return np.asarray([label(valid[i]["answer"]) + 1 for i in keys])
        xe, xc = measurements(exp), measurements(conf)
        ye, yc = answers(exp), answers(conf)
        cols = [v["name"] for v in feature_schema()]
        table = pd.DataFrame(x, columns=cols)
        backtest = pd.DataFrame(xe, columns=cols)
        rules = conditions(x, xe, cols)
        method = published_method(HLLM.read_bytes(), frozen["hllm_sha256"], "H_LLM", "fit_many_policies",
                                  {"accuracy_score": accuracy_score})
        # 单次、单窗口的作者函数没有跨任务缓存；唯一替代的是已冻结的本地条件来源。
        source = SimpleNamespace(context=None, _solution_cache={}, _generate_cache_key=lambda *_: "round-1",
                                 get_queries=lambda *args, **kwargs: (rules, rules))
        fits = OfficialFit(Model, parent, x, y)
        author = method(source, fits, backtest, table, y, backtest, ye - 1)
        pools = {kind: [f"{kind}-{i}" for i in range(1, 4)] for kind in ("remove", "subset", "random", "matched")}
        if len(fits.runs) != 6:
            raise ValueError("作者的候选拟合未全部完成")
        for j, seed in enumerate(plan["seeds"]):
            selected = fits.runs[3+j][1]["train_indices"]
            excluded = fits.runs[j][1]["train_indices"]
            if len(selected) != 56 or set(selected) != set(range(len(y))) - set(excluded):
                raise ValueError("两个方向未得到成对清单")
            sample = sorted(random.Random(seed).sample(range(len(y)), len(selected)))
            fits.fit(table.iloc[sample], y[sample])
        for j, seed in enumerate(plan["seeds"]):
            selected = fits.runs[3+j][1]["train_indices"]
            counts = np.bincount(y[selected], minlength=len(base.meta["classes"]))
            rng = random.Random(seed)
            # 在完整训练池分层抽样，允许和原清单重合，保留同样的类别配比和样本总数。
            sample = sorted(i for c, n in enumerate(counts)
                            for i in rng.sample(np.flatnonzero(y == c).tolist(), int(n)))
            fits.fit(table.iloc[sample], y[sample])
        models, costs = {"static": base}, {}
        for name, (model, cost) in zip(sum(pools.values(), []), fits.runs, strict=True):
            models[name] = model
            counts = np.bincount(y[cost["train_indices"]], minlength=len(base.meta["classes"]))
            costs[name] = {**cost, "class_counts": dict(zip(base.meta["classes"], counts.tolist(), strict=True))}
        ep = {k: predicted(m, xe) for k, m in models.items()}
        co = {k: predicted(m, xc) for k, m in models.items()}
        for kind, field in (("remove", "scores_remove"), ("subset", "scores_retrain")):
            if not np.array_equal(author[field], [accuracy_score(ye, ep[n]) for n in pools[kind]]):
                raise ValueError("作者回测分与逐条预测不一致")
        decisions = {k: choose(names, ep, ye, co, yc, "macro_f1", True, True) for k, names in pools.items()}
        atomic_json(output/"decisions.json", {"conditions": rules, "costs": costs, "author_scores": author,
                    "explore": {k: score(ye, v) for k,v in ep.items()},
                    "confirm": {k: score(yc, v) for k,v in co.items()}, "decisions": decisions})

        regions = {str(b.batch_number): b.records for b in read_gas_batches(
            ROOT/"data/public/gas-drift/gas-drift.zip", expected_sha256=REGISTERED_SHA256) if 3 <= b.batch_number <= 8}
        features = np.concatenate([measurements(nxt)] + [np.asarray([r.values for r in rows]) for rows in regions.values()])
        keys = [("next", i) for i in nxt] + [(b, r.source_line) for b, rows in regions.items() for r in rows]
        preds = {k: predicted(m, features) for k,m in models.items()}
        old = list(map(json.loads, OLD.read_bytes().splitlines()))
        patch = list(map(json.loads, PATCH.read_bytes().splitlines()))
        for issued in (old, patch):
            if [(r["batch"], r["id"]) for r in issued] != keys:
                raise ValueError("历史参照和新预测的样本身份不同")
        for now, prev in (("static", "static"), *((f"remove-{i}", f"filter-{i}") for i in range(1, 4))):
            if not np.array_equal(preds[now], [r[prev] for r in old]):
                raise ValueError(f"原版或剔除条件未复现 {now}")
        preds["fixed-patch"] = np.asarray([r["served"] for r in patch])
        with (output/"issued.jsonl").open("x", encoding="utf8") as stream:
            for i, (batch, sid) in enumerate(keys):
                row = {"batch": batch, "id": sid, **{k: int(v[i]) for k,v in preds.items()}}
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    regions_y = {"next": answers(nxt), **{b: np.array([r.label for r in rows]) for b, rows in regions.items()}}
    stages, offset = {}, 0
    for stage, truth in regions_y.items():
        end = offset + len(truth)
        values = {k: v[offset:end] for k,v in preds.items()}
        common = (values["static"] != truth) & (values["fixed-patch"] != truth) & (values["remove-1"] != truth)
        stages[stage] = {k: {**repair(truth, p, values["static"]), "common_fixed": int(np.sum(common & (p == truth)))}
                         for k,p in values.items()}
        offset = end
    summary = {}
    for name in preds:
        rows = [stages[b][name] for b in regions]
        summary[name] = {"mean": float(np.mean([r["macro_f1"] for r in rows])), "worst": min(r["macro_f1"] for r in rows),
                         **{v: sum(r[v] for r in rows) for v in ("repaired", "broken", "common_fixed")}}
    upper = {b: max(stages[b][n]["macro_f1"] for n in ["fixed-patch", "static", *pools["subset"]]) for b in regions}
    atomic_json(output/"result.json", {"plan": sha(PLAN), "issued": sha(output/"issued.jsonl"),
                "decisions": sha(output/"decisions.json"), "stages": stages, "summary": summary,
                "best_by_batch_in_hindsight_fixed_patch_static_subsets": upper})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args().output)
