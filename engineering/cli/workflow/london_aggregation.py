"""在已核验的完整首案预测流上接入成熟在线组合，并比较强固定组合。"""
from argparse import ArgumentParser
import json
from pathlib import Path

import joblib
import numpy as np

from engineering.cli.workflow import london_field_driver as scene
from harness.models.online_aggregation import aggregate
from harness.models.sparse_reference import correct
from harness.models.station_residual import Rule, error
from harness.runtime.storage import LocalStore, atomic_json


def read(path): return json.loads(path.read_bytes())


def inputs(pool, site, tools, truth):
    """由已冻结基础预测重建21天热身和服务流；校正只消费已遮蔽的反馈。"""
    meta = read(pool / site / "deployment.json")
    y, _, _, service, returned = scene.roles(truth)
    clock = np.arange(service[0] - 21 * 24, len(y))
    known = np.r_[y[clock[clock < service[0]]], returned]
    parent, _ = joblib.load(pool / site / "initial.joblib")
    bases = {"initial": tools.issue(parent, clock)}
    for entry in meta["published"]:
        with np.load(pool / site / entry["source"] / "base.npz") as z:
            if not np.array_equal(z["ids"], clock): raise ValueError("基础预测时钟不同")
            bases[entry["source"]] = z["prediction"].copy()
    columns = np.column_stack([correct(bases[o["source"]][:, None], known, clock, tuple(o["feedback"]))[:, 0]
                               for o in meta["options"]])
    use = clock >= service[0]
    with np.load(pool / site / "issued.npz") as z:
        for i, option in enumerate(meta["options"]):
            if not np.allclose(z["bank-" + option["name"]], columns[use, i], rtol=0, atol=1e-10):
                raise ValueError("热身扩展改变了原服务期专家预测")
    return clock, known, columns, use, meta


def pair_candidates(count):
    """全局固定单项和两项组合；名单先固定，终局评分只用于给出最强对照。"""
    return [(a, a, 1.) for a in range(count)] + [(a, b, w) for a in range(count)
        for b in range(a + 1, count) for w in (.25, .5, .75)]


def main(argv):
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True); ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args(argv)
    original = read(args.pool / "scores.json")
    if not read(args.pool / "audit.json")["all_controls_and_scores_verified"]:
        raise ValueError("原首案流尚未核验")
    scene.configure(original["start"]); tools = scene.prepare(args.archive, True, True)
    tools.static = Rule(**original["initial_spec"]["recipe"])
    with LocalStore(args.output).exclusive():
        if (args.output / "scores.json").exists(): raise ValueError("不可覆盖实测")
        scores = {}; covariances = {}; pair_scores = []; histories = {}; identities = None
        for site in original["counts"]:
            dest = args.output / site; dest.mkdir()
            truth = scene.readings(args.archive, site).to_numpy()
            clock, known, columns, use, meta = inputs(args.pool, site, tools, truth)
            names = [o["name"] for o in meta["options"]] + ["previous-best-fixed"]
            if identities is not None and names != identities: raise ValueError("跨站候选身份不同")
            identities = names
            split = int(np.flatnonzero(use)[0]); issued = {}; saved = {}
            for mode in ("convex", "selection"):
                value, weight = aggregate(columns, known, mode)
                frozen, frozen_weight = aggregate(columns, known, mode, split)
                issued["boa-" + mode] = value[use]; issued["frozen-" + mode] = frozen[use]
                saved[mode + "_weights"] = weight[use]; saved[mode + "_opening"] = frozen_weight[split]
            issued["uniform"] = columns[use].mean(axis=1)
            with np.load(args.pool / site / "issued.npz") as z:
                for key in ("pool-online", "pool-history-direct", "pool-history-gate", "current-code",
                            "previous-project", "paper", "previous-best-fixed", "numeric", "static"):
                    issued[key] = z[key].copy()
            # 全时段答案仅在发放全部完成后进入评分。固定组合不能反向进入BOA状态。
            valid = np.isfinite(truth[clock[use]]); target = truth[clock[use]][valid]
            controls = np.column_stack((columns[use], issued["previous-best-fixed"]))
            residual = controls[valid] - target[:, None]
            covariance = residual.T @ residual / len(residual); covariances[site] = covariance
            pairs = pair_candidates(len(names))
            pair_scores.append([float(np.sqrt(max(0., w*w*covariance[a,a] + (1-w)**2*covariance[b,b]
                                + 2*w*(1-w)*covariance[a,b]))) for a,b,w in pairs])
            for key, prediction in issued.items(): scores.setdefault(key, {})[site] = error(target, prediction[valid])
            histories[site] = {"history_answers": int(np.isfinite(known[:split]).sum()),
                               "service_answers": int(np.isfinite(known[split:]).sum()), "scored": int(valid.sum())}
            np.savez_compressed(dest / "issued.npz", ids=clock[use], returned=known[use], experts=columns[use],
                                fixed_controls=controls, **issued, **saved)
            np.savez_compressed(dest / "warmup.npz", ids=clock[:split], returned=known[:split], experts=columns[:split])
            print(site + " 在线组合完整交付完成", flush=True)
        pair_scores = np.asarray(pair_scores); means = pair_scores.mean(axis=0); best = int(np.argmin(means))
        a,b,w = pairs[best]
        pair_table = [{"left": a, "right": b, "weight": w, "macro_rmse": float(means[i]),
                       "station_rmse": dict(zip(original["counts"], pair_scores[:,i].tolist(), strict=True))}
                      for i,(a,b,w) in enumerate(pairs)]
        atomic_json(args.output / "fixed-combinations.json", {"identities": identities, "candidates": pair_table,
                    "best_index": best, "scope": "事前固定组合名单；终局只选出最强对照，不回传给现场"})
        scores["best-fixed-combination"] = pair_table[best]["station_rmse"]
        for site in original["counts"]:
            with np.load(args.output / site / "issued.npz") as z:
                prediction = w*z["fixed_controls"][:,a] + (1-w)*z["fixed_controls"][:,b]
                np.savez_compressed(args.output / site / "best-fixed.npz", ids=z["ids"], prediction=prediction)
        result = {"scope": "已见开发，原首案已训练预测流上的成熟BOA适配；没有新训练或Luna调用",
                  "pool": str(args.pool), "start": original["start"], "initial_spec": original["initial_spec"],
                  "counts": histories, "primary": "boa-convex", "macro_rmse": {k:float(np.mean(list(v.values()))) for k,v in scores.items()},
                  "station_rmse": scores, "options": len(identities)-1, "fixed_candidates": len(pairs),
                  "best_fixed": pair_table[best], "paper": "doi:10.1007/s10994-016-5592-6，图2和式(2)",
                  "adaptation": "均匀先验，c=20；只在到期答案后更新；先用21日合法历史热身，无答案小时保持组合权重。"}
        atomic_json(args.output / "scores.json", result)
        print(json.dumps({"完整交付评分": result["macro_rmse"], "新增训练": 0, "新增现场调用": 0}, ensure_ascii=False))
