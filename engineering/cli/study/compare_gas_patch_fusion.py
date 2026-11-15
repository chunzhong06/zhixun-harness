"""在同一权重上逐项加入两条气体修正，区分选样和修正的作用。"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from compare_gas_paper_modules import (
    ROOT, PACKAGE, SPLIT, OLD_ISSUED, REGISTERED_SHA256, atomic_json, build_pair_memory,
    choose, np, pair_model, predicted, read_gas_batches, repair, score, sha,
)
from harness.models.gas_views import append_pair

PROTOCOL = ROOT / "engineering/configs/experiments/gas-patch-factorial-smoke.json"
PREVIOUS = ROOT / "results/experiments/official-gas/module-smoke/measured"


def compare(out):
    cfg = json.loads(PROTOCOL.read_bytes())
    prev = json.loads((PREVIOUS / "result.json").read_bytes())
    if out.exists() or sha(PREVIOUS / "issued.jsonl") != prev["issued_sha256"]:
        raise ValueError("既有输出或前轮预测身份不符")
    earlier = [json.loads(line) for line in (PREVIOUS / "issued.jsonl").read_bytes().splitlines()]
    historical = {(str(r["batch"]),r["line"]):r["project"] for r in
                  map(json.loads, OLD_ISSUED.read_bytes().splitlines())}
    rounds = json.loads(SPLIT.read_bytes())["rounds"]
    explore, confirm = rounds["round-1"]["explore"], rounds["round-1"]["adopt"]
    following = rounds["round-2"]["explore"]
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model

    with TemporaryDirectory(prefix="gas-patch-fusion-") as tmp, ZipFile(PACKAGE) as archive:
        parent = Path(tmp) / "parent.npz"
        parent.write_bytes(archive.read("checkpoint.npz"))
        train = list(map(json.loads, archive.read("public/train.jsonl").splitlines()))
        valid = {r["id"]:r for r in map(json.loads, archive.read("public/validation.jsonl").splitlines())}
        index = Model.load(parent).meta["classes"].index
        x = np.array([r["inputs"]["features"] for r in train])
        y = np.array([index(r["answer"]) for r in train])
        batches = {str(b.batch_number):b.records for b in read_gas_batches(
            ROOT / "data/public/gas-drift/gas-drift.zip", expected_sha256=REGISTERED_SHA256)
            if 3 <= b.batch_number <= 8}
        windows = [explore, confirm, following]
        features = np.concatenate([np.array([valid[k]["inputs"]["features"] for k in ids]) for ids in windows]
                                  + [np.array([r.values for r in rows]) for rows in batches.values()])
        def truth(ids):
            return np.array([index(valid[k]["answer"])+1 for k in ids])
        ne, off = len(explore), len(explore)+len(confirm)
        predictions, costs, pools = {}, {}, {}
        for base in cfg["bases"]:
            model = pair_model(Model).load(parent)
            started = perf_counter()
            steps = model.meta["steps"]
            if base != "static":
                ids = prev["costs"][base]["train_indices"]
                model.fit(x[ids], y[ids], 1)
                if model.meta["steps"] - steps != prev["costs"][base]["steps"]:
                    raise ValueError("续训的步数与冻结对照不符")
            costs[base] = {"fit_seconds":perf_counter()-started, "steps":model.meta["steps"]-steps,
                           "training_rows":len(ids) if base != "static" else 0}
            pools[base] = [base,base+"+one",base+"+two"]
            for n, key in enumerate(pools[base]):
                started = perf_counter()
                if n:
                    spec = {"method":"shape-blend", **cfg["patches"][n-1]}
                    if n == 1:
                        model.meta["gas_pair_patch"] = build_pair_memory(spec, x, y, model.mean, model.scale)
                    else:
                        model.meta["gas_pair_patch"] = append_pair(model.meta["gas_pair_patch"], spec, "protected")
                built = perf_counter()
                predictions[key] = predicted(model, features)
                costs[base][key] = {"add_patch_s":built-started, "predict_s":perf_counter()-built}

        decisions = {b:choose(pool, {k:p[:ne] for k,p in predictions.items()},truth(explore),
                              {k:p[ne:off] for k,p in predictions.items()},truth(confirm),
                              "macro_f1", True, True) for b,pool in pools.items()}
        keys = [("next",k) for k in following] + [(b,r.source_line) for b,rows in batches.items() for r in rows]
        if [(r["batch"],r["id"]) for r in earlier] != keys:
            raise ValueError("前后两次预测的样本身份或顺序变化")
        for key, old in [(b,b) for b in pools]+[("static+one","shape-blend")]:
            if not np.array_equal(predictions[key][off:], [r[old] for r in earlier]):
                raise ValueError(f"{key} 未复现前次预测")
        if not np.array_equal(predictions["static+two"][off+len(following):],
                              [historical[k] for k in keys if k[0] != "next"]):
            raise ValueError("两项修正的共同父版未复现历史逐条预测")
        out.mkdir(parents=True)
        atomic_json(out/"decisions.json", decisions)
        with (out/"issued.jsonl").open("x",encoding="utf8") as stream:
            for i,(batch,k) in enumerate(keys,off):
                row = {"batch":batch,"id":k,**{name:int(p[i]) for name,p in predictions.items()}}
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
        # 下一段和外部批的答案在预测发出后进入独立的评价部分。
        truths = {"next":truth(following), **{b:np.array([r.label for r in rows]) for b,rows in batches.items()}}
        metrics, changes = {}, {}
        for part, values in truths.items():
            local = {k:p[off:off+len(values)] for k,p in predictions.items()}
            metrics[part] = {k:repair(values,p,local["static"]) for k,p in local.items()}
            changes[part] = {v:repair(values,local[v],local[u]) for group in pools.values()
                             for u,v in zip(group,group[1:])}
            off += len(values)
        summary = {}
        for name in predictions:
            rows = [metrics[b][name] for b in batches]
            summary[name] = {"mean":float(np.mean([r["macro_f1"] for r in rows])),
                             "worst":min(r["macro_f1"] for r in rows),
                             **{k:sum(r[k] for r in rows) for k in ("repaired","broken")}}
        atomic_json(out/"result.json", {"protocol_sha256":sha(PROTOCOL), "issued_sha256":sha(out/"issued.jsonl"),
                    "decisions_sha256":sha(out/"decisions.json"), "parent_sha256":sha(parent),
                    "previous_result_sha256":sha(PREVIOUS/"result.json"), "summary":summary,
                    "stage_metrics":metrics,"incremental_changes":changes,"costs":costs,"pools":pools})
        print(json.dumps({"decisions":decisions,"summary":summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    compare(parser.parse_args().output)
