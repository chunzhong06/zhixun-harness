"""在相同父权重上分别更换选样、候选排序和换版规则。所有外部答案最后读取计分。"""
import argparse
import ast
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from check_gas_patching_start import ROOT, PACKAGE, SOURCE, compile_bridge, dependencies, export_weights, sha
from compare_gas_patching import author_predict, SPLIT
from harness.data.gas_input import feature_schema, read_gas_batches, REGISTERED_SHA256
from harness.evaluation.metrics import macro_f1
from harness.models.gas_views import pair_model, build_pair_memory
from harness.runtime.storage import atomic_json

PROTOCOL = ROOT / "engineering/configs/experiments/gas-paper-module-smoke.json"
OLD_ISSUED = ROOT / "results/experiments/official-gas/patching-comparison/run-36/issued.jsonl"
HLLM = next((ROOT / "references/implementations/pauliusrauba/Self_Healing_ML").rglob("HLLM.py"))
MODYN = next((ROOT / "references/implementations/eth-easl/modyn").rglob("upstream-source.zip"))


def published_method(source, expected, cls, name, globals_):
    """从固定源码加载指定方法体，避免启动作者的 API、数据库和训练服务。"""
    if sha256(source).hexdigest() != expected:
        raise ValueError("作者源码身份变化")
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = dict(globals_)
    exec(compile(ast.Module(body=[method], type_ignores=[]), f"{cls}.{name}", "exec"), scope)
    return scope[name]


def predicted(model, x):
    return np.concatenate([model.predict(x[start:start+128]).argmax(axis=1)+1 for start in range(0, len(x), 128)])


def score(y, p):
    return macro_f1(zip(y.tolist(), p.tolist(), strict=True))


def repair(y, p, original):
    return {"macro_f1": score(y, p), "repaired": int(np.sum((original != y) & (p == y))),
            "broken": int(np.sum((original == y) & (p != y)))}


def conditions(x, current, columns):
    """首项条件已在上次只读核验中冻结。其他条件仅看当时可见的两份输入，不读测试批。"""
    k = 56
    rules = [(0, ">", float(current[:, 0].max()))]
    iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    delta = np.median(current, axis=0) - np.median(x, axis=0)
    offset = np.divide(delta, iqr, out=np.zeros_like(delta), where=iqr > 0)
    used = {0}
    for j in sorted(range(len(columns)), key=lambda j: (-abs(offset[j]), j)):
        if j//8 in used or offset[j] == 0:
            continue
        ordered = np.sort(x[:, j])
        tail = k if offset[j] > 0 else len(ordered)-k
        if ordered[tail-1] == ordered[tail]:
            continue
        cut = float((ordered[tail-1]+ordered[tail])/2)
        rules.append((j, "<" if offset[j] > 0 else ">", cut))
        used.add(j//8)
        if len(rules) == 3:
            break
    result = [f"`{columns[j]}` {op} {cut!r}" for j, op, cut in rules]
    if len(result) != 3:
        raise ValueError("冻结生成器未能形成三个等量条件")
    return result


class OfficialFit:
    """每次从同一个官方检查点恢复 Adam。用训练行身份核对两种选样的预算。"""
    def __init__(self, model_type, parent, x, y):
        self.model_type, self.parent, self.x, self.y = model_type, parent, x, y
        self.runs = []

    def fit(self, table, labels):
        ids = np.asarray(table.index, dtype=int)
        if (len(np.unique(ids)) != len(ids) or not np.all(ids == np.sort(ids))
                or not np.array_equal(self.y[ids], labels)):
            raise ValueError("训练行或目标与官方清单不符")
        start = perf_counter()
        model = self.model_type.load(self.parent)
        previous = model.meta["steps"]
        model.fit(self.x[ids], self.y[ids], 1)
        self.model = model
        self.runs.append((model, {"train_indices": ids.tolist(), "steps": model.meta["steps"]-previous,
                                  "fit_wall_s": perf_counter()-start}))
        return self

    def predict(self, table):
        return predicted(self.model, table.to_numpy())-1


def choose(pool, exp_preds, y_exp, confirm_preds, y_confirm, metric, include_parent, confirm):
    order = (["static"] if include_parent else []) + list(pool)
    measure = accuracy_score if metric == "accuracy" else score
    nominee = max(order, key=lambda k: measure(y_exp, exp_preds[k]))
    selected = nominee
    if confirm and score(y_confirm, confirm_preds[nominee]) <= score(y_confirm, confirm_preds["static"])+1e-12:
        selected = "static"
    return {"nominee": nominee, "served": selected}


def smoke(folder):
    cfg = json.loads(PROTOCOL.read_bytes())
    if folder.exists() or any(sha(p) != cfg[k] for p,k in [(PACKAGE,"package_sha256"),
                 (SPLIT,"split_sha256"),(OLD_ISSUED,"prior_issued_sha256")]):
        raise ValueError("输出已存在或共同输入发生变化")
    split = json.loads(SPLIT.read_bytes())["rounds"]
    columns = [row["name"] for row in feature_schema()]
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model
    with TemporaryDirectory(prefix="gas-module-smoke-") as tmp, ZipFile(PACKAGE) as archive:
        work = Path(tmp)
        parent = work/"checkpoint.npz"
        parent.write_bytes(archive.read("checkpoint.npz"))
        if sha(parent) != cfg["parent_sha256"]:
            raise ValueError("父权重不符")
        base = Model.load(parent)
        train = [json.loads(line) for line in archive.read("public/train.jsonl").splitlines()]
        valid = {r["id"]:r for r in map(json.loads, archive.read("public/validation.jsonl").splitlines())}
        x = np.asarray([r["inputs"]["features"] for r in train])
        label = base.meta["classes"].index
        y = np.asarray([label(r["answer"]) for r in train])
        exp, conf, nxt = split["round-1"]["explore"], split["round-1"]["adopt"], split["round-2"]["explore"]
        train_ids = [r["id"] for r in train]
        if len(set(train_ids + exp + conf + nxt)) != len(train_ids)+len(exp)+len(conf)+len(nxt):
            raise ValueError("不同用途的样本身份重叠")
        def inputs(ids):
            return np.asarray([valid[s]["inputs"]["features"] for s in ids])
        def answers(ids):
            return np.asarray([label(valid[s]["answer"])+1 for s in ids])
        xe, xc = inputs(exp), inputs(conf)
        ye, yc = answers(exp), answers(conf)
        table, backtest = pd.DataFrame(x, columns=columns), pd.DataFrame(xe, columns=columns)
        models, costs = {"static":base}, {}
        original_explore = predicted(base, xe)
        timed = perf_counter()
        queries = conditions(x, xe, columns)
        query_cost = perf_counter()-timed
        method = published_method(HLLM.read_bytes(), cfg["hllm_sha256"], "H_LLM", "fit_model",
                                  {"accuracy_score":accuracy_score, "np":np})
        source = SimpleNamespace(information={}, datasets={}, get_queries=lambda *_:queries)
        estimator = OfficialFit(Model, parent, x, y)
        timed = perf_counter()
        method(source, estimator, backtest, table, y, backtest, ye-1)
        hllm_wall = perf_counter()-timed
        if len(estimator.runs) != 4 or any(len(r["train_indices"]) != 389 or r["steps"] != 4 for _,r in estimator.runs):
            raise ValueError("作者回测未完成相同预算的三候选及胜者重拟合")
        filters = ["filter-"+str(i+1) for i in range(3)]
        for name,(model, receipt) in zip(filters, estimator.runs[:3], strict=True):
            models[name], costs[name] = model, receipt
        refit, repeat = estimator.runs[-1]
        chosen_filter = next(name for name in filters if costs[name]["train_indices"] == repeat["train_indices"])
        if len({tuple(costs[name]["train_indices"]) for name in filters}) != 3:
            raise ValueError("筛选条件产生了重复候选")
        with ZipFile(MODYN) as modyn:
            rs_source = modyn.read(next(n for n in modyn.namelist() if n.endswith("/remote_rs2_downsampling.py")))
        draws = []
        for seed in cfg["seeds"]:
            step = published_method(rs_source, cfg["modyn_rs2_source_sha256"], "RemoteRS2Downsampling",
                                    "_epoch_step_no_r", {"random":random.Random(seed)})
            sampler = SimpleNamespace(_current_subset=-1, _subsets=[], _all_sample_ids=list(range(len(y))))
            timed = perf_counter()
            step(sampler, 389)
            ids = sorted(sampler._subsets[sampler._current_subset])
            selection_s = perf_counter()-timed
            estimator.fit(table.iloc[ids], y[ids])
            name = f"rs2-{seed}"
            draws.append(name)
            models[name], costs[name] = estimator.runs[-1]
            costs[name]["selection_wall_s"] = selection_s
        estimator.fit(table, y)
        models["full"],costs["full"] = estimator.runs[-1]
        wrong = Counter((int(a),int(b)) for a,b in zip(ye,original_explore,strict=True) if a != b)
        (target, reported), count = min(wrong.items(), key=lambda item:(-item[1], item[0]))
        if count < 2:
            raise ValueError("现行规则在本轮不会试改")
        patches = ["prior","raw-knn","shape-blend"]
        for name in patches:
            timed = perf_counter()
            model = pair_model(Model).load(parent)
            spec = {"method":name,"reported":reported,"target":target}
            model.meta["gas_pair_patch"] = build_pair_memory(spec, x, y, model.mean, model.scale)
            models[name] = model
            costs[name] = {"training_rows":len(y),"steps":0,"fit_wall_s":perf_counter()-timed}
        regions = {b.batch_number:b.records for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",
                    expected_sha256=REGISTERED_SHA256) if b.batch_number in range(3,9)}
        all_x = np.concatenate([xe, xc, inputs(nxt)] + [np.asarray([r.values for r in rs]) for rs in regions.values()])
        weights = work/"weights.bin"
        export_weights(parent, weights)
        cp = compile_bridge(work, dependencies(SOURCE.parent.parent/"runtime-dependencies"))
        author, java_times = author_predict(work, cp, weights, x, y, all_x)
        predictions = {"patching":author}
        for name, model in models.items():
            timed = perf_counter()
            predictions[name] = predicted(model, all_x)
            costs.setdefault(name,{})["predict_wall_s"] = perf_counter()-timed
        if not np.array_equal(predicted(refit, all_x), predictions[chosen_filter]):
            raise ValueError("作者胜者重拟合没有重现逐条预测")
        ne, nc, nn = len(exp), len(conf), len(nxt)
        ep = {k:v[:ne] for k,v in predictions.items()}
        co = {k:v[ne:ne+nc] for k,v in predictions.items()}
        pools = {"filters":filters,"rs2":draws,"local_tools":patches,"old_rule":["shape-blend"],
                 "full":["full"],"external_patch":["patching"]}
        decisions = {pool:{f"{metric}/parent{int(inc)}/confirm{int(gate)}":
                           choose(names, ep, ye, co, yc, metric, inc, gate)
                           for metric in ("accuracy","macro_f1") for inc in (False,True) for gate in (False,True)}
                     for pool,names in pools.items()}
        if decisions["filters"]["accuracy/parent0/confirm0"]["served"] != chosen_filter:
            raise ValueError("复算提名与作者实际回测不一致")
        folder.mkdir(parents=True)
        atomic_json(folder/"decisions.json", decisions)
        future_ids = [("next",sid) for sid in nxt] + [(str(b),r.source_line) for b,rs in regions.items() for r in rs]
        issued = folder/"issued.jsonl"
        with issued.open("x",encoding="utf8") as stream:
            for i,(batch,sid) in enumerate(future_ids,ne+nc):
                record = {"batch":batch,"id":sid,**{k:int(v[i]) for k,v in predictions.items()}}
                stream.write(json.dumps(record,separators=(",",":"))+"\n")
        # 此后才给已发出的预测打分。既有双修正模型仅用于六批共同错例归因。
        prior = {(r["batch"],r["line"]):r for r in map(json.loads,OLD_ISSUED.read_bytes().splitlines())}
        expected_keys = {(b,r.source_line) for b,rs in regions.items() for r in rs}
        if expected_keys != set(prior):
            raise ValueError("历史对照批次或行身份不符")
        ys = {"next":answers(nxt), **{str(b):np.asarray([r.label for r in rs]) for b,rs in regions.items()}}
        metrics, offset = {}, ne+nc
        for stage, truth in ys.items():
            end = offset+len(truth)
            values = {k:v[offset:end] for k,v in predictions.items()}
            shared = None
            if stage != "next":
                previous = [prior[int(stage), r.source_line] for r in regions[int(stage)]]
                if any(not np.array_equal(values[k], [r[old] for r in previous])
                       for k,old in (("static","static"),("patching","author"))):
                    raise ValueError("原版或作者原始预测未能复现")
                values["legacy_double"] = np.asarray([r["project"] for r in previous])
                shared = (values["static"] != truth) & (values["legacy_double"] != truth)
            metrics[stage] = {k:{**repair(truth, v, values["static"]),
                                 "shared_fixed": int(np.sum(shared & (v == truth))) if shared is not None else None}
                              for k,v in values.items()}
            offset = end
        summary = {}
        for k in metrics["3"]:
            rows = [metrics[str(b)][k] for b in regions]
            summary[k] = {"mean":float(np.mean([r["macro_f1"] for r in rows])),
                          "worst":min(r["macro_f1"] for r in rows),
                          **{m:sum(r[m] for r in rows) for m in ("repaired","broken","shared_fixed")}}
        report = {"protocol_sha256":sha(PROTOCOL), "issued_sha256":sha(issued), "decision_sha256":sha(folder/"decisions.json"),
                  "support":{"explore":ne,"confirm":nc,"next":nn,"future":sum(map(len,regions.values()))},
                  "queries":queries, "author_backtest":source.information, "author_refit":chosen_filter,
                  "author_refit_wall_s":repeat["fit_wall_s"], "author_backtest_fit_refit_wall_s":hllm_wall,
                  "local_condition_selection_wall_s":query_cost, "java_cost_ns":java_times, "costs":costs,
                  "pools":pools, "target":target, "reported":reported, "explore":{
                      k:{"accuracy":float(accuracy_score(ye,p)), "macro_f1":score(ye,p)} for k,p in ep.items()},
                  "confirmation":{k:score(yc,p) for k,p in co.items()}, "stage_metrics":metrics,"summary":summary}
        atomic_json(folder/"result.json", report)
        print(json.dumps({"published_winner":chosen_filter,"summary":summary},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    smoke(parser.parse_args().output)
