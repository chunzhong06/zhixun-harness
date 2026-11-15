"""在已有开发样本逐项关闭修正规则，并沿冻结模型追查后续采集批的乙醛误判。"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BASELINE_CPU_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(ROOT / "engineering/src"))
import numpy as np
from harness.data.gas_input import read_gas_batches
from harness.data.official_gas import LABELS, read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.models.gas_views import pair_model, represent
from harness.runtime.storage import atomic_json


def load_json(path):
    return json.loads(path.read_bytes())


def frozen_models(config, package_path):
    sys.path.insert(0, str(package_path))
    from baselines.model import Model
    model_type = pair_model(Model)
    models = {}
    for name in ("base", "double"):
        arm = config["arms"][name]
        path = ROOT / arm["path"]
        if sha256(path.read_bytes()).hexdigest() != arm["sha256"]:
            raise ValueError("模型已偏离冻结版本")
        models[name] = model_type.load(path)
    path = ROOT / config["arms"]["double"]["path"]
    single = model_type.load(path)
    memory = single.meta["gas_pair_patch"]
    if memory["policy"] != "protected" or memory["method"] != "shape-blend" or memory["rules"] != [
        {"reported": 1, "target": 4}, {"reported": 1, "target": 2}
    ]:
        raise ValueError("消融需要冻结的乙醛、乙烯两个方向")
    # 乙烯单项完全沿用同一检查点，只在本次评价实例移除第一条规则。
    single.meta["gas_pair_patch"] = {k: v for k, v in memory.items() if k not in ("rules","policy")}
    single.meta["gas_pair_patch"].update(memory["rules"][1], version=1)
    models["ethylene"] = single
    # 两条规则有相同入口、不同出口。串行执行时，第一项改判的行不再符合第二项入口。
    serial = model_type.load(path)
    serial.meta["gas_pair_patch"]["policy"] = "cascade"
    models["serial"] = serial
    return models


def timed_predictions(models, inputs, repeats):
    names = ["ethylene", "double", "serial"]
    reference = {}
    durations = {name: [] for name in names}
    for name in names:
        reference[name] = np.concatenate([models[name].predict(x) for x in inputs])
    if not np.array_equal(reference["double"], reference["serial"]):
        raise ValueError("两种执行顺序的六类分数不一致")
    for rep in range(repeats):
        shift = rep % len(names)
        for name in names[shift:] + names[:shift]:
            start = perf_counter_ns()
            output = [models[name].predict(x) for x in inputs]
            durations[name].append((perf_counter_ns() - start) / len(inputs) / 1000)
            if not np.array_equal(np.concatenate(output), reference[name]):
                raise ValueError("计时轮输出不稳定")
    return {
        "samples": len(inputs), "per_pass_us": durations,
        "median_us": {k: float(np.median(v)) for k,v in durations.items()},
        "probability_sha256": {k: sha256(v.astype("<f8").tobytes()).hexdigest() for k,v in reference.items()}
    }


def development(rows, held, models):
    records = []
    for r in rows:
        x = np.asarray([r.values])
        pred = {k: int(model.predict(x)[0].argmax() + 1) for k,model in models.items()}
        if pred["serial"] != pred["double"] or (r.sample_id in held and pred["double"] != held[r.sample_id]):
            raise ValueError(f"历史逐条预测不一致 {r.sample_id}")
        records.append({"id": r.sample_id, "answer": r.label, "primary": r.sample_id in held, **pred})
    if {r["id"] for r in records if r["primary"]} != set(held):
        raise ValueError("开发子集身份缺失")
    summary = {}
    for name, primary in (("primary",True), ("remainder",False)):
        subset = [r for r in records if r["primary"] == primary]
        changes = Counter((r["answer"], r["ethylene"], r["double"]) for r in subset if r["ethylene"] != r["double"])
        summary[name] = {
            "rows": len(subset),
            "macro_f1": {k:macro_f1((r["answer"],r[k]) for r in subset) for k in ("base","ethylene","double")},
            "correct": {k:sum(r["answer"] == r[k] for r in subset) for k in ("base","ethylene","double")},
            "changes": [{"answer":a, "ethylene":b, "double":c, "count":n} for (a,b,c),n in sorted(changes.items())]}
    return records,summary


def trace_acetaldehyde(rows, models, expected):
    """按实测入口逐行展开五票及原版概率，并与真实执行和旧预测逐项核对。"""
    double = models["double"]
    bank = double._prediction_bank(double.meta["gas_pair_patch"])
    traces = []
    for sid, label, values in rows:
        if label != 4:
            continue
        x = np.asarray([values])
        base = models["base"].predict(x)[0]
        final = double.predict(x)[0]
        shape = represent(x, "shape", double.mean, double.scale)
        order = np.argsort(bank["train_norms"] - (2 * shape @ bank["train"].T)[0], kind="stable")
        neighbor = bank["labels"][order]
        votes = np.bincount(neighbor[:5], minlength=6)
        blend = (votes/5 + base)/2
        start, proposed, end = [int(p.argmax()+1) for p in (base,blend,final)]
        enters = start == 1
        applies = enters and proposed in (2,4)
        if not np.allclose(final, blend if applies else base, rtol=0, atol=1e-14):
            raise ValueError(f"逐步诊断与真实运行分数不符 {sid}")
        if sid in expected and (start, end) != expected[sid]:
            raise ValueError(f"诊断对不上历史预测 {sid}")
        traces.append({"id":sid, "entered":enters, "base":start, "knn":int(votes.argmax()+1),
                       "blend":proposed, "final":end, "base_scores":base.tolist(), "votes":votes.tolist(),
                       "nearest_target_rank":int(np.flatnonzero(neighbor==3)[0]+1)})
    slices = {"all_true_target": traces, "entered":[t for t in traces if t["entered"]],
              "remaining_as_ethanol":[t for t in traces if t["final"]==1]}
    summary = {}
    for name, members in slices.items():
        summary[name] = {"rows":len(members), "target_vote_counts":dict(sorted(Counter(t["votes"][3] for t in members).items())),
            "knn_winner":dict(Counter(t["knn"] for t in members)),
            "blended_winner":dict(Counter(t["blend"] for t in members)),
            "final_class":dict(Counter(t["final"] for t in members)),
            "target_tied_for_most_knn":sum(t["votes"][3] == max(t["votes"]) for t in members),
            "median_nearest_target_rank":float(np.median([t["nearest_target_rank"] for t in members])) if members else None,
            "zero_target_neighbors":sum(t["votes"][3]==0 for t in members)}
    return traces, summary


def verify_shared_execution(models, rows, archived):
    """旧、新执行在同一输入上复核所有分数；归档行还要求命中当时的预测。"""
    exact = 0
    batch_max = 0.0
    for start in range(0, len(rows), 64):
        block = rows[start:start+64]
        x = np.asarray([values for _,values in block])
        new = models["double"].predict(x)
        old = models["serial"].predict(x)
        if not np.array_equal(new.argmax(1),old.argmax(1)):
            raise ValueError("成组调用改变了预测类别")
        batch_max = max(batch_max, float(np.abs(new-old).max()))
        for sid, values in block:
            one = np.asarray([values])
            shared = models["double"].predict(one)
            serial = models["serial"].predict(one)
            if not np.array_equal(shared, serial):
                raise ValueError(f"逐条服务输出改变 {sid}")
            if sid in archived and int(shared.argmax()+1) != archived[sid]:
                raise ValueError(f"与归档预测不一致 {sid}")
            exact += 1
    if batch_max > 1e-12:
        raise ValueError("成组调用的分数差异超过浮点误差")
    return {"single_row_identical":exact, "archived_predictions_checked":len(archived),
            "bulk_max_score_difference":batch_max}


def run(protocol, folder, timing_only):
    config = load_json(protocol)
    frozen = load_json(ROOT / config["frozen_inputs"])
    path = ROOT / frozen["official"]
    if sha256(path.read_bytes()).hexdigest() != frozen["official_sha256"]:
        raise ValueError("官方数据和执行代码版本不同")
    package = read_official_gas(path)
    models = frozen_models(frozen, path)
    if tuple(models["double"].meta["classes"]) != LABELS:
        raise ValueError("类别映射与记录不符")
    held = load_json(ROOT / config["development_reference"])["holdout"]["predictions"]
    by_id = {r.sample_id:r for r in package.validation}
    inputs = [np.asarray([by_id[sid].values]) for sid in held]
    folder.mkdir(parents=True, exist_ok=True)
    timing = timed_predictions(models, inputs, config["timing_repeats"])
    if timing_only:
        first = load_json(folder / "initial-timing.json")
        if timing["probability_sha256"] != first["probability_sha256"]:
            raise ValueError("优化改变了开发主比较的概率输出")
        timing["output_checks"] = {"development":verify_shared_execution(models,
            [(r.sample_id,r.values) for r in package.validation], held)}
        previous = [json.loads(line) for line in (ROOT/config["prior_future_predictions"]).read_bytes().splitlines()]
        for n in config["diagnosis_batches"]:
            batch, = read_gas_batches(ROOT/frozen["source"],expected_sha256=frozen["source_sha256"],batch_number=n)
            archived = {r["line"]:r["double"] for r in previous if r["batch"]==n}
            timing["output_checks"][str(n)] = verify_shared_execution(models,
                [(r.source_line,r.values) for r in batch.records], archived)
        atomic_json(folder/"shared-timing.json", timing)
        return
    records, dev_summary = development(package.validation, held, models)
    atomic_json(folder/"development.json", records)
    atomic_json(folder/"initial-timing.json", timing)
    historical = [json.loads(line) for line in (ROOT/config["prior_future_predictions"]).read_bytes().splitlines()]
    subsets = {"development":[(r.sample_id,r.label,r.values) for r in package.validation]}
    expectations = {"development":{r["id"]:(r["base"],r["double"]) for r in records}}
    for n in config["diagnosis_batches"]:
        batch, = read_gas_batches(ROOT/frozen["source"],expected_sha256=frozen["source_sha256"],batch_number=n)
        name = str(n)
        subsets[name] = [(r.source_line,r.label,r.values) for r in batch.records]
        expectations[name] = {r["line"]:(r["base"],r["double"]) for r in historical if r["batch"]==n}
        if set(expectations[name]) != {r.source_line for r in batch.records}:
            raise ValueError("诊断批与已发布预测的来源行不一致")
    diagnoses = {}
    details = {}
    for name, subset in subsets.items():
        details[name], diagnoses[name] = trace_acetaldehyde(subset,models,expectations[name])
    atomic_json(folder/"diagnostic-rows.json", details)
    atomic_json(folder/"result.json", {"development":dev_summary, "diagnosis":diagnoses,
                "training_class_counts":dict(Counter(r.label for r in package.train))})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-only", action="store_true", help="要求与原顺序实测概率摘要完全一致")
    args = parser.parse_args()
    run(args.protocol, args.output, args.timing_only)
