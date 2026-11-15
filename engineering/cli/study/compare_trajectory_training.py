"""复用公开维护入口检验同机配对。audit 只复算旧预测，run 才会产生新的工业权重。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from math import isclose, sqrt

import numpy as np

from check_prior_device_prediction import OUT as PRIOR
from compare_error_slice import run_external, transfer
from compare_proposal_frequency import (
    BASE, ROOT, TASKS, LocalStore, atomic_json, measure_streams, official, read, reference_folders,
    remaining_cycles, remaining_minutes,
)
from engineering.cli.workflow.public_maintenance import local_policy, run_task
from harness.models.trajectory_training import POLICIES, training_pairs

OUT = BASE/"trajectory-training"
COHORTS = ("bearing", "pronostia", "engine", "phm08")


def error_parts(pred, samples, truth):
    """总体 RMSE 的平方是两个分量的平方和，按各台设备的观测数加权。"""
    groups = defaultdict(list)
    for sid, value in pred.items():
        groups[samples[sid]["unit_id"]].append(value-truth[sid])
    count = len(pred)
    bias = sum(len(v)*float(np.mean(v))**2 for v in groups.values())/count
    within = sum(len(v)*float(np.var(v)) for v in groups.values())/count
    return {"rmse": sqrt(bias+within), "device_bias_rms": sqrt(bias), "within_device_rms": sqrt(within),
            "bias_fraction_mse": bias/(bias+within) if bias+within else 0.}


def audit(cohort):
    task = TASKS[cohort]
    package, checksum, batches = official(task)
    fit_ids = {r.sample_id for r in batches[0].samples if r.role == "fit"}
    rows = [r for r in package.train if r.sample_id in fit_ids]
    pairs, singles = training_pairs([(r.unit_id, r.time_end, r.label) for r in rows])
    if len(rows) != len(fit_ids) or len(fit_ids) != len(pairs)*2 + len(singles):
        raise ValueError("训练配对没有按官方划分覆盖每行一次")
    gaps = [rows[b].time_end-rows[a].time_end for a,b in pairs]
    fixed, _ = reference_folders(cohort, "mature")
    samples = read(fixed/"state.json")["samples"]
    truth = {r.sample_id: r.label for r in package.validation}
    if cohort == "phm08":
        truth = remaining_cycles()
    elif cohort == "pronostia":
        truth = remaining_minutes("full-test-manifest.json")
    previous = read(PRIOR/cohort/"scores.json")
    path = PRIOR/cohort/"issued.json"
    if sha256(path.read_bytes()).hexdigest() != previous["issued_sha256"]:
        raise ValueError("用于拆分的旧预测与原评分记录不同")
    parts = {}
    for arm, streams in read(path)["outputs"].items():
        parts[arm] = {}
        for name, preds in streams.items():
            parts[arm][name] = v = error_parts(preds, samples, truth)
            if not isclose(v["rmse"], previous["scores"][arm][name]["score"], rel_tol=1e-12):
                raise ValueError("误差分解未复现原始交接分数")
    atomic_json(OUT/"diagnostics"/(cohort+".json"), {
        "prior_issued_sha256": previous["issued_sha256"], "official_sha256": checksum, "fit_rows": len(rows),
        "fit_devices": len({r.unit_id for r in rows}), "pairs": len(pairs), "singletons": len(singles),
        "gap_min_median_max": np.quantile(gaps, [0,.5,1]).tolist(), "error_parts": parts})
    print(cohort, len(pairs), parts["project"]["parent"])


def run(cohort, policy):
    if cohort != TASKS[cohort]:
        run_external(cohort, policy=policy, destination=OUT/policy)
    else:
        folder = OUT/policy/cohort
        with LocalStore(folder/"study-lock").exclusive():
            run_task(cohort, ROOT/"data/official", folder, local_policy(folder, policy), 10, {"mode":"rule"})


def score(cohort):
    fixed, paper = reference_folders(cohort, "mature")
    task = TASKS[cohort]
    paths = {"fixed": fixed, "mature": paper, "old": BASE/"selected"/task if cohort == task else fixed.parent/"project"}
    if cohort != task:
        paths["simple"] = fixed.parent/"simple"
    paths.update({k: OUT/k/cohort for k in POLICIES})
    replies = {k: read(p/"driver-exchanges.json")["exchanges"] if k in ("mature", "simple") else []
               for k,p in paths.items()}
    table, _, controls = measure_streams(cohort, paths, replies)
    provenance = {}
    for policy, recipe in POLICIES.items():
        jobs = paths[policy]/"jobs"
        builds = []
        for f in sorted(jobs.glob("*/training.json")):
            report = read(f)
            request = read(f.with_name("input.json"))["request"]
            trace = report["trajectory"]
            if (trace["recipe"] != recipe or request["trajectory_sha256"] != trace["code_sha256"]
                    or report["train_rows"] != 2*trace["pairs"]+trace["singletons"]
                    or report["parent_checkpoint_sha256"] != "sha256:"+request["parent_sha256"]):
                raise ValueError("训练权重、算法、父版或训练量与请求不一致")
            builds.append({"pair_ids_sha256":trace["pair_ids_sha256"], "parent":request["parent_sha256"],
                           "update_steps":report["final_steps"]-report["initial_steps"], "fit_rows":report["train_rows"]})
        if not builds or len(builds) != table[policy]["physical_builds"]:
            raise ValueError("缺训练回执")
        provenance[policy] = builds
    if len({(r["pair_ids_sha256"],r["update_steps"],r["fit_rows"]) for runs in provenance.values() for r in runs}) != 1:
        raise ValueError("两种配方的配对身份或每遍成本不同")
    atomic_json(OUT/"scores"/(cohort+".json"), {"controls":controls, "arms":table, "builds":provenance})
    print(cohort, {k: v["score"] for k,v in table.items()})


def score_gas():
    fixed, paper = reference_folders("gas", "mature")
    paths = {"fixed":fixed, "old":BASE/"selected"/"gas", "mature":paper,
             "trajectory-difference":OUT/"trajectory-difference"/"gas"}
    replies = {arm:read(paper/"driver-exchanges.json")["exchanges"] if arm == "mature" else [] for arm in paths}
    table, streams, controls = measure_streams("gas", paths, replies)
    if streams["old"] != streams["trajectory-difference"]:
        raise ValueError("实验入口意外改变了气体交接")
    atomic_json(OUT/"scores"/"gas.json", {"controls":controls, "arms":table})
    print("gas", {arm:v["score"] for arm,v in table.items()})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("audit","run","score","transfer"))
    parser.add_argument("cohort", choices=(*COHORTS,"gas","all"))
    args = parser.parse_args()
    cohorts = (*COHORTS, "gas") if args.cohort == "all" and args.operation in ("run","score") else (
        COHORTS if args.cohort == "all" else (args.cohort,))
    for cohort in cohorts:
        if cohort == "gas":
            if args.operation == "score":
                score_gas()
            elif args.operation == "run":
                run("gas", "trajectory-difference")
            else:
                parser.error("气体共用交付流程，仅支持 run 和 score")
            continue
        if args.operation == "run":
            for policy in POLICIES:
                run(cohort, policy)
        elif args.operation == "transfer":
            if cohort not in ("phm08", "pronostia"):
                if args.cohort != "all":
                    parser.error("末版移用从两条外部流出发")
                continue
            transfer(cohort, additional={k:OUT/k/cohort for k in POLICIES}, destination=OUT)
        else:
            (audit if args.operation == "audit" else score)(cohort)
