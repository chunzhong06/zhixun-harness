"""检验等量训练中定向、保留背景及全池抽样的影响；独立运行后按实际交接计分。"""
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_error_slice import (BASE, DEST as PRIOR, ROOT, TASKS, LocalStore, ModelArtifacts,
                                 atomic_json, measure_streams, read, reference_folders, run_external, transfer)
from engineering.cli.workflow.public_maintenance import local_policy, run_task
from compare_proposal_frequency import metric, official, remaining_cycles, remaining_minutes
from engine_endpoint_check import PACKAGES
from harness.models.official_rul_cpu import _trainer

OUT = BASE / "training-coverage"
ARMS = ("targeted", "mixed", "uniform")
RUL = ("phm08", "pronostia", "engine", "bearing")


def execute(arm):
    policy = "coverage-" + arm
    for cohort in (*RUL, "gas") if arm == "mixed" else RUL:
        if cohort != TASKS[cohort]:
            run_external(cohort, policy=policy, destination=OUT/arm)
        else:
            folder = OUT/arm/cohort
            with LocalStore(folder/"study-lock").exclusive():
                run_task(cohort, ROOT/"data/official", folder, local_policy(folder, policy), 10, {"mode":"rule"})
        print(arm, cohort, "交接完成", flush=True)


def training_receipts(path, task):
    """连落选的权重一起核对。首轮必须同父，后续保留各组自己的服务历史。"""
    state = read(path/"state.json")
    store = ModelArtifacts(path/"models")
    pool = {key:row for key,row in state["samples"].items() if row["role"] == "fit"}
    n = 992 if task == "engine" else 18
    if len(pool) != 4*n:
        raise ValueError("训练池变化，等量条件失效")
    result = []
    for r in read(path/"run.json")["rounds"]:
        trial, = r["trials"]
        model = state["models"][trial["model"]]
        parent = state["models"][r["current"]]
        parent_hash = store.read(parent["artifact"])[0]["sha256"]
        request = store.read(model["artifact"])[0]["request"]
        ids = request["fit_ids"]
        if (len(ids) != n or len(set(ids)) != n or not set(ids) <= pool.keys()
                or request["epochs"] != 1 or request["parent_sha256"] != parent_hash):
            raise ValueError("实际父权重、样本数量、训练来源或遍数不符")
        result.append({"round":r["batch"],"parent":parent_hash,"manifest":request["fit_manifest"],
                       "training_rows":n,"training_groups":sorted({pool[i]["group"] for i in ids}),
                       "candidate":model["artifact"],"exploration_rmse":trial["exploration_score"],
                       "confirmation":r["confirmation"],"outcome":r["outcome"]})
    return result


def first_pair(cohort, paths):
    """对同父模型的首个候选逐一预测，包括被确认拒绝的候选；另核对真实服务输出。"""
    task = TASKS[cohort]
    if cohort == "phm08":
        answers = remaining_cycles()
    elif cohort == "pronostia":
        answers = remaining_minutes("full-test-manifest.json")
    else:
        package, _, _ = official(task)
        answers = {s.sample_id:s.label for s in package.validation}
    archive = ROOT/"data/official"/task/PACKAGES[task][0]
    output = {}
    with TemporaryDirectory(prefix="coverage-pair-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"C" if task=="engine" else "B") as trainer:
        for arm in ARMS:
            folder = paths[arm]
            state = read(folder/"state.json")
            r = read(folder/"run.json")["rounds"][0]
            issued = read(folder/r["submitted"])["payload"]["predictions"]
            ids = sorted(issued)
            inputs = np.asarray([state["samples"][i]["values"] for i in ids])
            predictions = {}
            store = ModelArtifacts(folder/"models")
            for key in (r["current"],r["trials"][0]["model"]):
                _, blob = store.read(state["models"][key]["artifact"])
                checkpoint = Path(tmp)/"model.npz"
                checkpoint.write_bytes(blob)
                values = trainer.Model.load(checkpoint).predict(inputs)
                if values.shape != (len(ids),) or not np.isfinite(values).all():
                    raise ValueError("首轮候选预测无效")
                predictions[key] = dict(zip(ids,map(float,values),strict=True))
            if not all(abs(predictions[r["selected"]][i]-issued[i])<1e-8 for i in ids):
                raise ValueError("独立预测未复现该组首轮实际交接")
            output[arm] = {"parent":metric(task,answers,predictions[r["current"]]),
                           "candidate":metric(task,answers,predictions[r["trials"][0]["model"]]),
                           "served":metric(task,answers,issued),"n":len(ids)}
    return output


def score(cohort):
    task = TASKS[cohort]
    fixed, paper = reference_folders(cohort, "mature")
    external = cohort != task
    old = fixed.parent/"project" if external else BASE/"selected"/task
    previous = PRIOR/cohort if external else PRIOR/"public"/task
    paths = {"old":old, "fixed":fixed, "mature":paper, "previous":previous}
    if external:
        paths["simple"] = fixed.parent/"simple"
    arms = ("mixed",) if task == "gas" else ARMS
    paths.update({arm:OUT/arm/cohort for arm in arms})
    replies = {arm:read(path/"driver-exchanges.json")["exchanges"] if arm in ("mature","simple") else []
               for arm,path in paths.items()}
    scores, streams, control = measure_streams(cohort, paths, replies)
    receipts = {}
    if task != "gas":
        receipts = {arm:training_receipts(paths[arm],task) for arm in arms}
        first = {(rr[0]["parent"],rr[0]["manifest"]) for rr in receipts.values()}
        if len(first) != 1:
            raise ValueError("首轮起点或诊断分界不同")
    for arm, values in streams.items():
        scores[arm]["different_from_old"] = sum(p != streams["old"][i] for i,p in values.items())
    atomic_json(OUT/"scores"/(cohort+".json"), {"cohort":cohort, "inputs_and_budget":control,
                                               "arms":scores, "training_receipts":receipts,
                                               "same_parent_first_service":first_pair(cohort, paths) if receipts else {}})
    print(cohort, {a:round(v["score"],4) for a,v in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("run","score","transfer","history"))
    parser.add_argument("name",choices=(*ARMS,*RUL,"gas","all"))
    args = parser.parse_args()
    if args.operation == "run" and args.name in ARMS:
        execute(args.name)
    elif args.operation == "score" and args.name in (*RUL,"gas","all"):
        for cohort in (*RUL,"gas") if args.name == "all" else (args.name,):
            score(cohort)
    elif args.operation == "transfer" and args.name in ("phm08","pronostia","all"):
        for cohort in ("phm08","pronostia") if args.name == "all" else (args.name,):
            transfer(cohort, additional={a:OUT/a/cohort for a in ARMS}, destination=OUT)
    elif args.operation == "history" and args.name in ("phm08","pronostia","all"):
        for cohort in ("phm08","pronostia") if args.name == "all" else (args.name,):
            transfer(cohort, destination=OUT/"history", history=True)
    else:
        parser.error("操作与对照名称不匹配")
