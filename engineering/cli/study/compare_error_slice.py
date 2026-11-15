"""重跑提案改动的外部服务流；与旧交付及成熟适配的真实逐条交接对照。"""
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_proposal_frequency import (BASE, ROOT, TASKS, LocalStore, asdict, assemble_public,
                                        atomic_json, digest, external_batches, measure_streams,
                                        read, reference_folders, run_prepared)
from engine_endpoint_check import RAW, SUBSETS, PACKAGES, read_cmapss, test_answers
from harness.models.model_artifacts import ModelArtifacts
from harness.models.bearing_stumps import stump_model
from harness.models.official_rul_cpu import _trainer

DEST = BASE / "error-slice"
COHORTS = ("gas", "bearing", "engine", "pronostia", "phm08")


def run_external(cohort, *, policy="error-slice", destination=DEST):
    task = TASKS[cohort]
    fixed, _ = reference_folders(cohort, "mature")
    driver = {"mode":"rule"}
    path = destination / cohort
    with LocalStore(path / "study-lock").exclusive():
        harness, official = assemble_public(task, ROOT / "data/official", path, policy, driver)
        batches = external_batches(task, official, match_window=cohort == "pronostia")
        # 外部设备只参与分开的探索、确认及服务；梯度更新沿用同一官方训练池。
        harness.tools.values = {s.sample_id:s.values for b in batches for s in b.samples if s.role != "fit"}
        old = read(fixed / "state.json")["completed"]
        if any(old[b.batch_id]["input_digest"] != digest(asdict(b)) for b in batches):
            raise ValueError("外部输入与原参照不同")
        run_prepared(task, path, policy, 10, driver, harness, batches)


def score(cohort):
    task = TASKS[cohort]
    fixed, paper = reference_folders(cohort, "mature")
    if task == cohort:
        selected, candidate = BASE / "selected" / task, DEST / "public" / task
    else:
        selected, candidate = fixed.parent / "project", DEST / cohort
    paths = {"selected":selected, "fixed":fixed, "mature":paper, "error-slice":candidate}
    if cohort != task:
        # 此两套外部资料另有同预算的简单代理，保留这一较强历史参照。
        paths["simple"] = fixed.parent / "simple"
    replies = {k:[] for k in paths}
    replies["mature"] = read(paper / "driver-exchanges.json")["exchanges"]
    if "simple" in paths:
        replies["simple"] = read(paths["simple"] / "driver-exchanges.json")["exchanges"]
    results, streams, checks = measure_streams(cohort, paths, replies)
    for name, stream in streams.items():
        results[name]["changed_from_selected"] = sum(v != streams["selected"][i] for i,v in stream.items())
    rows = read(candidate / "run.json")["rounds"]
    atomic_json(DEST / "scores" / (cohort+".json"), {
        "cohort":cohort, "input_and_budget":checks, "arms":results,
        "candidate_rounds":[{k:r[k] for k in ("batch", "planned_actions", "current", "selected", "outcome")} for r in rows]})
    print(cohort, {k:round(v["score"],5) for k,v in results.items()}, flush=True)


def released_checkpoints(folder, arm, history):
    """每个版本须与源流真实交接相符。历史诊断只允许沿各方法自己的路径取权重。"""
    rounds = read(folder/"run.json")["rounds"]
    if len(rounds) != 10:
        raise ValueError(f"{arm} 缺少十轮交接")
    selected = {}
    if history:
        models = read(folder/"state.json")["models"]
        selected[arm+"@00"] = models[rounds[0]["current"]]["artifact"]
    for i, record in enumerate(rounds, start=1):
        if i < len(rounds) and not history:
            continue
        if read(folder/record["submitted"])["payload"]["model_id"] != record["selected"]:
            raise ValueError(f"{arm} 第 {i} 轮模型与交接不符")
        if history and models[record["selected"]]["artifact"] != record["model_artifact"]:
            raise ValueError("实际权重与模型登记不符")
        selected[f"{arm}@{i:02}"] = record["model_artifact"]
    selected[arm] = selected.pop(f"{arm}@10")
    return selected


def transfer(cohort, *, additional=None, destination=DEST, history=False):
    """先用源流交出的权重产生逐设备预测，再计算目标来源的误差。"""
    task = TASKS[cohort]
    historical = ROOT / "results/experiments/production-selection"
    fixed, mature = reference_folders(cohort, "mature")
    if task == "bearing":
        manifest = ROOT / "data/public/kaist-bearing-v6/hourly-windows.json"
        legacy = read(historical / "short-window/bearing/kaist/issued.json")
        if legacy["source_sha256"] != sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("KAIST 切片与原对照不符")
        windows = read(manifest)["rows"]
        valid = [r for r in windows if "invalid_nonfinite_values" not in r]
        ids = ["KAIST/"+r["file"] for r in valid]
        inputs = np.asarray([r["features"] for r in valid])
    else:
        legacy = read(historical / "engine-transfer/issued.json")
        tracks = {s:read_cmapss(RAW, subset=s, partition="test") for s in SUBSETS}
        devices = [t for group in tracks.values() for t in group]
        ids = [t.unit_id for t in devices]
        inputs = np.asarray([(*t.values[-1][3:], *t.values[-1][:3]) for t in devices])
    folders = {"project":fixed.parent/"project", "fixed":fixed, "mature":mature,
               "simple":fixed.parent/"simple", "error-slice":DEST/cohort}
    if additional:
        folders.update(additional)
    name, expected, _ = PACKAGES[task]
    archive = ROOT / "data/official" / task / name
    if sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("预测原包身份不符")
    predictions, hashes = {}, {}
    with TemporaryDirectory() as tmp, ZipFile(archive) as pack, _trainer(pack, Path(tmp), "B" if task=="bearing" else "C") as trainer:
        cached = {}
        for arm, folder in folders.items():
            store = ModelArtifacts(folder/"models")
            releases = released_checkpoints(folder, arm, history and arm in ("project", "mature"))
            for label, reference in releases.items():
                header, blob = store.read(reference)
                identity = header["sha256"]
                hashes[label] = identity
                if identity not in cached:
                    model_path = Path(tmp)/"checkpoint.npz"
                    model_path.write_bytes(blob)
                    predictor = stump_model(trainer.Model) if "bearing_stump" in header["request"] else trainer.Model
                    values = predictor.load(model_path).predict(inputs)
                    if values.shape != (len(ids),) or not np.isfinite(values).all():
                        raise ValueError("移用预测无效")
                    cached[identity] = dict(zip(ids, map(float,values), strict=True))
                predictions[label] = cached[identity]
            if arm in ("project", "fixed", "mature", "simple"):
                old = legacy["predictions"][arm]
                old_hash = legacy["model_sha256"] if task=="bearing" else legacy["source"]["weights"]
                if hashes[arm] != old_hash[arm] or set(old)!=set(ids) or not np.allclose(list(predictions[arm].values()),[old[i] for i in ids],rtol=1e-12,atol=1e-8):
                    raise ValueError(f"{arm} 未复现相同权重的封存逐条预测")
    output = destination/"transfer"/task
    atomic_json(output/"issued.json", {"weights":hashes, "input_sha256":digest([ids,inputs.tolist()]),
                                     "predictions":predictions})
    if task=="bearing":
        end = datetime.fromisoformat(windows[-1]["time"])
        groups = {"pooled":{i:(end-datetime.fromisoformat(r["time"])).total_seconds()/60
                            for r,i in zip(valid, ids, strict=True)}}
    else:
        groups = {s:test_answers(RAW,s,group) for s,group in tracks.items()}
        groups["pooled"] = {i:y for g in groups.values() for i,y in g.items()}
    scores = {arm:{s:float(np.sqrt(np.mean([(out[i]-y)**2 for i,y in truth.items()])))
                   for s,truth in groups.items()} for arm,out in predictions.items()}
    atomic_json(output/"scores.json", {"issued_sha256":sha256((output/"issued.json").read_bytes()).hexdigest(),
                                      "observations":len(ids),"rmse":scores})
    print(task, {k:round(v["pooled"],3) for k,v in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("external", "score", "transfer"))
    parser.add_argument("cohort", choices=(*COHORTS, "all"))
    args = parser.parse_args()
    if args.operation != "score" and args.cohort not in ("pronostia", "phm08", "all"):
        parser.error("外部运行和末版移用只接收两个外部来源；官方包用 public-run 运行")
    cohorts = COHORTS if args.operation == "score" else ("pronostia", "phm08")
    for cohort in (cohorts if args.cohort == "all" else [args.cohort]):
        {"external":run_external, "transfer":transfer, "score":score}[args.operation](cohort)
