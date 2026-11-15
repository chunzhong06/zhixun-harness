"""只读复算固定训练均值对已交接预测的影响，含两个已封存的末版跨来源检验。"""
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from math import isclose

import numpy as np

from check_prior_device_prediction import OUT as PRIOR
from compare_error_slice import RAW, SUBSETS, read_cmapss, test_answers
from compare_group_objectives import BASE, ROOT, atomic_json, delivered, digest, metric, official, read, reference_folders, summary
from compare_proposal_frequency import TASKS, remaining_cycles, remaining_minutes
from harness.models.model_artifacts import ModelArtifacts

DEST = BASE/"service-cadence"/"source-mean"


def stored_mean(blob):
    with np.load(BytesIO(blob), allow_pickle=False) as checkpoint:
        meta = json.loads(str(checkpoint["metadata"]))
    value = float(meta["target_mean"])
    if meta["task"]!="regression" or not np.isfinite(value) or value<0:
        raise ValueError("已发布模型没有有限的寿命均值")
    return value


def transfer_truth(task):
    if task == "engine":
        groups, truth = {}, {}
        for subset in SUBSETS:
            units = read_cmapss(RAW,subset=subset,partition="test")
            group = test_answers(RAW,subset,units)
            truth.update(group)
            groups.update(dict.fromkeys(group,subset))
        return truth,groups
    rows = read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
    end = datetime.fromisoformat(rows[-1]["time"])
    values = {"KAIST/"+r["file"]:(end-datetime.fromisoformat(r["time"])).total_seconds()/60
              for r in rows if "invalid_nonfinite_values" not in r}
    return values,dict.fromkeys(values,"KAIST")


def evaluate(cohort, transfer=False):
    task = TASKS[cohort]
    package, _, _ = official(task)
    target_mean = stored_mean(package.checkpoint)
    fixed, paper = reference_folders(cohort,"mature")
    old = BASE/"selected"/task if task==cohort else fixed.parent/"project"
    paths = {"project":old, "paper":paper}
    name = cohort
    if transfer:
        name = "kaist" if task=="bearing" else "cmapss"
        folder = ROOT/"results/experiments/production-selection"/(
            "short-window/bearing/kaist" if task=="bearing" else "engine-transfer")
    else:
        folder = PRIOR/cohort
    source = folder/"issued.json"
    baseline = read(folder/"scores.json")
    if sha256(source.read_bytes()).hexdigest() != baseline["issued_sha256"]:
        raise ValueError("旧预测与原评分摘要不符")
    prior = read(source)
    output = {}
    for arm,path in paths.items():
        records = read(path/"run.json")["rounds"]
        store = ModelArtifacts(path/"models")
        releases = [records[-1]] if transfer else records
        for r in releases:
            header,blob = store.read(r["model_artifact"])
            if stored_mean(blob)!=target_mean or read(path/r["submitted"])["payload"]["model_id"] != r["selected"]:
                raise ValueError("在用版均值或交接身份不同")
        alias = "mature" if arm=="paper" else arm
        if transfer:
            hashes = prior["model_sha256"] if task=="bearing" else prior["source"]["weights"]
            if header["sha256"]!=hashes[alias]:
                raise ValueError("用于移用的末版身份不符")
            raw = prior["predictions"][alias]
        else:
            raw = prior["outputs"][arm]["parent"]
        output[arm] = {"parent":raw, "mean":dict.fromkeys(raw,target_mean),
                       "half":{i:(v+target_mean)/2 for i,v in raw.items()}}
    sealed = DEST/name/"issued.json"
    receipt = {"source_sha256":baseline["issued_sha256"], "source_mean":target_mean,"outputs":output}
    if sealed.exists() and read(sealed)!=receipt:
        raise ValueError("重算预测不同")
    atomic_json(sealed,receipt)
    if transfer:
        truth,devices = transfer_truth(task)
    else:
        samples = read(fixed/"state.json")["samples"]
        devices = {i:r["unit_id"] for i,r in samples.items() if r["role"]=="serve"}
        if cohort==task:
            truth = {r.sample_id:r.label for r in package.validation}
        else:
            truth = remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
    table = {arm:summary(task,truth,streams,devices) for arm,streams in output.items()}
    for arm, row in table.items():
        alias = "mature" if arm=="paper" else arm
        if transfer:
            expected = baseline["arms"][alias]["rmse"] if task=="bearing" else baseline["rmse"][alias]["pooled"]
        else:
            expected = baseline["scores"][arm]["parent"]["score"]
        if not isclose(row["parent"]["score"], expected, rel_tol=1e-10):
            raise ValueError("原分与独立回执不匹配")
    atomic_json(DEST/name/"scores.json", {"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),
                                        "source_mean":target_mean, "arms":table})
    print(name, round(target_mean,2), {a:[round(v["score"],2) for v in s.values()] for a,s in table.items()})


def initial_feedback(cohort, fixed, samples, roles, checkpoint_sha, archive_sha):
    """只取官方根权重产生的预测，额外归档的工作者必须使用相同输入及同一来源。"""
    task = TASKS[cohort]
    if cohort == task:
        file = ROOT/f"results/experiments/official-{task}/baseline/{task}-baseline.json"
        saved = read(file)
        if saved["checkpoint_sha256"]!=checkpoint_sha or saved["archive_sha256"]!=archive_sha:
            raise ValueError("首轮参照不是同一官方检查点")
        return {role:{i:saved["predictions"][i] for i in ids} for role,ids in roles.items()}
    predictions = {role:{} for role in roles}
    for arm in ("fixed","project","mature","simple"):
        folder = fixed.parent/arm
        state = read(folder/"state.json")
        root, = [m for m in state["models"].values() if m["parent_id"] is None]
        if ModelArtifacts(folder/"models").read(root["artifact"])[0]["sha256"]!=checkpoint_sha:
            raise ValueError("首轮诊断使用了不同的父权重")
        for role, ids in roles.items():
            op = state["operations"].get("round-1:"+role+":"+root["model_id"])
            if op is None:
                continue
            job = folder/"jobs"/op["execution"]["request_id"]
            packet, status = read(job/"input.json"), read(job/"state.json")
            req = packet["request"]
            expected_rows = {i:samples[i]["values"] for i in ids}
            if (req["artifact"]!=root["artifact"] or req["inputs_digest"]!=digest(packet["rows"])
                    or req["package_sha256"]!=archive_sha or digest(req)!=status["request_id"]
                    or status["nonce"]!=packet["nonce"] or status["status"]!="completed"
                    or {r["sample_id"]:r["values"] for r in packet["rows"]}!=expected_rows
                    or set(status["value"])!=ids or op["value"]["predictions"]!=status["value"]):
                raise ValueError("首轮反馈与已确认的工作进程预测不符")
            if predictions[role] and predictions[role]!=status["value"]:
                raise ValueError("不同运行的同一根权重预测不一致")
            predictions[role] = status["value"]
    if any(set(predictions[r])!=ids for r,ids in roles.items()):
        raise ValueError("归档缺少完整的首轮反馈")
    return predictions


def transfer_decision(cohort, accepted, mean, checkpoint_sha):
    task = TASKS[cohort]
    group = "kaist" if task=="bearing" else "cmapss"
    folder = ROOT/"results/experiments/production-selection"/(
        "short-window/bearing/kaist" if task=="bearing" else "engine-transfer")
    src = folder/"issued.json"
    expected_digest = read(folder/"scores.json")["issued_sha256"]
    prior = read(src)
    hashes = prior["model_sha256"] if task=="bearing" else prior["source"]["weights"]
    if hashes["original"]!=checkpoint_sha or sha256(src.read_bytes()).hexdigest()!=expected_digest:
        raise ValueError("移用没有复现原权重及已封存输出")
    raw = prior["predictions"]["original"]
    streams = {"parent":raw, "gated":{i:(v+mean)/2 for i,v in raw.items()} if accepted else raw,
               "old":prior["predictions"]["project"], "paper":prior["predictions"]["mature"]}
    path = DEST/"first-proposal"/group/"issued.json"
    record = {"source_decision":cohort,"accepted":accepted, "parent_sha256":checkpoint_sha,
              "source_prediction_sha256":expected_digest, "outputs":streams}
    if path.exists() and read(path)!=record:
        raise ValueError("移用输出未复现")
    atomic_json(path,record)
    truth, groups = transfer_truth(task)
    table = summary(task,truth,streams,groups)
    atomic_json(path.with_name("scores.json"),{"issued_sha256":sha256(path.read_bytes()).hexdigest(),
                                              "source_decision":cohort,"accepted":accepted,"arms":table})
    print("transfer",group,{k:round(v["score"],2) for k,v in table.items()})


def one_proposal(cohort):
    task = TASKS[cohort]
    pkg, checksum, _ = official(task)
    mean = stored_mean(pkg.checkpoint)
    fixed, paper = reference_folders(cohort,"mature")
    samples = read(fixed/"state.json")["samples"]
    roles = {role:{i for i,r in samples.items() if r["role"]==role and r["available_at"]==0}
             for role in ("explore","adopt")}
    raw = initial_feedback(cohort, fixed, samples, roles, pkg.checkpoint_sha, checksum)
    scores = {}
    for role, preds in raw.items():
        labels = {i:samples[i]["label"] for i in roles[role]}
        scores[role] = [metric(task,labels,preds), metric(task,labels,{i:(v+mean)/2 for i,v in preds.items()})]
    adopt = all(a-b>1e-12 for a,b in scores.values())
    original = BASE/"original"/task if task==cohort else fixed.parent/"original"
    version = ModelArtifacts(original/"models")
    rounds = read(original/"run.json")["rounds"]
    for r in rounds:
        if version.read(r["model_artifact"])[0]["sha256"] != pkg.checkpoint_sha:
            raise ValueError("静态交接中出现不同权重")
    pred = delivered(original)
    half = {i:(v+mean)/2 for i,v in pred.items()}
    old = BASE/"selected"/task if task==cohort else fixed.parent/"project"
    streams = {"parent":pred,"half":half,"gated":half if adopt else pred,"old":delivered(old),"paper":delivered(paper)}
    result = DEST/"first-proposal"/cohort
    path = result/"issued.json"
    receipt = {"parent_sha256":pkg.checkpoint_sha, "feedback_digest":digest(raw), "feedback_scores":scores,
               "accepted":adopt,"service_rounds":len(rounds),"outputs":streams}
    if path.exists() and read(path)!=receipt:
        raise ValueError("首轮决定或后续预测未复现")
    atomic_json(path,receipt)
    truth = {r.sample_id:r.label for r in pkg.validation}
    if cohort=="phm08":
        truth = remaining_cycles()
    elif cohort=="pronostia":
        truth = remaining_minutes("full-test-manifest.json")
    devices = {i:r["unit_id"] for i,r in samples.items() if r["role"]=="serve"}
    if set(pred)!=set(devices) or any(samples[i]["label"] is not None for i in pred):
        raise ValueError("服务集与已确认的原分片不同")
    table = summary(task,truth,streams,devices)
    atomic_json(result/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(), "arms":table,
        "accepted":adopt, "feedback_scores":scores, "feedback_counts":{k:len(v) for k,v in roles.items()}})
    print("first",cohort,adopt,{k:round(v["score"],2) for k,v in table.items()})
    if cohort!=task:
        transfer_decision(cohort, adopt, mean, pkg.checkpoint_sha)


if __name__ == "__main__":
    for cohort in ("bearing","pronostia","engine","phm08"):
        evaluate(cohort)
        one_proposal(cohort)
        if cohort!=TASKS[cohort]:
            evaluate(cohort,transfer=True)
