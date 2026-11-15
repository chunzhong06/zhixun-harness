"""真实维护三轮，按实际生效的模型评价下一段，同时复核被拒候选的预测。"""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from compare_gas_paper_modules import (
    ROOT, PACKAGE, SOURCE, atomic_json, author_predict, compile_bridge, dependencies,
    export_weights, np, predicted, repair, sha,
)
from harness.data.official_gas import read_official_gas
from harness.evaluation.gas_pilot import partition, run_arm
from harness.models.model_artifacts import ModelArtifacts

CONFIG = ROOT / "engineering/configs/experiments/official-gas-observed-queries.json"
ROOT_RESULTS = ROOT / "results/experiments/official-gas"
OUTPUT = ROOT_RESULTS / "observed-queries"


def compare():
    config = json.loads(CONFIG.read_bytes())
    package = read_official_gas(PACKAGE)
    batches, _ = partition(package, config)
    stages = batches[1:config["maintenance_prefix"]+1]
    windows = [[r for r in batch.samples if r.role == "explore"] for batch in stages]
    cases = [(batch.batch_id,r.sample_id) for batch,rows in zip(stages,windows) for r in rows]
    results = {arm:run_arm(PACKAGE, CONFIG, OUTPUT, arm) for arm in config["arms"]}
    previous = json.loads((ROOT_RESULTS/"pilot-11/sequence-protected/state.json").read_bytes())
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model
    with TemporaryDirectory(prefix="gas-observed-score-") as tmp:
        work = Path(tmp)
        weights = work/"model.npz"
        weights.write_bytes(package.checkpoint)
        x = np.asarray([r.values for rows in windows for r in rows])
        predictions = {"static":predicted(Model.load(weights), x)}
        export_weights(weights, work/"official.bin")
        cp = compile_bridge(work, dependencies(SOURCE.parent.parent/"runtime-dependencies"))
        predictions["patching"], author_cost = author_predict(work, cp, work/"official.bin",
            np.asarray([r.values for r in package.train]), np.asarray([r.label-1 for r in package.train]), x)
        old = []
        for bid,sid in cases:
            incumbent = previous["completed"][bid]["decision"]["current_model"]
            old.append(previous["operations"][f"{bid}:explore:{incumbent}"]["value"]["predictions"][sid])
        predictions["old_harness"] = np.array(old)
        for arm, result in results.items():
            folder = OUTPUT/arm
            state = json.loads((folder/"state.json").read_bytes())
            artifacts = ModelArtifacts(folder/"models")
            served, trial = [], []
            for i,(window,round_) in enumerate(zip(windows,result["rounds"],strict=True)):
                inputs = np.asarray([r.values for r in window])
                selected = round_["selected"]
                candidate = round_["backtested_candidates"]
                if len(candidate) > 1:
                    raise ValueError("超出每轮一个候选的预算")
                for mid, collection in [(selected,served),(candidate[0]["model"] if candidate else selected,trial)]:
                    _, content = artifacts.read(state["models"][mid]["artifact"])
                    weights.write_bytes(content)
                    collection.append(predicted(Model.load(weights), inputs))
                if i+1 < len(result["rounds"]):
                    next_id = stages[i].batch_id
                    logged = state["operations"][f"{next_id}:explore:{selected}"]["value"]["predictions"]
                    if not np.array_equal(served[-1], [logged[r.sample_id] for r in window]):
                        raise ValueError("重载权重的预测与下一段实际在用模型不符")
            predictions[arm] = np.concatenate(served)
            predictions[arm+"-candidate"] = np.concatenate(trial)
    issued = OUTPUT/"issued.jsonl"
    with issued.open("w",encoding="utf8") as stream:
        for i,(bid,sid) in enumerate(cases):
            row = {"stage":bid,"id":sid,**{k:int(v[i]) for k,v in predictions.items()}}
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    # 两组的训练、审批和用于对照的下一段预测均已固定，现在读取答案计分。
    metrics, offset = {}, 0
    for stage,rows in zip(stages,windows,strict=True):
        truth = np.asarray([r.label for r in rows])
        idx = slice(offset,offset+len(rows))
        metrics[stage.batch_id] = {k:repair(truth,v[idx],predictions["static"][idx]) for k,v in predictions.items()}
        offset += len(rows)
    summary = {key:{"mean":float(np.mean([s[key]["macro_f1"] for s in metrics.values()])),
                    "worst":min(s[key]["macro_f1"] for s in metrics.values()),
                    "repaired":sum(s[key]["repaired"] for s in metrics.values()),
                    "broken":sum(s[key]["broken"] for s in metrics.values())} for key in predictions}
    report = {"protocol":sha(CONFIG),"issued_sha256":sha(issued), "next_stage_counts":list(map(len,windows)),
              "summary":summary,"stages":metrics,"arms":{arm:{"epochs":r["total_epochs"],
              "fit_job_cpu_seconds":r["training_worker_cpu_seconds"],"holdout":r["holdout"]["macro_f1"],
              "decisions":[{"round":v["round"],"current":v["incumbent"],"served":v["selected"],
                            "reason":v["decision"],"confirmation":v["confirmation"]} for v in r["rounds"]]}
              for arm,r in results.items()},"patching_ns":author_cost}
    atomic_json(OUTPUT/"comparison.json",report)
    print(json.dumps(report,ensure_ascii=False))


def verify_reuse():
    """从公开包重走相同提案规则，比较实际作业以及每个下一段的候选、在用预测。"""
    config_path = ROOT/"engineering/configs/experiments/official-gas-query-reuse.json"
    config = json.loads(config_path.read_bytes())
    directory = ROOT_RESULTS/"observed-queries-reuse"
    arm = "query-current"
    new = run_arm(PACKAGE, config_path, directory, arm)
    old = json.loads((OUTPUT/arm/"result.json").read_bytes())
    package = read_official_gas(PACKAGE)
    batches, _ = partition(package,config)
    previous = list(map(json.loads,(OUTPUT/"issued.jsonl").read_bytes().splitlines()))
    src, dst = OUTPUT/arm, directory/arm
    old_state,new_state = [json.loads((p/"state.json").read_bytes()) for p in (src,dst)]
    old_records,new_records = [json.loads((p/"query-proposals.json").read_bytes())["records"] for p in (src,dst)]
    if old_records != new_records or old["holdout"]["predictions"] != new["holdout"]["predictions"]:
        raise ValueError("提案或留出预测变化")
    old_models,new_models = ModelArtifacts(src/"models"),ModelArtifacts(dst/"models")
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model
    compared = []
    with TemporaryDirectory(prefix="gas-query-reuse-") as tmp:
        weight = Path(tmp)/"model.npz"
        for i,(left,right) in enumerate(zip(old["rounds"],new["rounds"],strict=True)):
            bid = left["round"]
            if (bid != right["round"] or left["decision"] != right["decision"]
                    or left["confirmation"] != right["confirmation"]):
                raise ValueError("本轮换版决定变化")
            lt,rt = old_state["completed"][bid]["trials"][0],new_state["completed"][bid]["trials"][0]
            if lt["training_ids"] != rt["training_ids"] or lt["exploration_score"] != rt["exploration_score"]:
                raise ValueError("训练行或当轮探索成绩变化")
            stage = batches[i+1]
            future = [r for r in stage.samples if r.role == "explore"]
            logged = [r for r in previous if r["stage"] == stage.batch_id]
            if [r.sample_id for r in future] != [r["id"] for r in logged]:
                raise ValueError("配对观测顺序错误")
            identities = {}
            for role, lid, rid, key in (("candidate",lt["model_id"],rt["model_id"],arm+"-candidate"),
                                        ("served",left["selected"],right["selected"],arm)):
                lm,_ = old_models.read(old_state["models"][lid]["artifact"])
                rm,content = new_models.read(new_state["models"][rid]["artifact"])
                weight.write_bytes(content)
                output = predicted(Model.load(weight), np.asarray([r.values for r in future]))
                if lm["sha256"] != rm["sha256"] or not np.array_equal(output,[r[key] for r in logged]):
                    raise ValueError("复用后模型内容或下一段逐条预测发生变化")
                identities[role] = rm["sha256"]
            compared.append({"round":bid,"weights":identities,"next_count":len(future),
                             "reused":bool(right["candidate"].get("reused")) if right["candidate"] else
                             any(e["kind"]=="fit_reused" and e["batch_id"]==bid for e in new_state["events"])})
    def costs(path):
        jobs = [json.loads(f.read_bytes()) for f in (path/"jobs").glob("*/state.json")]
        if not jobs or any(j["status"]!="completed" for j in jobs):
            raise ValueError("费用记录不完整")
        return {"jobs":len(jobs), "cpu_seconds":sum(j["cpu_seconds"] for j in jobs)}
    report = {"protocol":sha(config_path),"rounds":compared,"all_next_predictions_identical":True,
              "heldout_count":len(new["holdout"]["predictions"]), "heldout_predictions_identical":True,
              "before":{"epochs":old["total_epochs"],"fit_cpu":old["training_worker_cpu_seconds"],**costs(src)},
              "after":{"epochs":new["total_epochs"],"fit_cpu":new["training_worker_cpu_seconds"],**costs(dst)}}
    atomic_json(directory/"comparison.json",report)
    print(json.dumps(report))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-reuse",action="store_true")
    if parser.parse_args().verify_reuse:
        verify_reuse()
    else:
        compare()
