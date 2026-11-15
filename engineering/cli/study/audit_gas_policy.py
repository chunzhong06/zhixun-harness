"""配对比较选择方向的规则；归因范围是同工具、同信息和同预算的十轮公开气体回放。"""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.runtime.storage import atomic_json
from harness.contracts import digest


def read(path):
    return json.loads(path.read_bytes())


def transfer_replay(folder, last_batch, final_weight):
    """只在历史评价侧比较整批使用已确认修正或原版。采集批标识来自原始文件，逐行答案随后计分。"""
    from harness.data.gas_input import read_gas_batches, REGISTERED_SHA256
    from harness.evaluation.round_evidence import gas_service_coverage

    state = read(folder/"sequence-first-window"/"state.json")
    benchmark = ROOT/"results/experiments/official-gas/patching-comparison/run-36"
    saved = read(benchmark/"result.json")
    raw = (benchmark/"issued.jsonl").read_bytes()
    if (final_weight != "sha256:"+saved["project_sha256"] or saved["issued_sha256"] != sha256(raw).hexdigest()
            or saved["source_sha256"] != REGISTERED_SHA256):
        raise ValueError("跨批参照与所发布修正的权重、原始输入或逐条预测不符")
    observations = {(r["batch"],r["line"]):r for r in map(json.loads,raw.splitlines())}
    if len(observations) != len(raw.splitlines()):
        raise ValueError("跨批预测含重复行号")
    # 查共同错误时纳入既有论文选样、等量随机、续训及修正组合。各方法费用不同；
    # 逐条知道真值后择优只用于定位工具缺口，不是任何预算下的服务成绩。
    library = {}
    for name, rel, field in (("subset","subset-direction/measured","issued"),
                             ("fusion","module-smoke/fusion","issued_sha256")):
        path = folder.parent/rel
        issued = (path/"issued.jsonl").read_bytes()
        if sha256(issued).hexdigest() != read(path/"result.json")[field]:
            raise ValueError(f"{name} 候选预测与实验回执不符")
        records = [r for r in map(json.loads,issued.splitlines()) if r["batch"] != "next"]
        indexed = {(int(r["batch"]),r["id"]):r for r in records}
        if len(indexed) != len(records) or indexed.keys() != observations.keys():
            raise ValueError(f"{name} 候选没有与同一批的同一行配对")
        library[name] = indexed
    stages, consumed = [], set()
    base_errors, shared, per_tool = Counter(), Counter(), Counter()
    for source in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip", expected_sha256=REGISTERED_SHA256):
        if source.batch_number not in range(3,9):
            continue
        coverage = gas_service_coverage(state,last_batch,{f"source-{source.batch_number}"})
        if coverage["status"] not in ("covered","outside"):
            raise ValueError("本对照要求有可追溯的采集批与当时的采用依据")
        choice = "project" if coverage["status"] == "covered" else "static"
        pairs = [observations[source.batch_number,r.source_line] for r in source.records]
        # 两版不一致的条数在答案回来前可观察；它不是正确率，不能自己给新版投票。
        discordant = sum(obs["static"] != obs["project"] for obs in pairs)
        rows = [(r.label,obs) for r,obs in zip(source.records,pairs,strict=True)]
        for r,obs in zip(source.records,pairs,strict=True):
            key = (source.batch_number,r.source_line)
            values = [group[key] for group in library.values()]
            if (any(t["static"] != obs["static"] for t in values)
                    or library["fusion"][key]["static+two"] != obs["project"]):
                raise ValueError("候选资料与原版或双修正的预测不符")
            if obs["static"] != r.label:
                direction = f"{r.label}->{obs['static']}"
                base_errors[direction] += 1
                options = {f"{name}/{k}":v for name,data in library.items() for k,v in data[key].items()
                           if k not in ("id","batch","static")}
                options.update(author=obs["author"],project=obs["project"])
                hits = [k for k,v in options.items() if v == r.label]
                per_tool.update(hits)
                if not hits:
                    shared[direction] += 1
        consumed.update((source.batch_number,r.source_line) for r in source.records)
        stages.append({"source":source.batch_number,"count":len(rows),"coverage":coverage,"scope_choice":choice,
            "prediction_disagreements_before_truth":discordant,
            "macro_f1":{a:macro_f1((y,obs[a]) for y,obs in rows) for a in ("static","author","project")},
            "foregone_repairs":sum(obs[choice]!=y and obs["project"]==y for y,obs in rows),
            "avoided_new_errors":sum(obs[choice]==y and obs["project"]!=y for y,obs in rows)})
    if consumed != observations.keys() or {x["source"] for x in stages} != set(range(3,9)):
        raise ValueError("来源分片与历史逐条预测未完整对应")
    return {"note":"比较按已知采集来源决定一次发布的范围；不训练、不模拟新来源出现反馈后的再确认。",
            "historical_issued_sha256":saved["issued_sha256"],"stages":stages,
            "candidate_coverage":{"scope":["subset-direction/measured","module-smoke/fusion","patching-comparison/run-36"],
                "baseline_errors":sum(base_errors.values()),"not_repaired_by_any":sum(shared.values()),
                "by_truth_to_original":{k:{"original_errors":n,"not_repaired_by_any":shared[k]}
                                        for k,n in base_errors.most_common()},
                "original_errors_repaired_by_candidate":dict(per_tool)}}


def service_replay(folder, package_path, static_folder):
    """只读已结束的作业。按当轮发布版本重放公开服务分片，不把末版预测倒填到早期。"""
    from engineering.cli.workflow.public_maintenance import gas_service_windows
    from harness.evaluation.gas_pilot import partition
    from harness.evaluation.round_evidence import gas_service_coverage

    comparison = audit(folder, package_path, persist=False)
    protocol = read(folder/"protocol.json")
    package = read_official_gas(package_path)
    batches, held = partition(package, protocol["protocol"])
    windows = gas_service_windows(batches, held)
    ids = [r.sample_id for r in held]
    if ids != protocol["holdout"] or len(set(ids)) != len(ids):
        raise ValueError("公开服务留出清单与原协议不同")
    truth = {r.sample_id:r.label for r in held}
    inputs = [{"sample_id":r.sample_id, "values":r.values} for r in held]
    input_key = digest(inputs)
    static = read(static_folder/"result.json")
    static_plan = read(static_folder.parent/"protocol.json")
    if (not static["baseline_reproduced"] or static["protocol_digest"] != digest(static_plan)
            or any(static_plan[k] != protocol[k] for k in ("package_sha256","holdout","training","validation","rounds"))):
        raise ValueError("官方原版来自不同权重或资料")
    original = static["holdout"]["predictions"]
    if set(original) != set(ids) or any(not package.task.accepts_target(v) for v in original.values()):
        raise ValueError("原版预测与留出身份或类别不符")

    # 用版本和全部输入的指纹直接定位当年的预测作业，再核对工作进程回执。
    # 早期版本和末版都已预测过这批资料，无需新推理，答案只用于下方计分。
    arms = {}
    for alias, arm in (("fixed","sequence-first-window"), ("adaptive","sequence-protected")):
        sub = folder/arm
        state, result = read(sub/"state.json"), read(sub/"result.json")
        versions, forecasts, jobs, weights = {}, {}, {}, {}
        for step in result["rounds"]:
            bid, mid = step["round"], step["selected"]
            if state["completed"][bid]["model_id"] != mid:
                raise ValueError("维护决定与已发布版本不一致")
            versions[bid] = mid
            if step["candidate"]:
                weights[step["candidate"]["model"]] = step["candidate"]["checkpoint_sha256"]
            if mid in forecasts:
                continue
            request = {"kind":"predict","artifact":state["models"][mid]["artifact"],"inputs_digest":input_key,
                       "package_sha256":protocol["package_sha256"],"evidence_kind":"real_model"}
            key = digest(request)
            saved, receipt, worker = (read(sub/"jobs"/key/file) for file in ("input.json","state.json","result.json"))
            preds = receipt["value"]
            if (saved["request"] != request or digest(saved["rows"]) != input_key or receipt["request_id"] != key
                    or receipt["status"] != worker["status"] or receipt["status"] != "completed"
                    or not saved["nonce"] == receipt["nonce"] == worker["nonce"] or preds != worker["value"]
                    or set(preds) != set(ids) or any(not package.task.accepts_target(v) for v in preds.values())):
                raise ValueError(f"{arm} {mid} 无匹配的真实预测回执")
            forecasts[mid], jobs[mid] = preds, key
        if forecasts[versions[windows[-1].batch_id]] != result["holdout"]["predictions"]:
            raise ValueError("末版预测未能重现")
        arms[alias] = versions, forecasts, jobs, weights, state

    def pair(left, right, sample_ids):
        changed = [s for s in sample_ids if left[s] != right[s]]
        return {"different":len(changed), "fixed":sum(right[s] == truth[s] for s in changed),
                "broken":sum(left[s] == truth[s] for s in changed)}

    rounds, pooled = [], {k:{} for k in ("original","first_only","fixed","adaptive")}
    for batch in windows:
        bid = batch.batch_id
        rows = [r for r in batch.samples if r.role == "serve"]
        if not rows or any(r.label is not None for r in rows):
            raise ValueError("服务没有与作决定时可见的答案分开")
        sample_ids = [r.sample_id for r in rows]
        versions = {a:v[0][bid] for a,v in arms.items()}
        preds = {a:v[1][versions[a]] for a,v in arms.items()}
        first_model = arms["fixed"][0][windows[0].batch_id]
        preds.update(original=original, first_only=arms["fixed"][1][first_model])
        for a, values in preds.items():
            if set(sample_ids) & pooled[a].keys():
                raise ValueError("服务分片重叠")
            pooled[a].update((s,values[s]) for s in sample_ids)
        rounds.append({"batch":bid, "ids":sample_ids, "models":versions,
            "sha256":{a:v[3][versions[a]] for a,v in arms.items()},
            "service_coverage":{a:gas_service_coverage(v[4],bid,{r.group for r in rows}) for a,v in arms.items()},
            "job":{a:v[2][versions[a]] for a,v in arms.items()},
            "macro_f1":{a:macro_f1((truth[s],p[s]) for s in sample_ids) for a,p in preds.items()},
            "adaptive_vs_fixed":pair(preds["fixed"],preds["adaptive"],sample_ids),
            "disagreements":[{"id":s,"truth":truth[s],"fixed":preds["fixed"][s],"adaptive":preds["adaptive"][s]}
                             for s in sample_ids if preds["fixed"][s]!=preds["adaptive"][s]]})
    if any(set(p) != set(ids) for p in pooled.values()):
        raise ValueError("服务分片遗漏留出样本")
    return {"kind":"retrospective_service_replay", "protocol_digest":digest(protocol),
        "note":"按事先存在的公开入口规则分片；原运行未提交这条服务流。first_only 固定使用首轮已采用版本。",
        "count":len(ids), "rounds":rounds, "heldout_prediction_input_digest":input_key,
        "transfer":transfer_replay(folder,rounds[-1]["batch"],rounds[-1]["sha256"]["fixed"]),
        "summary":{a:{"pooled":macro_f1((truth[s],p[s]) for s in ids),
                      "batch_mean":mean(r["macro_f1"][a] for r in rounds),
                      "worst":min(r["macro_f1"][a] for r in rounds),"vs_original":pair(original,p,ids)}
                   for a,p in pooled.items()},
        "adaptive_vs_fixed":pair(pooled["fixed"],pooled["adaptive"],ids),
        "historical_job_cpu_seconds":{r["arm"]:r["total_job_cpu_seconds_including_final_evaluation"]
                                      for r in comparison["arms"]}}


def audit(folder, package_path, *, persist=True):
    """默认保存研究对照；交付复核使用相同计算，但只读历史记录。"""
    plan = read(folder/"protocol.json")
    if sha256(package_path.read_bytes()).hexdigest()!=plan["package_sha256"]:
        raise ValueError("申报的官方气体包已改变")
    package = read_official_gas(package_path)
    answers = {r.sample_id:r.label for r in package.validation}
    rows = []
    arm_records = {}
    allowed = set(plan["protocol"]["arms"])
    expected = {"sequence-first-window", "sequence-protected"}
    if allowed not in (expected, expected | {"static"}):
        raise ValueError("气体取舍核查需要固定方向与逐轮选择两个对照")
    for arm in ("sequence-first-window", "sequence-protected"):
        sub = folder/arm
        result, state = read(sub/"result.json"), read(sub/"state.json")
        if (result["arm"]!=arm or state["evidence_kind"]!="real_model" or result["protocol_digest"]!=digest(plan)
                or [r["round"] for r in result["rounds"]] != list(plan["rounds"])):
            raise ValueError("需要两组的真实运行")
        frozen = []
        predictions = {}
        for r in result["rounds"]:
            batch = r["round"]
            observed = state["operations"][f"{batch}:explore:{r['incumbent']}"]["value"]["predictions"]
            if set(observed)!=set(plan["rounds"][batch]["explore"]) or predictions.keys() & observed.keys():
                raise ValueError("各轮须核对更新前的同一批样本")
            predictions.update(observed)
            candidate = r["candidate"]
            pair = candidate["gas_pair"] if candidate else None
            frozen.append({"round":batch,"before_macro_f1":macro_f1((answers[s],p) for s,p in observed.items()),
                "pair":pair, "adopted":r["selected"]!=r["incumbent"], "confirmation":r["confirmation"],
                "model":r["selected"], "explore_predictions":observed,
                "after_holdout_macro_f1":r["heldout_macro_f1"]})
        held = result["holdout"]["predictions"]
        if set(held)!=set(plan["holdout"]) or set(held)&predictions.keys():
            raise ValueError("末版评价身份与探索资料重合或不符")
        held_score = macro_f1((answers[s],p) for s,p in held.items())
        if abs(held_score-result["holdout"]["macro_f1"])>1e-12:
            raise ValueError("末版评分与原始预测不一致")
        jobs = [read(p) for p in (sub/"jobs").glob("*/state.json")]
        if not jobs or any(j["status"]!="completed" for j in jobs):
            raise ValueError("作业回执不齐")
        arm_records[arm] = {"rounds":frozen,"predictions":predictions,"holdout":held,"config":state["config"]}
        rows.append({"arm":arm, "initial_sha256":package.checkpoint_sha256,"before_pooled":macro_f1(
            (answers[s],p) for s,p in predictions.items()),"before_batch_mean":mean(x["before_macro_f1"] for x in frozen),
            "before_batch_worst":min(x["before_macro_f1"] for x in frozen),"holdout_macro_f1":held_score,
            "proposals":sum(x["pair"] is not None for x in frozen),"adopted":[x["round"] for x in frozen if x["adopted"]],
            "maintenance_wall_seconds":result["maintenance_wall_seconds"],"build_worker_cpu_seconds":result["training_worker_cpu_seconds"],
            "total_job_cpu_seconds_including_final_evaluation":sum(j["cpu_seconds"] for j in jobs),
            "peak_single_job_rss_bytes":max(j["observed_peak_rss_bytes"] for j in jobs),"gradient_epochs":result["total_epochs"]})
    fixed, adaptive = (arm_records[k] for k in ("sequence-first-window","sequence-protected"))
    if fixed["config"]!=adaptive["config"]:
        raise ValueError("控制器预算或其他配置不同")
    before_changes = []
    for fr, ar in zip(fixed["rounds"],adaptive["rounds"],strict=True):
        if fr["round"]!=ar["round"] or fr["explore_predictions"].keys()!=ar["explore_predictions"].keys():
            raise ValueError("轮次或预测样本不对应")
        for sid,f in fr["explore_predictions"].items():
            a = ar["explore_predictions"][sid]
            if f!=a:
                before_changes.append({"round":ar["round"],"id":sid,"answer":answers[sid],"fixed":f,"adaptive":a})
    result = {"arms":rows, "fixed_budget":fixed["config"], "before_changes":before_changes,
        "before_rows":len(fixed["predictions"]), "held_rows":len(fixed["holdout"]),
        "final_different_predictions":sum(v!=adaptive["holdout"][s] for s,v in fixed["holdout"].items()),
        "iterations":{arm:[{k:v for k,v in x.items() if k!="explore_predictions"} for x in data["rounds"]]
                      for arm,data in arm_records.items()}}
    if "static" in allowed:
        base = read(folder/"static/result.json")
        held = base["holdout"]
        if (not base["baseline_reproduced"] or base["protocol_digest"] != digest(plan)
                or set(held["predictions"]) != set(plan["holdout"])):
            raise ValueError("静态组与维护组的留出范围或官方起点不一致")
        score = macro_f1((answers[s], p) for s, p in held["predictions"].items())
        if abs(score - held["macro_f1"]) > 1e-12:
            raise ValueError("静态成绩与实际预测不符")
        result["static"] = {"holdout_macro_f1": score, "evaluated":len(held["predictions"]),
                            "initial_sha256":package.checkpoint_sha256}
    if persist:
        atomic_json(folder/"comparison.json",result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--package",type=Path,default=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
    parser.add_argument("--service-replay", action="store_true", help="单独保存各轮在用模型的服务分片回放")
    parser.add_argument("--static-folder",type=Path,default=ROOT/"results/experiments/official-gas/pilot-1/static")
    args=parser.parse_args()
    if args.service_replay:
        result = service_replay(args.output,args.package,args.static_folder)
        atomic_json(args.output/"service-replay.json",result)
        print(json.dumps({"summary":result["summary"],"adaptive_vs_fixed":result["adaptive_vs_fixed"]}))
    else:
        audit(args.output,args.package)
