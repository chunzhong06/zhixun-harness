"""用已发出的公开验证预测复核寿命误差；不加载模型，不改变任何预测。"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from harness.data.cmapss_input import condition_key
from harness.runtime.storage import atomic_json


def direction(value):
    return (value > 0) - (value < 0)


def audit(task):
    folder = ROOT / "results/experiments" / f"official-{task}" / "baseline"
    result_file = folder / f"{task}-baseline.json"
    baseline = json.loads(result_file.read_bytes())
    name, expected, _ = PACKAGES[task]
    if not baseline["matches_official"] or baseline["archive_sha256"] != expected:
        raise ValueError("需要已复现随包汇总的原版预测")
    predictions = baseline["predictions"]
    phase_errors = defaultdict(lambda: defaultdict(list))
    seen = set()
    with (ROOT / "data/official" / task / name).open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("评价包发生变化")
        source.seek(0)
        with ZipFile(source) as archive, archive.open("public/validation.jsonl") as rows:
            for line in rows:
                raw = json.loads(line)
                sid, target, meta = raw["id"], raw["answer"], raw["meta"]
                if sid in seen:
                    raise ValueError("重复评价样本")
                seen.add(sid)
                unit = meta.get("bearing") if task == "bearing" else f"{meta['source_subset']}:u{meta['unit']:04d}"
                if 1 <= target <= 30 or target > 100:
                    phase = "remaining_1_to_30" if target <= 30 else "remaining_over_100"
                    phase_errors[phase][unit].append(predictions[sid] - target)
    if seen != set(predictions):
        raise ValueError("已发预测与验证样本不对应")

    phases = {}
    for phase, units in phase_errors.items():
        errors = [e for device in units.values() for e in device]
        phases[phase] = {"observations": len(errors), "units": len(units), "mean_error": math.fsum(errors) / len(errors),
                         "over": sum(e > 0 for e in errors), "positive_mean_units": sum(math.fsum(v) > 0 for v in units.values())}

    pairs = []
    for unit, record in baseline["measured"]["units"].items():
        windows = [w for w in record["windows"] if w["rows"] == 50]
        for previous, current in zip(windows, windows[1:]):
            if previous["end"] + 1 != current["start"]:
                raise ValueError("设备窗口不连续")
            # 下一个窗口的方向只用来评分。每个规则只可使用这台设备此前已揭晓窗口的答案。
            pairs.append({"unit": unit, "start": current["start"], "end": current["end"],
                          "actual": direction(current["mean_signed_error"]),
                          "first_window": direction(windows[0]["mean_signed_error"]),
                          "last_window": direction(previous["mean_signed_error"])})
    scores = {}
    for method in ("always_over", "always_under", "first_window", "last_window"):
        by_unit = defaultdict(list)
        by_direction = defaultdict(list)
        for pair in pairs:
            estimate = 1 if method == "always_over" else -1 if method == "always_under" else pair[method]
            correct = estimate == pair["actual"]
            by_unit[pair["unit"]].append(correct)
            by_direction[pair["actual"]].append(correct)
        scores[method] = {
            "correct": sum(map(sum, by_unit.values())),
            "device_weighted_accuracy": (sum(sum(v) / len(v) for v in by_unit.values()) / len(by_unit)) if by_unit else None,
            "correct_by_actual_direction": {str(k): sum(v) for k, v in by_direction.items()},
        }
    outcome = {"source_report_sha256": sha256(result_file.read_bytes()).hexdigest(),
               "remaining_life_groups_for_retrospective_diagnosis": phases, "window_observations": 50,
               "paired_windows": len(pairs), "contributing_devices": len({p["unit"] for p in pairs}),
               "actual_direction_counts": dict(Counter(p["actual"] for p in pairs)), "rules": scores, "pairs": pairs}
    atomic_json(folder / "error-diagnosis.json", outcome)


def compare_engine_conditions():
    """只用已有模型的配对输出核对工况分流。先冻结确认组决定，再打开服务设备答案。"""
    folder = ROOT / "results/experiments/official-engine/fit-comparison"
    order = ("fixed", "original", "observed")
    source_hashes = {}

    def load(path):
        data = path.read_bytes()
        source_hashes[path.relative_to(ROOT).as_posix()] = sha256(data).hexdigest()
        return json.loads(data)

    runs = {name: load(folder/name/"result.json") for name in ("fixed", "observed")}
    states = {name: load(folder/name/"workflow/state.json") for name in runs}
    if runs["fixed"]["parent"] != runs["observed"]["parent"]:
        raise ValueError("两个选样分支起点不同")
    confirmation = {}
    for name, run in runs.items():
        ops = states[name]["operations"]
        decision = run["decision"]
        for key, model in (("original", decision["current_model"]), (name, decision["nominee"])):
            value = ops["maintenance-1:adopt:" + model]["value"]
            if key in confirmation and value["predictions"] != confirmation[key]["predictions"]:
                raise ValueError("原版在两组确认记录中的预测不同")
            confirmation[key] = value
    base = confirmation["original"]
    groups = {g["group"] for g in base["groups"]}
    if any(set(v["sample_versions"]) != set(base["sample_versions"]) or
           {g["group"] for g in v["groups"]} != groups for v in confirmation.values()):
        raise ValueError("确认样本修订或工况范围不同")
    group_scores = {k: {g["group"]: g["score"] for g in v["groups"]} for k,v in confirmation.items()}
    chosen = {g: min(order, key=lambda k: group_scores[k][g]) for g in sorted(groups)}
    global_choice = min(order, key=lambda k: confirmation[k]["score"])
    # 工况真正在预测时可见。仅当没有该工况的确认记录时才采用全局选择。
    samples = states["fixed"]["samples"]
    base_report = load(folder.parent/"baseline/engine-baseline.json")
    service = {name: v["candidate_service_predictions"] for name,v in runs.items()}
    ids = set(service["fixed"])
    if ids != set(service["observed"]) or ids & set(base["predictions"]):
        raise ValueError("两组服务样本未配对或参与了确认")
    service["original"] = {sid: base_report["predictions"][sid] for sid in ids}
    conf_units = {samples[sid]["unit_id"] for sid in base["predictions"]}
    assignment = load(folder/"fixed/protocol.json")["devices"]
    if assignment != load(folder/"observed/protocol.json")["devices"]:
        raise ValueError("两条分支的设备分工不一致")
    serve_units = {u for u, role in assignment.items() if role == "serve"}
    if conf_units & serve_units or conf_units != {u for u, role in assignment.items() if role == "adopt"}:
        raise ValueError("确认与服务未按整台设备隔离")
    name, expected, _ = PACKAGES["engine"]
    archive = ROOT / "data/official/engine" / name
    with archive.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("官方答案身份变化")
    source_hashes[archive.relative_to(ROOT).as_posix()] = expected
    squared = defaultdict(lambda: defaultdict(list))
    seen = set()
    with ZipFile(archive) as package, package.open("public/validation.jsonl") as rows:
        for line in rows:
            r = json.loads(line)
            sid = r["id"]
            unit = f"{r['meta']['source_subset']}:u{r['meta']['unit']:04d}"
            if unit not in serve_units:
                continue
            group = condition_key(r["inputs"]["settings"])
            if sid in seen or sid not in ids or (sid in samples and group != samples[sid]["group"]):
                raise ValueError("工况或样本身份不符")
            seen.add(sid)
            model_for_group = chosen.get(group, global_choice)
            for policy, model in [(k,k) for k in order] + [("by_condition", model_for_group)]:
                squared[policy][group].append((service[model][sid]-r["answer"])**2)
    if seen != ids:
        raise ValueError("服务组真值不完整")
    per_group = {k: {g: math.sqrt(math.fsum(v)/len(v)) for g,v in gs.items()} for k,gs in squared.items()}
    overall = {k: math.sqrt(math.fsum(math.fsum(v) for v in gs.values())/len(ids)) for k,gs in squared.items()}
    for k in runs:
        if not math.isclose(overall[k], runs[k]["service_comparison"]["candidate"]["rmse"]) or not math.isclose(
                overall["original"], runs[k]["service_comparison"]["original"]["rmse"]):
            raise ValueError("未能复现原表的逐观测 RMSE")
    oracle = {g: min(order, key=lambda k: per_group[k][g]) for g in squared["fixed"]}
    lower = math.sqrt(math.fsum(math.fsum(squared[k][g]) for g,k in oracle.items())/len(ids))
    atomic_json(folder/"工况选择核对.json", {"sources": source_hashes, "confirm_units": len(conf_units),
                "service_units": len(serve_units), "service_rows": len(ids), "confirmation_scores": group_scores,
                "global_choice": global_choice, "choices_by_condition": chosen, "service_rmse": overall,
                "service_group_rmse": per_group, "best_per_condition_in_hindsight": oracle,
                "best_possible_condition_rmse": lower, "new_training_or_inference": 0})


def compare_confirmation_cutoff():
    """在同一次已经完成的提名上替换确认行。只读取冻结的开发记录。"""
    spec_path = ROOT/"engineering/configs/experiments/engine-confirmation-cutoff.json"
    spec = json.loads(spec_path.read_bytes())
    raw = (ROOT/spec["source"]).read_bytes()
    if sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("开发回执发生变化")
    state = json.loads(raw)
    bid = "maintenance-1"
    decision = state["completed"][bid]["decision"]
    model_ids = [decision["current_model"], decision["nominee"]]
    evaluations = [state["operations"][f"{bid}:adopt:{m}"]["value"] for m in model_ids]
    base, new = evaluations
    if base["sample_versions"] != new["sample_versions"] or base["predictions"].keys() != new["predictions"].keys():
        raise ValueError("两个版本的确认行无法配对")
    # 确认的输入时点可从设备编号和运行时钟确定，截取时不使用寿命标签。
    units = defaultdict(list)
    for sid in base["predictions"]:
        row = state["samples"][sid]
        if row["role"] != "adopt":
            raise ValueError("混入其他用途的样本")
        units[row["unit_id"]].append(sid)
    endpoint = {u:[i for i in ids if state["samples"][i]["time_end"] == spec["cutoff_cycle"]]
                for u, ids in units.items()}
    threshold = state["config"]["min_improvement"]

    def measure(ids):
        before, after = [math.sqrt(math.fsum((v["predictions"][sid]-state["samples"][sid]["label"])**2
                                             for sid in ids)/len(ids)) for v in evaluations]
        return {"rows":len(ids), "original_rmse":before, "candidate_rmse":after,
                "selected":model_ids[int(before-after > threshold+1e-12)]}

    all_rows = measure(list(base["predictions"]))
    if (all_rows["selected"] != decision["selected_model"] or not math.isclose(all_rows["original_rmse"],base["score"])
            or not math.isclose(all_rows["candidate_rmse"], new["score"])):
        raise ValueError("未复现原确认成绩和决定")
    sufficient = (len(units)==35 and {u.split(":")[0] for u in units}=={"FD001","FD002","FD003","FD004"}
                  and all(len(ids)==1 for ids in endpoint.values()))
    report = {"protocol":sha256(spec_path.read_bytes()).hexdigest(),"source":spec["sha256"],
              "cycle":spec["cutoff_cycle"],"devices":len(units),"coverage_sufficient":sufficient,
              "all_rows":all_rows,"threshold":threshold}
    if sufficient:
        ids = sorted(v[0] for v in endpoint.values())
        report.update(endpoint=measure(ids), endpoint_ids=ids)
        report["changed"] = report["endpoint"]["selected"] != all_rows["selected"]
    atomic_json(ROOT/"results/experiments/official-engine/fit-comparison/开发截点确认.json",report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(PACKAGES))
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--compare-conditions", action="store_true", help="用发动机两张选样清单的既有预测复核工况分流")
    checks.add_argument("--compare-cutoff", action="store_true", help="从发动机原确认记录复算两种换版决定")
    args = parser.parse_args()
    if args.compare_conditions or args.compare_cutoff:
        if args.task != "engine":
            parser.error("工况核对使用发动机成对维护记录")
        if args.compare_cutoff:
            compare_confirmation_cutoff()
        else:
            compare_engine_conditions()
    else:
        audit(args.task)
