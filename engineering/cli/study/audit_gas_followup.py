"""按来源依次揭晓已有气体预测。策略只收到其本批选中方案的反馈。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

from evaluate_gas_future import ROOT, read_gas_batches, repair_feedback, score
from harness.evaluation.metrics import macro_f1
from harness.runtime.storage import atomic_json

ACETALDEHYDE, ETHYLENE = 4, 2
TARGETS = (ACETALDEHYDE, ETHYLENE)
NAMES = {4: "acetaldehyde", 2: "ethylene"}
PROTOCOL = ROOT / "engineering/configs/experiments/gas-source-followup.json"
OUTPUT = ROOT / "results/experiments/official-gas/source-followup.json"


def checked(path, checksum):
    data = path.read_bytes()
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"来源文件与冻结版本不同：{path}")
    return data


def single_prediction(base, both, target):
    # 仅适用于这两条保护式规则：同一原判、一次查询、两个互斥出口。
    if both != base and (base != 1 or both not in TARGETS):
        raise ValueError("存在不能拆分的级联改判")
    return target if base == 1 and both == target else base


def opposite(target):
    return ETHYLENE if target == ACETALDEHYDE else ACETALDEHYDE


def next_choice(current, feedback):
    """这里没有另一项候选在上一批的预测，也没有下一批样本或答案。"""
    if feedback is None:
        return current, "feedback_unavailable"
    if (feedback["opportunities"] and feedback["fixed"] <= feedback["broken"]
            and feedback["other_target_opportunities"]):
        return opposite(current), "observed_no_net_fix_with_alternative_errors"
    return current, "retain_after_observed_feedback"


def replay(cfg):
    frozen_bytes = (ROOT / cfg["archive_protocol"]).read_bytes()
    frozen = json.loads(frozen_bytes)
    prior = json.loads(checked(ROOT / cfg["reference_scores"], cfg["reference_scores_sha256"]))
    if (prior["protocol_sha256"] != sha256(frozen_bytes).hexdigest()
            or prior["predictions_sha256"] != cfg["predictions_sha256"]):
        raise ValueError("预测、原对照和协议身份不一致")
    dev = json.loads(checked(ROOT / cfg["development"], cfg["development_sha256"]))
    for r in dev:
        if single_prediction(r["base"], r["double"], ETHYLENE) != r["ethylene"]:
            raise ValueError("拆分与开发批实际乙烯单项不符")
    held = [r for r in dev if r["primary"]]
    if len(held) != 414:
        raise ValueError("既定开发留出范围发生变化")
    dev_scores = {t: macro_f1((r["answer"], single_prediction(r["base"], r["double"], t)) for r in held) for t in TARGETS}
    first = max(TARGETS, key=dev_scores.get)
    predictions = [json.loads(line) for line in checked(ROOT/cfg["archive_predictions"], cfg["predictions_sha256"]).splitlines()]
    if set(r["batch"] for r in predictions) != {9, 10}:
        raise ValueError("后续来源应恰好覆盖两个已评价批")
    availability = {"initial_target": NAMES[first], "development_scores": {NAMES[t]: dev_scores[t] for t in TARGETS}}
    strategy, feedback = first, None
    rounds = []
    for index, batch_no in enumerate((9, 10)):
        strategy, reason = next_choice(strategy, feedback)
        choices = {"feedback": strategy, "fixed_acetaldehyde": ACETALDEHYDE, "fixed_ethylene": ETHYLENE,
                   "rotation": opposite(first) if index else first, "delayed_feedback": first}
        rows = [r for r in predictions if r["batch"] == batch_no]
        original = [r["base"] for r in rows]
        outputs = {name: [single_prediction(r["base"], r["double"], target) for r in rows]
                   for name, target in choices.items()}
        if outputs["fixed_acetaldehyde"] != [r["single"] for r in rows]:
            raise ValueError("拆分与实际发布的乙醛单项不符")
        # 当批决定和各策略发出的预测均已固定。下面才取该来源答案。
        source, = read_gas_batches(ROOT/frozen["source"], expected_sha256=frozen["source_sha256"], batch_number=batch_no)
        by_id = {s.source_line:s.label for s in source.records}
        ids = [r["line"] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != set(by_id):
            raise ValueError("评价所用的来源行有遗漏或重复")
        answers = [by_id[i] for i in ids]
        scored = score([dict(r, answer=y) for r, y in zip(rows, answers)])
        old = prior["batches"][str(batch_no)]
        if scored["models"] != old["models"] or scored["changes"] != old["changes"]:
            raise ValueError("原对照的逐批评分或纠错统计发生变化")
        metrics = {}
        for name, issued in {**outputs, "base": original, "both": [r["double"] for r in rows]}.items():
            metrics[name] = {"macro_f1": macro_f1(zip(answers, issued)),
                "correct": sum(p==y for p,y in zip(issued,answers)),
                "predictions_sha256": sha256(bytes(issued)).hexdigest()}
        prior_feedback = feedback
        feedback = repair_feedback(original, outputs["feedback"], answers, strategy)
        feedback["other_target_opportunities"] = sum(p == 1 and y == opposite(strategy) for p,y in zip(original,answers))
        rounds.append({"batch": batch_no, "rows": len(ids), "before": {"available_previous_selected_feedback": prior_feedback,
            "choice_reason": reason, "choices": {name:NAMES[t] for name,t in choices.items()}},
            "after": {"scores": metrics, "returned_selected_feedback": feedback,
                      "full_pair_diagnostics": scored["repair_feedback"],
                      "earlier_fixes_lost_when_adding_second": scored["changes"]["double"]["prior_fixes_lost"]}})
    averages = {name:fmean(r["after"]["scores"][name]["macro_f1"] for r in rounds)
                for name in rounds[0]["after"]["scores"]}
    return {"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"start":availability,
            "rounds":rounds,"equal_source_macro_f1":averages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    path = parser.parse_args().output
    report = replay(json.loads(PROTOCOL.read_bytes()))
    if path.exists():
        if json.loads(path.read_bytes()) != report:
            raise ValueError("已有对照与本次复算不符")
    else:
        atomic_json(path, report)
    print(json.dumps(report["equal_source_macro_f1"]))
