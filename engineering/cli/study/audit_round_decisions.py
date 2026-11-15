"""把公开气体、轴承固定与自主组的冻结状态转为逐轮可读对照，只读取历史记录。"""
import argparse
import json
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.evaluation.round_evidence import round_evidence

GASES = ("乙醇", "乙烯", "氨", "乙醛", "丙酮", "甲苯")
OUTCOMES = {"no_trial": "跳过试改", "exploration_kept_incumbent": "探索未胜出，保持",
            "confirmation_rejected": "确认后拒绝", "adopted": "确认后采用",
            "confirmation_unavailable": "缺少共同确认，保持", "output_fallback": "输出回退",
            "no_completed_challenger": "未取得完整候选"}


def describe_observation(state, batch, row):
    obs = row["observation"]
    if "confusions" in obs:
        worst = obs["lowest_f1_class"]
        name = GASES[worst-1] if worst is not None else "无错判"
        pairs = "、".join(f"{GASES[e['target']-1]}误报{GASES[e['reported']-1]}×{e['count']}"
                         for e in obs["confusions"])
        return f"最低类别 F1：{name}；{pairs or '无错判'}"
    # 原版当场已用这些探索答案。此处只还原单根轴承的触发依据，不使用服务答案。
    scored = state["operations"][f"{batch}:explore:{row['current']}"]["value"]
    samples = sorted((state["samples"][i] for i in scored["sample_ids"]), key=lambda s: s["time_end"])
    if len({s["unit_id"] for s in samples}) != 1:
        raise ValueError("该表的轴承段只解释单根探索设备")
    recent = samples[len(samples)//2:]
    errors = [scored["predictions"][s["sample_id"]] - s["label"] for s in recent]
    return f"近{len(errors)}次，{sum(e>0 for e in errors)}次高估、{sum(e<0 for e in errors)}次低估，平均偏差{fmean(errors):+.1f}分钟"


def describe_decision(row, arm, rule_driven):
    parts = []
    for action in row["planned_actions"]:
        pair = action.get("gas_pair")
        parts.append(f"复查报为{GASES[pair['reported']-1]}、目标{GASES[pair['target']-1]}" if pair
                     else f"续训{action['epochs']}遍")
    parts.append(OUTCOMES[row["outcome"]])
    if row["outcome"] == "no_trial" and rule_driven:
        obs = row["observation"]
        if arm == "harness" and obs.get("device_bias_issue") is None:
            parts.append("近期同向误差未达到触发条件")
        if arm in ("sequence-first-window", "sequence-protected"):
            active = {(p["reported"], p["target"]) for p in obs["active_pairs"]}
            if len(active) >= 2:
                parts.append("两项修正已生效")
            elif arm == "sequence-protected" and not any(
                    (e["reported"], e["target"]) not in active and e["count"] >= 2 for e in obs["confusions"]):
                parts.append("未处理方向均未重复两次")
    scores = row["confirmation"]
    if scores[1] is not None:
        parts.append(f"确认 {scores[0]:.5g} → {scores[1]:.5g}")
        scope = row.get("confirmation_scope")
        if scope:
            before = "、".join(scope["exploration_sources"])
            after = "、".join(scope["confirmation_sources"])
            parts.append(f"探索来源 {before}，确认来源 {after}，重合样本 {scope['overlapping_sample_count']} 条")
    elif row["trials"] and row["trials"][0]["exploration_score"] is not None:
        parts.append(f"探索 {row['observation']['exploration_score']:.5g} → {row['trials'][0]['exploration_score']:.5g}")
    return "；".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/experiments/public-runtime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = ["# 公开回放逐轮维护决定", "", "左栏为出计划之前在用模型的观察。两组使用各自当轮的模型，观察相同则合并显示。分数顺序为在用版、候选版。静态组始终使用原版。", ""]
    for task, title, names, state_path in (("gas", "气体 · 类别 Macro-F1 越高越好",
                      ("sequence-first-window", "sequence-protected"), "state.json"),
                      ("bearing", "轴承 · 寿命 RMSE（分钟）越低越好",
                      ("fixed_confirmed", "harness"), "workflow/state.json")):
        states = [json.loads((args.input/task/a/state_path).read_bytes()) for a in names]
        rule_driven = json.loads((args.input/task/"run-definition.json").read_bytes())["driver"] in ("rule","mock")
        lines.extend([f"## {title}", "", "| 轮次 | 当时观察 | 固定组的动作与去留 | 自主组的动作与去留 |",
                      "| --- | --- | --- | --- |"])
        for i in range(1, 11):
            bid = f"round-{i}"
            rows = [round_evidence(s, bid) for s in states]
            observed = [describe_observation(s, bid, r) for s, r in zip(states, rows)]
            facts = observed[0] if observed[0] == observed[1] else f"固定：{observed[0]}；自主：{observed[1]}"
            lines.append(f"| {i} | {facts} | {describe_decision(rows[0], names[0], True)} | {describe_decision(rows[1], names[1], rule_driven)} |")
        lines.append("")
    lines.append("气体固定组沿首轮排定的顺序尝试，自主组从尚未处理且本轮至少重复两次的方向中选择，最多保留两项。轴承固定组按日程续训，自主组仅在近期误差至少四分之三同向时试改。服务答案不参与这些选择。")
    lines.extend(["", "## 另行开展的首次寿命维护", "",
                  "以下为早先的一轮设备对照，取样协议与上述十轮不同。", ""])
    for task, title in (("bearing","轴承"), ("engine","发动机")):
        folder = ROOT / f"results/experiments/official-{task}/maintenance-1"
        state = json.loads((folder/"workflow/state.json").read_bytes())
        row = round_evidence(state, "maintenance-1")
        observations = json.loads((folder/"observation.json").read_bytes())["devices"]
        device = row["observation"].get("device_bias_issue") or observations[0]
        confirmed = json.loads((folder/"selected.json").read_bytes())["confirmation"]
        lines.append(f"- {title}：{device['unit']} 最近 {device['recent_count']} 次平均偏差 {device['recent_bias']:+.2f}，"
                     f"同向 {device['matching']} 次。固定组按日程试后确认，误差 {confirmed['original']['rmse']:.2f} → "
                     f"{confirmed['candidate']['rmse']:.2f}；自主组{describe_decision(row, 'harness', True)}。")
    args.output.write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
