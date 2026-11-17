"""从已保存的运行状态生成维护流程说明。"""


def render_workflow(state: dict) -> str:
    description=("本页记录公共程序的工程衔接。计划与预测使用预先编写的样例；未训练模型、未调用Qwen，数值不是比赛成绩。"
                 if state["evidence_kind"]=="interface_fixture" else "本页按实际执行记录生成，数据、模型、决定和费用可按事件身份回查。")
    lines = ["# 维护流程记录", "", description, "",
             f"运行方式：{state['config']['policy']}", "", "| 批次 | 新增/重发/修订 | 实际执行的候选 | 决定 | 下一批使用版本 |", "| --- | --- | --- | --- | --- |"]
    for batch_id, b in state["completed"].items():
        counts = b["received"]
        descriptions = []
        for trial in b["trials"]:
            action = trial.get("action_plan")
            if action and action.get("pipeline"):
                spec = action["pipeline"]
                processor = {"none": "保留原量纲", "standard": "标准化", "robust": "按中位数和分位距缩放"}[spec["processor"]]
                features = "剔除恒定特征" if spec["features"] == "nonconstant" else "保留全部特征"
                predictor = "树模型" if spec["predictor"] == "forest" else "线性模型"
                descriptions.append(f"{processor}、{features}、{predictor}（{trial['status']}）")
            else:
                descriptions.append(f"{trial['action']}（{trial['status']}）")
        trials = "；".join(descriptions) or "保持检查"
        decision = b["decision"]
        lines.append(f"| {batch_id} | {counts['new']}/{counts['replayed']}/{counts['revised']} | {trials} | {decision['reason']} | {b['model_id']} |")
    contexts = {e["key"]: e["context"] for e in state["events"] if e["kind"] == "plan_context"}
    accepted = [e for e in state["events"] if e["kind"] == "plan_accepted"]
    if accepted:
        lines += ["", "## 提出改法时看到了什么", "",
                  "以下按当时保存的计划快照展示；事实编号可在state.json的planning_evidence中找到样本修订或操作来源。", ""]
        for event in accepted:
            context = contexts[event["context_key"]]
            facts = {item["kind"]: item for item in context["facts"]["evidence"]}
            performance = facts["performance"]["value"]
            cost = facts["resources"]["value"]
            lines.append(f"**{event['context_key']}**：模型{performance['model_id']}在本批{performance['count']}条探索资料上的{performance['metric']}为{performance['score']}；"
                         f"当时已用{cost['current_batch']['demo_units']}个工程点数，剩余{cost['remaining_demo_units']}个。")
            lines.append("")
            cards = context.get("method_guidance", {}).get("cards", [])
            if cards:
                sources = "；".join(f"{card['title']}（{card['id']}，[来源]({card['source']['url']})）" for card in cards)
                lines.append(f"本次已提供的方法依据：{sources}。具体适用条件和关联观察保存在同一计划快照。")
                lines.append("")
            for action in event["actions"]:
                reason = action["reason"].replace("\n", " ")
                lines.append(f"- {reason}。依据：{'、'.join(action['evidence_refs'])}。")
            lines.append("")
    if state.get("inspections"):
        lines += ["", "## 检查怎样影响改法", ""]
        for batch_id, checks in state["inspections"].items():
            for check in checks:
                outcome = ("资料不足，本次没有形成判断" if check["signal"] is None else
                           "达到本次设定的检查条件" if check["signal"] else "未达到本次设定的检查条件")
                lines.append(f"- {batch_id}：{check['question']}。实际检查：{outcome}；具体观察及两种后续改法见 {check['check_key']}。")
    if state.get("methods"):
        lines += ["", "## 改法怎样积累依据", "",
                  "下列对照使用相同拟合与探索资料；执行来源与各观察一起保存。", ""]
        for method in state["methods"].values():
            for observation in method["observations"]:
                result = (f"相对同资料重训对照的改善量为{observation['utility_gain']}" if
                          observation["comparison"] == "matched_retraining" else "本次保存候选结果，尚无同资料重训对照")
                lines.append(f"- {observation['batch_id']}，{method['id']}：{result}；"
                             f"{'采用该版本' if observation['adopted'] else '保留另一个版本'}。")
    work = [e for e in state["events"] if e["kind"] == "work_selected"]
    if work:
        labels = {"inspect": "补查证据", "experiment": "执行试改", "recheck": "复核旧方法", "finish": "结束本批尝试"}
        lines += ["", "## 试改之后怎样继续", ""]
        for entry in work:
            decision = entry["decision"]
            lines.append(f"- {entry['context_key']}：{labels[decision['operation']]}。依据：{decision['reason']}")
        lines += ["", "退出原因由控制器按实际发生条件记录：", ""]
        for bid, result in state.get("maintenance_exits", {}).items():
            lines.append(f"- {bid}：{result['reason']}；已完成工作轮次{result['rounds']}。")
    lines += ["", "逐步记录见events.jsonl；输入用途、样本清单、版本和执行账目见state.json。", "",
              "demo_units为工程测试额度；elapsed_seconds为接口执行耗时。SwanLab实际记录状态见summary.json及独立核验回执。", ""]
    return "\n".join(lines)
