"""两个有界提案槽：仅改变发现阶段的历史可见性，随后复用原自由选择器。"""
from hashlib import sha256
import json
from pathlib import Path

from harness.adapters.providers.research_broker import ACTOR
from harness.adapters.storage.local import atomic_json
from harness.research.free_research_driver import (
    InvalidAction, ResearchControlError, _remaining_calls, _validate_action,
    _bind_model_call, _check_model_call, _check_response_model_call,
    _bind_executor_query_inputs,
    run_free_research as original_free_research,
)
from harness.research.research_ledger import history_entries


PROPOSAL_INSTRUCTION = """你是Harness的候选提案者。本次只处理一个有界提案槽，目标是发现能改善当前任务完整交付的候选。
依据本请求提供的合法任务、源码和执行资源，返回纯JSON，选择一种动作：
1. {"action":"experiment","hypothesis":"中文可检验机制","python_source":"完整单文件Python程序"}。通过任务的evaluate_model或submit_model交付真实可重载候选，允许保留有效部件；不指定你必须使用哪类模型。
2. {"action":"skip","reason":"没有值得执行的候选机制的具体理由"}。没有依据时可以跳过，不为消耗额度制造试验。
本槽不执行搜索或最终提名。有效experiment或skip均消耗这个提案槽；无效JSON最多连续纠正三次，所有请求计入共同模型预算。
程序只使用任务允许的输入与标签，不能读取其他目录、请求或外层答案。失败、超时与退步保留，不能伪造评价。
总执行预算包括初始化、研究、最终重训和逐行服务；等待模型不计执行时间。需在剩余额度内为发布交付留出时间。
"""


def _proposal(answer):
    def reject_constant(value):
        raise ValueError("JSON不允许非有限常量：" + value)
    if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
        raise InvalidAction("response_text_missing", "回复须包含字符串text")
    try:
        action = json.loads(answer["text"], parse_constant=reject_constant)
    except ValueError as error:
        raise InvalidAction("invalid_json", "回复不是有效JSON：" + str(error)) from error
    if not isinstance(action, dict):
        raise InvalidAction("action_not_object", "提案动作须为JSON对象")
    if action.get("action") == "skip":
        if not isinstance(action.get("reason"), str) or not action["reason"].strip():
            raise InvalidAction("skip_reason_missing", "skip须有具体理由")
        return action
    if action.get("action") != "experiment":
        raise InvalidAction("proposal_action_invalid", "提案槽只接受experiment或skip")
    return _validate_action(answer, [], {})[0]


def _programs(records):
    """只整理本臂已执行内容；确认和服务预测不进入提案或选择上下文。"""
    return [{"execution": record["sequence"], "phase": record["phase"],
             "source": Path(record["source"]).read_text(encoding="utf-8"),
             "stdout": record["stdout"], "stderr": record["stderr"], "returncode": record["returncode"]}
            for record in records]


async def run_branch_research(task_description, task, workspace, executor, broker, *, mode,
                              instruction=None, reference_sources=None, resume=False, model_call=None, **kwargs):
    """调用前须已在本臂预算内执行原三个初始化；两个模式共用解析及执行路径。

每个请求都是单条新用户上下文。independent 不携带上一槽或初始化结果，
本槽无效回复只用于本槽纠错。有效源码交给原独立执行器，不共享Python状态。
"""
    if mode not in ("shared_proposals", "independent_proposals"):
        raise ValueError("未知提案模式")
    _bind_executor_query_inputs(executor, task)
    model_call = _bind_model_call(workspace, broker, model_call)
    if not resume or len(executor.records) != 3 or any(
            record["actor"] != "harness_reference_initialization" for record in executor.records):
        raise ValueError("提案阶段要求原三个初始化记录，不自行补跑或复用旧运行")
    references = dict(reference_sources or {})
    if not references or executor.execution_budget is None:
        raise ValueError("提案需要共同参考及真实执行预算")
    work, events = Path(workspace) / "project", []
    ledger = Path(workspace) / "proposal-decisions.json"
    source_hashes = {name: sha256(code.encode()).hexdigest() for name, code in references.items()}
    atomic_json(Path(workspace) / "proposal-policy.json", {"mode": mode, "slots": 2,
        "max_consecutive_invalid_actions": 3, "instruction_sha256": sha256(PROPOSAL_INSTRUCTION.encode()).hexdigest(),
        "task_description_sha256": sha256(task_description.encode()).hexdigest(),
        "reference_sources": source_hashes, "selector": "original_run_free_research",
        "rankings_visible_to_proposer": mode == "shared_proposals"})

    def stop(code, message):
        details = {"action": "controller_stopped", "error_code": code, "error": message,
                   "model_calls": broker.calls, "remaining_model_invocations": _remaining_calls(broker)}
        events.append(details)
        atomic_json(ledger, events)
        raise ResearchControlError(code, message, details)

    for slot in (1, 2):
        invalid = []
        while True:
            if _remaining_calls(broker) <= 0:
                stop("model_budget_exhausted", "提案阶段模型预算耗尽，不自动提名")
            ACTOR.set("harness_proposal_" + str(slot))
            context = {"task_description": task_description,
                "available_references": [{"name": name, "source_sha256": source_hashes[name]} for name in references],
                "proposal_slot": slot, "proposal_slots": 2,
                "resources": {"remaining_model_invocations": _remaining_calls(broker),
                    "remaining_execution_seconds": executor.execution_budget.remaining_seconds,
                    "total_execution_seconds": executor.execution_budget.total_seconds,
                    "program_seconds": executor.timeout, "cpu_threads": 1,
                    "publication_and_service_counted_in_total": True},
                "action_feedback": invalid, "max_consecutive_invalid_actions": 3}
            if mode == "shared_proposals":
                context.update(history=history_entries(executor.records, events),
                               executed_programs=_programs(executor.records),
                               previous_proposals=[event for event in events if event.get("slot", slot) < slot])
            _check_model_call(broker, model_call, allow_unspecified=True)
            payload = {"request": {**model_call, "config": {"system_instruction": PROPOSAL_INSTRUCTION},
                "contents": [{"role": "user", "parts": [{"text": json.dumps(context, ensure_ascii=False)}]}]},
                "search_allowed": False}
            answer = await broker.exchange("model", payload)
            _check_response_model_call(answer, model_call)
            try:
                action = _proposal(answer)
            except InvalidAction as error:
                feedback = {"slot": slot, "action": "invalid_response", "error_code": error.code,
                    "error": str(error), "response_text": answer.get("text") if isinstance(answer, dict) else None,
                    "model_calls": broker.calls, "consecutive_invalid_actions": len(invalid) + 1}
                invalid.append(feedback)
                events.append(feedback)
                atomic_json(ledger, events)
                if len(invalid) >= 3:
                    stop("invalid_action_limit", "同一提案槽连续三次无效，保留原始回复后停止")
                continue
            event = {**action, "slot": slot, "model_calls": broker.calls,
                     "request_sequence": broker.sequence, "slot_consumed": True}
            events.append(event)
            atomic_json(ledger, events)
            if action["action"] == "experiment":
                before = len(executor.records)
                executor(action["python_source"], work, "candidate.py", executor.timeout)
                if len(executor.records) != before + 1:
                    raise RuntimeError("提案执行没有形成唯一真实执行回执")
                record = executor.records[-1]
                event.update(execution=record["sequence"], returncode=record["returncode"],
                             execution_seconds=record["execution_time"], source_sha256=record["source_sha256"])
                atomic_json(ledger, events)
            break

    # 原选择器默认只附前三份成功源码；显式附上两槽全记录，避免低分/失败提案正文丢失。
    discovery = {"proposal_events": events, "executed_programs": _programs(executor.records[3:]),
                 "history": history_entries(executor.records, events), "selection_reopened": False}
    atomic_json(Path(workspace) / "proposal-stage-complete.json", discovery)
    selector_task = task_description + "\n两个提案槽已结束；下列均为本臂真实记录，可提名任一有效候选或继续原自由研究：\n" + json.dumps(discovery, ensure_ascii=False)
    return await original_free_research(selector_task, task, workspace, executor, broker,
        instruction=instruction, reference_sources=references, resume=True, model_call=model_call or None, **kwargs)
