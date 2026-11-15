"""用首轮已保存的预测与官方气体包，复核提案输入和真实训练器会读到的资料。"""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.contracts import ActionPlan, digest
from harness.data.official_gas import LABELS, exploration_jsonl, read_official_gas
from harness.evaluation.gas_pilot import partition
from harness.models.gas_fit_queries import GasQueryReplayPlanner, GasQueryTools, RemovalQuery, execute_query_job, observation_key
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import OfficialGasTools
from harness.runtime.execution import LocalJobRunner
from harness.runtime.storage import atomic_json


def diagnosis(task, fit, explore, predictions):
    """只使用作决定时可见的训练资料和已回标探索行，按作者的前后测量摘要构造提案输入。"""
    import numpy as np
    if set(predictions) != {r.sample_id for r in explore} or any(r.role != "explore" for r in explore):
        raise ValueError("原版预测与本批合法探索编号不一致")
    before, now = [np.asarray([r.values for r in rows]) for rows in (fit, explore)]
    columns = {}
    for i, feature in enumerate(task.features):
        columns[feature] = [np.quantile(matrix[:, i], [0, .25, .5, .75, 1]).tolist() for matrix in (before, now)]
    errors = Counter((r.label, predictions[r.sample_id]) for r in explore if predictions[r.sample_id] != r.label)
    return {"summary_order": ["official_fit", "current_exploration"], "quantiles": [0, .25, .5, .75, 1],
            "measurements": columns, "observed_errors": [
                {"actual": LABELS[a-1], "predicted": LABELS[b-1], "count": n} for (a,b), n in sorted(errors.items())],
            "permitted_action": {"feature": "one measurement name", "op": ["lt","le","gt","ge"],
                                 "threshold": "finite number", "meaning": "exclude matching official training rows"}}


def response_condition(packet, response):
    """从一次完整对话回复取出单个实验建议。确认样本、父版和可执行字段由本地绑定。"""
    from harness.integrations.pytrio_planner import _invalid_constant, _strict_object
    choices = response["choices"]
    if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
        raise ValueError("需要一条完整的提案回复")
    content = choices[0]["message"]["content"]
    if not isinstance(content,str) or len(content)>65536:
        raise ValueError("提案内容超过接口范围")
    reply = json.loads(content,object_pairs_hook=_strict_object,parse_constant=_invalid_constant)
    if (set(reply) != {"observation_key","parent_sha256","reason","condition"} or
            any(reply[k] != packet[k] for k in ("observation_key","parent_sha256")) or
            not isinstance(reply["reason"],str) or not reply["reason"].strip()):
        raise ValueError("回复须绑定已发送的观察与父权重，并给出试改理由")
    condition = reply["condition"]
    return (None if condition is None else RemovalQuery(**condition)), reply["reason"]


def inspect(query=None, *, diagnosis_output=None, response=None, messages_output=None):
    path = ROOT / "data/official/gas/cpu-a-guide-20261110-v1.zip"
    run = ROOT / "results/experiments/official-gas/pilot-11"
    lock = json.loads((run / "protocol.json").read_bytes())
    package = read_official_gas(path)
    batches, _ = partition(package, lock["protocol"])
    batch = batches[0]
    explore = tuple(r for r in batch.samples if r.role == "explore")
    if (package.train_sha256 != lock["training"] or package.validation_sha256 != lock["validation"]
            or [r.sample_id for r in explore] != lock["rounds"][batch.batch_id]["explore"]):
        raise ValueError("原包或首轮编号与冻结协议不符")
    state = json.loads((run / "sequence-first-window/state.json").read_bytes())
    current = state["completed"][batch.batch_id]["decision"]["current_model"]
    baseline = state["operations"][f"{batch.batch_id}:explore:{current}"]["value"]
    packet = diagnosis(package.task, package.train, explore, baseline["predictions"])
    packet.update(batch=batch.batch_id, current_model=current, observation_key=observation_key(package, batch, baseline),
                  parent_sha256=package.checkpoint_sha256)
    if diagnosis_output:
        atomic_json(diagnosis_output, packet)
    if messages_output:
        atomic_json(messages_output, [{"role":"system","content":
            "请根据训练与当前探索摘要、已见错判，给出一项可检验的剔除条件。"
            "仅返回 JSON 对象，包括 observation_key、parent_sha256（复制本次输入的值）、reason、"
            "condition。condition 仅包含 feature、op、threshold，符合 permitted_action；依据不足可填 null 并说明维持理由。"},
            {"role":"user","content":json.dumps(packet,ensure_ascii=False,allow_nan=False)}])
    reason = None
    if response is not None:
        if query is not None:
            raise ValueError("不能同时接受手工条件和服务回复")
        query, reason = response_condition(packet,response)
        if query is None:
            source = "imported chat response/" + digest(response)
            return {"scope":"recorded keep decision; no fit, inference or driver call", "batch":batch.batch_id,
                    "proposal_reason":reason, "gas_fit_queries":{batch.batch_id:{
                    "source":source,"observation_key":packet["observation_key"],
                    "parent_sha256":packet["parent_sha256"],"conditions":[]}}}
    # 默认条件取自当轮已见的一项测量上界，仅用于端口验收；不作为研究提案或推荐的阈值。
    feature = package.task.features[0]
    example = RemovalQuery(feature, "gt", packet["measurements"][feature][1][-1]) if query is None else query
    with TemporaryDirectory() as temp:
        work = Path(temp)
        artifacts = ModelArtifacts(work / "models")
        runner = LocalJobRunner(work / "jobs", artifacts)
        dto = asdict(example)
        source = "illustrative boundary from current exploration" if query is None else "supplied condition"
        if response is not None:
            source = "imported chat response/" + digest(response)
        proposals = {batch.batch_id: {"source": source, "observation_key": packet["observation_key"],
                                     "parent_sha256": packet["parent_sha256"], "conditions": [dto]}}
        tools = GasQueryTools(path, package, batches, runner, proposals)
        catalog = tools.fit_manifest_context(batch, baseline)
        context = next(event["context"] for event in state["events"] if event["key"] == f"{batch.batch_id}:plan_context:0")
        response = GasQueryReplayPlanner().propose({**context, "fit_manifests": catalog})
        if response["stop"]:
            raise ValueError("条件未形成非空且不同于原版的候选清单")
        action = ActionPlan.parse(response["actions"][0])
        rows = tools.select_fit_rows(action.fit_manifest, package.train)
        initial = tools._build_request(package.task, batch.batch_id, ActionPlan("recent"), package.train, True, None)
        parent = artifacts.create(initial, lambda: package.checkpoint)
        req = tools._build_request(package.task, batch.batch_id, action, rows, False, parent)
        legacy = OfficialGasTools(path, package, batches, runner)
        try:
            legacy._build_request(package.task, batch.batch_id, action, rows, False, parent)
        except ValueError as exc:
            legacy_result = str(exc)
        else:
            raise ValueError("旧版已接受此清单，需重审对照")
        receipt = execute_query_job(work, {"request": req, "backend": tools.worker,
            "rows": exploration_jsonl(explore).decode()}, artifacts, inspect_only=True)
        if receipt["parent_checkpoint_sha256"] != package.checkpoint_sha256:
            raise ValueError("输入核验使用的父版本与原版不符")
    confirmation = Counter(row.label for row in batch.samples if row.role == "adopt")
    result = {"scope": "official reader only; no fit, inference or driver call", "batch": batch.batch_id,
            "proposal_origin": source, "gas_fit_queries": proposals,
            "stored_incumbent": current, "diagnosis_digest": digest(packet), "observed_errors": packet["observed_errors"],
            "feature_before_after": packet["measurements"][example.feature],
            "legacy_result": legacy_result, "class_counts_before_after": catalog["training_class_counts_before_after"][example.name],
            "confirmation_rows_kept_separate": len(lock["rounds"][batch.batch_id]["adopt"]),
            "confirmation_pool_by_class_after_inspection": {label: confirmation[i] for i, label in enumerate(LABELS, 1)},
            "actual_official_reader": receipt}
    if reason is not None:
        result["proposal_reason"] = reason
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--query", help="可选的单条结构化剔除条件，JSON 字符串")
    source.add_argument("--response", type=Path, help="导入与观察和父权重绑定的 chat/completions 回复 JSON")
    parser.add_argument("--messages-output", type=Path, help="导出供同一观察使用的 messages，不发送请求")
    parser.add_argument("--diagnosis-output", type=Path, help="可选，导出合法的前后测量摘要供提出条件")
    parser.add_argument("--output", type=Path, help="可选，保存只读回执")
    args = parser.parse_args()
    supplied = RemovalQuery(**json.loads(args.query)) if args.query is not None else None
    result = inspect(supplied, diagnosis_output=args.diagnosis_output, messages_output=args.messages_output,
                     response=json.loads(args.response.read_bytes()) if args.response is not None else None)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
