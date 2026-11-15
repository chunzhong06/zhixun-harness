"""从指定驱动或离线服务取得一项首轮建议，用正式气体后端执行并评价生效版本。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from harness.contracts import EngineConfig, digest
from harness.data.gas_input import read_gas_batches, REGISTERED_SHA256
from harness.data.official_gas import LABELS, read_official_gas
from harness.evaluation.gas_pilot import partition
from harness.evaluation.metrics import macro_f1
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.integrations.service_transport import HttpChatTransport
from harness.methods.planning.gas_proposal import ExampleProposal, FirstProposalPlanner, obtain_proposal
from harness.models.gas_fit_queries import GasQueryTools, observation_key
from harness.models.gas_views import pair_model
from harness.models.official_gas_cpu import OfficialGasTools, PACKAGE_SHA
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.engine import Harness
from harness.runtime.execution import ExecutionLimits, LocalJobRunner
from harness.runtime.storage import LocalStore, atomic_json

ROOT = Path(__file__).resolve().parents[3]
GAS = ROOT/"results/experiments/official-gas"
CONFIG = ROOT/"engineering/configs/experiments/official-gas-first-proposal.json"
PACKAGE = ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip"
PACKET = ROOT/"results/inspections/gas/首轮共同提案输入.json"
SPLIT = GAS/"pilot-11/protocol.json"
REFERENCES = GAS/"module-smoke/measured/issued.jsonl"


def predict(model, rows):
    values = np.asarray(rows)
    return np.concatenate([model.predict(values[start:start+128]).argmax(axis=1)+1
                           for start in range(0,len(values),128)])


def load_inputs(protocol, model_type, work):
    for path, expected in [(PACKAGE,PACKAGE_SHA),(PACKET,protocol["observation_sha256"]),
                           (SPLIT,protocol["split_sha256"]),(REFERENCES,protocol["comparison_predictions_sha256"])]:
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"冻结输入发生变化：{path.name}")
    package = read_official_gas(PACKAGE)
    saved = json.loads(SPLIT.read_bytes())
    batches,_ = partition(package,saved["protocol"])
    batch = batches[0]
    rows = [r for r in batch.samples if r.role=="explore"]
    path = work/"base.npz"
    path.write_bytes(package.checkpoint)
    predictions = predict(model_type.load(path),[r.values for r in rows])
    baseline = {"model_id":batch.batch_id+"/initialize",
                "predictions":{r.sample_id:int(p) for r,p in zip(rows,predictions,strict=True)}}
    packet = json.loads(PACKET.read_bytes())
    if (packet["observation_key"]!=observation_key(package,batch,baseline)
            or packet["parent_sha256"]!=package.checkpoint_sha256):
        raise ValueError("已发布观察与当时权重的实际预测不符")
    packet["permitted_action"] = {"tools":protocol["tools"],"candidate_limit":protocol["candidate_budget"],
        "filter":"exclude matching rows of the original training set, then fit one epoch",
        "patch":"use frozen shape-blend on an observed reported-to-actual class pair",
        "keep":"serve current version","classes":list(LABELS)}
    return package,batches,packet


def execute(output, package, batch, reply, source, config):
    artifacts = ModelArtifacts(output/"models")
    limits = ExecutionLimits(30,20,536870912)
    runner = LocalJobRunner(output/"jobs",artifacts,limits)
    action = reply["action"]
    if action["kind"]=="filter":
        condition = {k:action[k] for k in ("feature","op","threshold")}
        record = {"source":source,"conditions":[condition],"observation_key":reply["observation_key"],
                  "parent_sha256":reply["parent_sha256"]}
        tools = GasQueryTools(PACKAGE,package,(batch,),runner,{batch.batch_id:record})
    else:
        tools = OfficialGasTools(PACKAGE,package,(batch,),runner,pair_method="shape-blend")
    settings = EngineConfig(execution_mode="real_model",recent_cap=len(package.train),max_trials=config["candidate_budget"],
                            policy="plan_feedback",maintenance_trigger="every_batch",
                            worker_cpu_seconds=limits.cpu_seconds,worker_memory_bytes=limits.memory_bytes)
    state = Harness(package.task,settings,tools,FirstProposalPlanner(reply),LocalStore(output),
                    allow_real_execution=True).run((batch,))
    done = state["completed"][batch.batch_id]
    if any(op["status"]=="failed" for op in state["operations"].values()):
        raise RuntimeError("受控作业失败，不能将失败作业解释为候选无效")
    trial = done["trials"][0] if done["trials"] else None
    if trial and not trial.get("model_id"):
        raise RuntimeError("候选未完成，详见运行回执")
    decision = done["decision"]
    ids = {"incumbent":decision["current_model"],"served":done["model_id"]}
    if trial:
        ids["candidate"] = trial["model_id"]
    costs = [json.loads(p.read_bytes()) for p in (output/"jobs").glob("*/state.json")]
    info = {"tool":action, "reason":reply["reason"], "model_ids":ids, "decision":decision["reason"],
            "explore_score":trial["exploration_score"] if trial else None,
            "confirmation":[decision.get("baseline_score"),decision.get("candidate_score")],
            "training_rows":len(trial["training_ids"]) if trial else 0,
            "dispositions":tools.dispositions[batch.batch_id] if action["kind"]=="filter" else [],
            "controlled_job_cpu_s":sum(j["cpu_seconds"] for j in costs)}
    return {role:artifacts.read(state["models"][mid]["artifact"])[1] for role,mid in ids.items()},info


def evaluate(output, following, weights, model_type, work):
    rows = [r for r in following.samples if r.role=="explore"]
    batches = {str(b.batch_number):b.records for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",
               expected_sha256=REGISTERED_SHA256) if 3<=b.batch_number<=8}
    features = [r.values for r in rows]
    keys = [("next",r.sample_id) for r in rows]
    for batch, records in batches.items():
        features.extend(r.values for r in records)
        keys.extend((batch,r.source_line) for r in records)
    references = [json.loads(line) for line in REFERENCES.read_bytes().splitlines()]
    if [(r["batch"],r["id"]) for r in references] != keys:
        raise ValueError("后续段或采集批与冻结参照编号不一致")
    predictions = {k:np.array([r[k] for r in references]) for k in ("static","shape-blend","filter-1")}
    for role,content in weights.items():
        path = work/(role+".npz")
        path.write_bytes(content)
        predictions[role] = predict(model_type.load(path),features)
    if not np.array_equal(predictions["static"],predictions["incumbent"]):
        raise ValueError("起点未复现已有原版预测")
    issued = output/"issued.jsonl"
    # 已选模型在下一段生效。另六批的结果属于冻结版本的跨批评价。
    with (work/"issued.jsonl").open("w",encoding="utf8") as stream:
        for i,(batch,sid) in enumerate(keys):
            row = {"batch":batch,"id":sid,**{k:int(p[i]) for k,p in predictions.items()}}
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    os.replace(work/"issued.jsonl",issued)
    truths = {"next":[r.label for r in rows],**{b:[r.label for r in rs] for b,rs in batches.items()}}
    stages = {}
    offset = 0
    for batch,labels in truths.items():
        y = np.array(labels)
        values = {k:v[offset:offset+len(y)] for k,v in predictions.items()}
        base = values["static"]
        common_errors = (base!=y)&(values["shape-blend"]!=y)&(values["filter-1"]!=y)
        stages[batch] = {k:{"macro_f1":macro_f1(zip(labels,p.tolist(),strict=True)),
                            "repaired":int(np.sum((base!=y)&(p==y))),"broken":int(np.sum((base==y)&(p!=y))),
                            "common_fixed":int(np.sum(common_errors&(p==y)))} for k,p in values.items()}
        offset += len(y)
    return {"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"stages":stages,
            "six_batch_mean":{k:float(np.mean([stages[b][k]["macro_f1"] for b in batches])) for k in predictions}}


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--enable-real-execution",action="store_true",required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline-example",choices=["filter","patch","alternate-patch","keep"])
    group.add_argument("--enable-service-network",action="store_true")
    parser.add_argument("--model-config",type=Path)
    args = parser.parse_args(arguments)
    protocol = json.loads(CONFIG.read_bytes())
    if bool(args.model_config)!=args.enable_service_network:
        parser.error("实际调用须提供比赛模型配置；离线样例使用自己的模拟配置")
    if sha256(PACKAGE.read_bytes()).hexdigest()!=PACKAGE_SHA:
        raise ValueError("官方模型包发生变化")
    sys.path.insert(0,str(PACKAGE))
    from baselines.model import Model
    model_type = pair_model(Model)
    if args.enable_service_network:
        config = PyTrioConfig(**json.loads(args.model_config.read_bytes()))
        if config.base_url.rstrip("/")!=PyTrioConfig.base_url or config.max_tokens>protocol["max_output_tokens"]:
            raise ValueError("须使用已核对的比赛服务及冻结的输出预算")
        transport = HttpChatTransport(config,os.environ.get("PYTRIO_API_KEY",""),allow_network=True)
        source = "designated-service"
    else:
        from harness.testing.fixtures import SimulatedHttpService
        fixed_filter = protocol["example_condition"]
        fixture = ExampleProposal(args.offline_example,fixed_filter)
        config = PyTrioConfig("offline-gas-proposal")
        transport = HttpChatTransport(config,"fixture-only-key",wire=SimulatedHttpService(fixture))
        source = "simulated/"+args.offline_example
    output = args.output.resolve()
    with LocalStore(output/"proposal-lock").exclusive(), TemporaryDirectory(dir=output) as temp:
        work = Path(temp)
        package,batches,packet = load_inputs(protocol,model_type,work)
        reply,receipt = obtain_proposal(output,packet,config,transport,source)
        weights,info = execute(output,package,batches[0],reply,source,protocol)
        result = {"protocol_sha256":sha256(CONFIG.read_bytes()).hexdigest(),"proposal_source":source,
                  "request_id":receipt["request_id"],"usage":receipt["usage"],"maintenance":info,
                  "weight_hashes":{k:sha256(v).hexdigest() for k,v in weights.items()},
                  **evaluate(output,batches[1],weights,model_type,work)}
        atomic_json(output/"comparison.json",result)
        print(output/"comparison.json")
