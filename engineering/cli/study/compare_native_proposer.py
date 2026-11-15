"""在真实交付入口上仅替换 propose，根据外部评分数据的原始答案复核交出的预测。"""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from math import fsum, sqrt
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT / "engineering/src")]
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from run_unseen_agents import ExternalReplay
from verify_unseen_scores import remaining_cycles, remaining_minutes
from harness.contracts import digest
from harness.runtime.storage import atomic_json

OUTPUT = ROOT/"results/experiments/proposer-swap"


def run(task, arm):
    workspace = OUTPUT/task/arm
    driver = ({"mode":"rule"} if arm=="original" else
              {"mode":"local-proposal","simple_proposal":True,"inbox":OUTPUT/"inbox"})
    engine, official = assemble_public(task,ROOT/"data/official",workspace,"rule",driver)
    with ExitStack() as stack:
        external = ExternalReplay(task,stack)
        # 外部测量只交给诊断、确认及预测。固定训练清单、官方工作者、训练日志使用的
        # 原包诊断文件及其轮次 ID 均来自未修改的公开装配入口。
        batches = tuple(replace(base,samples=tuple(s for s in base.samples if s.role=="fit")+
                               tuple(replace(s,values=tuple(map(float,s.values))) for s in ex.samples))
                        for base,ex in zip(official,external.batches,strict=True))
        engine.tools.values = {s.sample_id:s.values for b in batches for s in b.samples if s.role!="fit"}
        signature = {"task":task,"model":engine.tools.source,"config":asdict(engine.config),
                     "tools_identity":engine.tools.identity,"fit_ids":digest(sorted(engine.tools.fit)),
                     "input_digests":[digest([asdict(s) for s in b.samples]) for b in batches],
                     "service_ids":digest(sorted(external.truth))}
        fixed=workspace/"controlled-inputs.json"
        if fixed.exists():
            # 采用 JSON 摘要，元组恢复成列表后仍可核对。
            if digest(load(fixed))!=digest(signature):
                raise ValueError("恢复时的训练配置或测试流与冻结记录不同")
        else:
            atomic_json(fixed,signature)
    return run_prepared(task,workspace,"rule",10,driver,engine,batches)


def load(path):
    return json.loads(path.read_bytes())


def score(task):
    truth = (remaining_cycles if task=="engine" else remaining_minutes)()
    streams = {}; summary = {}
    base = OUTPUT/task
    if load(base/"original/controlled-inputs.json") != load(base/"simple/controlled-inputs.json"):
        raise ValueError("训练、控制器配置或轮次输入有区别")
    for arm in ("original","simple"):
        root=base/arm
        report=load(root/"run.json")
        state=load(root/"state.json")
        if len(report["rounds"])!=10:raise ValueError("十轮服务交接不齐")
        if any(v["status"]!="completed" for v in state["operations"].values()):
            raise ValueError("有未完成的操作")
        if any(e.get("kind")=="plan_invalid" for e in state["events"]):
            raise ValueError("提案未通过控制器检查")
        predictions={}; by_device=defaultdict(list); scores=[]; details=[]
        for i,row in enumerate(report["rounds"],1):
            payload=load(root/row["submitted"])["payload"]
            p=payload["predictions"]
            if predictions.keys() & p.keys():raise ValueError("服务 ID 跨轮重复")
            reference=load(ROOT/f"results/experiments/unseen-agents/issued/{task}/static/{i:02d}.json")["predictions"]
            if p.keys()!=reference.keys():raise ValueError("服务切片与原实验不符")
            bundle=root/"models"/row["model_artifact"]["key"]
            manifest=load(bundle/"manifest.json")
            if sha256((bundle/"pipeline.bin").read_bytes()).hexdigest()!=manifest["sha256"]:
                raise ValueError("实际交接权重摘要错误")
            predictions.update(p)
            errors=[]
            for sid,v in p.items():
                e=(v-truth[sid])**2
                errors.append(e);by_device[sid.rsplit("/",1)[0]].append(e)
            scores.append(sqrt(fsum(errors)/len(errors)))
            details.append({"round":i,"rmse":scores[-1],"decision":row["outcome"],"changed":row["current"]!=row["selected"],
                            "proposals":row.get("planned_actions",[]),"reason":row.get("reason"),
                            "weights_sha256":manifest["sha256"],"trials":len(row["trials"]),
                            "confirmation":row.get("confirmation")})
        if arm=="simple":
            exchanges=load(root/"driver-exchanges.json")["exchanges"]
            if len(exchanges)!=10 or any(e["status"]!="completed" for e in exchanges):
                raise ValueError("简单提案器的十轮交互不齐全")
        streams[arm]=predictions
        summary[arm]={"rmse":sqrt(fsum(fsum(v) for v in by_device.values())/len(predictions)),
                      "rows":len(predictions),"devices":len(by_device),"by_device":{k:sqrt(fsum(v)/len(v)) for k,v in by_device.items()},
                      "worst_round":max(scores),"rounds":details,
                      "proposed_rounds":[r["round"] for r in details if r["proposals"]],
                      "accepted_rounds":[r["round"] for r in details if r["changed"]]}
    a,b=streams.values()
    if a.keys()!=b.keys():raise ValueError("配对样本不一致")
    summary["changed_predictions"]=sum(a[k]!=b[k] for k in a)
    summary["different_weights_rounds"]=[x["round"] for x,y in zip(summary["original"]["rounds"],summary["simple"]["rounds"],strict=True)
                                         if x["weights_sha256"]!=y["weights_sha256"]]
    summary["different_proposal_rounds"]=[x["round"] for x,y in zip(summary["original"]["rounds"],summary["simple"]["rounds"],strict=True)
                                          if bool(x["proposals"])!=bool(y["proposals"])]
    atomic_json(base/"summary.json",summary)
    print(task, {k:round(summary[k]["rmse"],5) for k in streams},summary["changed_predictions"])


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("engine","bearing"))
    parser.add_argument("mode",choices=("original","simple","score"))
    args=parser.parse_args()
    if args.mode=="score":score(args.task)
    else:run(args.task,args.mode)
