"""用已交接的官方模型和源训练参照，按返回的既往标签维护气体分类选择。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.data.gas_input import REGISTERED_SHA256,read_gas_batches
from harness.models import feedback_routing,gas_feedback_bank,vote_feedback,gas_vote_guard
from harness.models import observer_errors
from harness.models.feedback_routing import FeedbackRouter,apply_choices
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.models.official_rul_cpu import _trainer
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json


def load(path):
    return json.loads(path.read_bytes())


def serve_batch(router,bank,current,keys,x,mode):
    """当前服务输入只有编号和特征；返回的候选预测供真实回标时更新计数。"""
    if len(set(keys))!=len(keys) or len(keys)!=len(x) or not keys:
        raise ValueError("服务编号和输入没有一一对应")
    alternatives=bank.predict(current,x)
    if isinstance(router,observer_errors.ObserverErrors):
        choices={"rule":"confusion-uniform","previous_counts_sha256":digest(router.counts.tolist())}
        outputs=router.predict(alternatives,"uniform")
    else:
        choices=router.choices(mode)
        candidates=alternatives if isinstance(router,FeedbackRouter) else router.bank(alternatives)
        outputs=apply_choices(candidates,choices)
    return dict(zip(keys,outputs.tolist(),strict=True)),choices,alternatives


def run(data_root,source_run,drift,output,seed,budget,through,mode):
    if budget<0 or not 3<=through<=10:
        raise ValueError("反馈预算或服务终点无效")
    report=load(source_run/"run.json")
    if report["task"]!="gas" or len(report["rounds"])<4:
        raise ValueError("需要至少四轮已交接的官方气体运行")
    record=report["rounds"][3]
    store=ModelArtifacts(source_run/"models")
    header,blob=store.read(record["model_artifact"])
    envelope=load(source_run/record["submitted"])
    if (Path(record["submitted"]).stem!=envelope["request_id"] or
            envelope["payload"]["batch_id"]!=record["batch"] or
            envelope["payload"]["model_id"]!=record["selected"]):
        raise ValueError("第4轮模型未实际交接")
    samples=load(source_run/"state.json")["samples"]
    ids={k for k,r in samples.items() if r["role"]=="fit"}
    archive=data_root/"gas/cpu-a-guide-20261110-v1.zip"
    if sha256(archive.read_bytes()).hexdigest()!=PACKAGE_SHA:
        raise ValueError("官方训练原包不同")
    from harness.data.official_gas import read_official_gas
    pkg=read_official_gas(archive)
    fit=[s for s in pkg.train if s.sample_id in ids]
    if len(fit)!=len(ids):
        raise ValueError("参照训练清单与原包不符")
    implementation=[sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (feedback_routing,gas_feedback_bank)]
    use_vote=mode in ("vote","vote-local","vote-confirm","guarded-vote")
    if use_vote:
        implementation.append(sha256(Path(vote_feedback.__file__).read_bytes()).hexdigest())
    if mode=="guarded-vote":
        implementation.append(sha256(Path(gas_vote_guard.__file__).read_bytes()).hexdigest())
    if mode=="observer":
        implementation.extend(sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (observer_errors,vote_feedback))
    import sklearn
    identity=digest([PACKAGE_SHA,REGISTERED_SHA256,header["sha256"],digest([s.sample_id for s in fit]),
                     seed,budget,mode,implementation,np.__version__,sklearn.__version__])
    checkpoint=LocalStore(output/"checkpoint")
    endpoint=LocalAcceptance()
    endpoint.output=output/"accepted"
    with checkpoint.exclusive(),TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"A") as trainer:
        saved=checkpoint.load() or {"identity":identity,"accepted":[],"counts":None}
        if saved["identity"]!=identity or len(saved["accepted"])>through-2:
            raise ValueError("续接输出属于其他模型、反馈设置或更后面的批次")
        report_path=output/"run.json"
        if len(saved["accepted"])==through-2 and report_path.exists():
            previous=load(report_path)
            if previous["identity"]==identity and previous["accepted"]==saved["accepted"]:
                for event in previous["accepted"]:
                    receipt=load(endpoint.output/(event["request"]+".json"))
                    if (receipt["request_id"]!=event["request"] or
                            digest([identity,receipt["payload"]])!=event["request"]):
                        raise ValueError("已交接的预测与原始回执不一致")
                return previous
        tic=perf_counter()
        bank=GasFeedbackBank(fit)
        setup_seconds=perf_counter()-tic
        router_type=FeedbackRouter
        if mode=="observer":
            router_type=observer_errors.ObserverErrors
        if use_vote:
            router_type=gas_vote_guard.GuardedVote if mode=="guarded-vote" else vote_feedback.VoteFeedbackRouter
        router=router_type(bank.names,6,saved["counts"])
        p=Path(tmp)/"frozen.npz"
        p.write_bytes(blob)
        model=pair_model(trainer.Model).load(p)
        for b in read_gas_batches(drift,expected_sha256=REGISTERED_SHA256):
            if b.batch_number < 3+len(saved["accepted"]):
                continue
            if b.batch_number>through:
                break
            tic=perf_counter()
            keys=[s.sample_id for s in b.records]
            pred,choice,alts=serve_batch(router,bank,model,keys,np.asarray([s.values for s in b.records]),mode)
            before=router.seen if use_vote else int(router.counts[0].sum())
            payload={"batch":b.batch_number,"inputs":b.member_sha256,"choices":choice,"bank":list(bank.names),
                     "feedback_seen":before,"predictions":pred,"model_sha256":header["sha256"]}
            request=digest([identity,payload])
            receipt=endpoint.submit(request,payload)
            if receipt["request_id"]!=request or receipt["accepted"] is not True:
                raise RuntimeError("当前批输出未完成交接")
            # 这里模拟检测结果返回，只在接受整批输出后揭晓抽中的行。
            ix=np.asarray(sorted(range(len(keys)),key=lambda j:digest([seed,keys[j]]))[:budget],int)
            router.update([b.records[i].label for i in ix],{k:v[ix] for k,v in alts.items()})
            saved["counts"]=router.state() if use_vote else router.counts.tolist()
            saved["accepted"].append({"batch":b.batch_number,"request":request,"feedback_seen":before,
                                      "returned":[keys[i] for i in ix],"seconds":perf_counter()-tic})
            checkpoint.save(saved)
        result={"identity":identity,"policy":mode,"feedback_per_batch":budget,"seed":seed,"source":REGISTERED_SHA256,
                "fit_rows":len(fit),"setup_seconds":setup_seconds,"accepted":saved["accepted"]}
        atomic_json(output/"run.json",result)
        return result


def main(argv):
    p=ArgumentParser(description=__doc__)
    p.add_argument("--source-run",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--data-root",default=public.ROOT/"data/official",type=Path)
    p.add_argument("--drift",default=public.ROOT/"data/public/gas-drift/gas-drift.zip",type=Path)
    p.add_argument("--seed",default=17,type=int)
    p.add_argument("--feedback",type=int,default=32)
    p.add_argument("--through",type=int,choices=range(3,11),default=10)
    p.add_argument("--policy",choices=("local","global","relabel","vote","vote-local","vote-confirm","guarded-vote","observer"),default="local")
    a=p.parse_args(argv)
    out=a.output.resolve()
    if any(out.is_relative_to(f) or f.is_relative_to(out) for f in (a.source_run.resolve(),a.data_root.resolve(),a.drift.resolve())):
        p.error("输出与来源须在独立目录")
    result=run(a.data_root.resolve(),a.source_run.resolve(),a.drift.resolve(),out,a.seed,a.feedback,a.through,a.policy)
    print(f"已交接 {len(result['accepted'])} 个后续采集批，记录在 {out/'run.json'}")
