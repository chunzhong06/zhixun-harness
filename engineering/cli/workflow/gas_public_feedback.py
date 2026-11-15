"""用之前公开轮次返回的标签更新气体合并器，向本地接收端交付本轮预测。"""
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from engineering.cli.workflow.gas_feedback import load,serve_batch
from engineering.cli.workflow import gas_observer_predict
from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.models import observer_errors,gas_feedback_bank,vote_feedback
from harness.models import field_classification, site_training
from harness.models import served_scale_gas, coverage_scale_gas
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.models.official_rul_cpu import _trainer
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json


def run(data_root,source,out,seed,budget,through,total_feedback=None,*,field_mode=None):
    archive=data_root/"gas/cpu-a-guide-20261110-v1.zip"
    if sha256(archive.read_bytes()).hexdigest()!=PACKAGE_SHA or not 1<=through<=10 or budget<0 or (total_feedback is not None and total_feedback<0):
        raise ValueError("需要有效的气体原包、轮次及反馈配置")
    if source is None:
        engine,batches=public.assemble_public("gas",data_root,out/"regular","selected",{"mode":"rule"})
        fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
        source_identity=["live-snapshot-v1",gas_observer_predict.implementation(),engine.tools.identity,engine.planner.identity,digest(asdict(engine.config)),
                         [digest(asdict(batch)) for batch in batches]]
        artifacts=engine.tools.runner.artifacts
    else:
        report=load(source/"run.json")
        state=load(source/"state.json")
        if report["task"]!="gas" or through>len(report["rounds"]):
            raise ValueError("公开来源任务或轮次不符")
        samples=state["samples"]
        fit_ids={k for k,v in samples.items() if v["role"]=="fit"}
        source_identity=[state["fingerprint"],report["receipt_key"]]
        artifacts=ModelArtifacts(source/"models")
    package=read_official_gas(archive)
    training=[r for r in package.train if r.sample_id in fit_ids]
    if len(training)!=len(fit_ids):
        raise ValueError("源端训练编号不属于原包")
    bank=gas_feedback_bank.GasFeedbackBank(training)
    implementation=[sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (observer_errors,gas_feedback_bank,vote_feedback)]
    identity=digest(["public-prior-feedback-v1",PACKAGE_SHA,*source_identity,seed,budget,implementation])
    if total_feedback is not None:
        identity=digest([identity,"earliest-within-cap",total_feedback])
    if field_mode is not None:
        identity=digest([identity,"field",field_mode,field_classification.implementation()])
        if field_mode in site_training.MODES:
            identity=digest([identity,site_training.implementation()])
        if field_mode == served_scale_gas.MODE:
            identity=digest([identity,served_scale_gas.implementation()])
        if field_mode == coverage_scale_gas.MODE:
            identity=digest([identity,coverage_scale_gas.implementation()])
    store=LocalStore(out/"checkpoint")
    endpoint=LocalAcceptance()
    endpoint.output=out/"accepted"
    with store.exclusive(),TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"A") as trainer:
        saved=store.load() or {"identity":identity,"accepted":[],"counts":None}
        if saved["identity"]!=identity or len(saved["accepted"])>through:
            raise ValueError("续接目录与来源、模型、反馈预算或轮次不一致")
        field=None
        if field_mode is not None:
            classifier=site_training.SiteTraining if field_mode in site_training.MODES else field_classification.FieldClassifier
            if field_mode == served_scale_gas.MODE:
                classifier=served_scale_gas.ServedScaleGas
            elif field_mode == coverage_scale_gas.MODE:
                classifier=coverage_scale_gas.CoverageScaleGas
            field=classifier([r.values for r in training],[r.label for r in training],bank.names,6,field_mode,saved.get("field"))
        router=field.observer if field is not None else observer_errors.ObserverErrors(bank.names,6,saved["counts"])
        received=set()
        for j in range(through):
            batch_id=batches[j].batch_id if source is None else report["rounds"][j]["batch"]
            if j<len(saved["accepted"]):
                event=saved["accepted"][j]
                rec=load(endpoint.output/(event["request"]+".json"))
                if rec["request_id"]!=event["request"] or digest([identity,rec["payload"]])!=event["request"] or event["round"]!=batch_id:
                    raise ValueError("历史预测的内容与确认不符")
                received.update(event["returned"])
                continue
            if source is None:
                state=engine.run((batches[j],))
                actual=state["completed"][batch_id]
                samples=state["samples"]
                artifact=state["models"][actual["model_id"]]["artifact"]
            else:
                r=report["rounds"][j]
                artifact=r["model_artifact"]
                source_event=load(source/r["submitted"])
                actual=source_event["payload"]
                if Path(r["submitted"]).stem!=source_event["request_id"] or actual["model_id"]!=r["selected"] or actual["batch_id"]!=batch_id:
                    raise ValueError("原服务记录与当轮权重不符")
            header,blob=artifacts.read(artifact)
            path=Path(tmp)/"current.npz"
            path.write_bytes(blob)
            current=pair_model(trainer.Model).load(path)
            keys=sorted(actual["predictions"])
            if any(samples[k]["role"]!="serve" or samples[k]["label"] is not None or samples[k]["available_at"]!=j for k in keys):
                raise ValueError("当轮服务输入的资格或时间不符")
            tic=perf_counter()
            features=np.asarray([samples[k]["values"] for k in keys])
            pred,rule,alts=serve_batch(router,bank,current,keys,features,"observer")
            if alts["current"].tolist()!=[actual["predictions"][k] for k in keys] or router.counts[0].sum()!=len(received):
                raise ValueError("原权重未复现或回标计数不一致")
            payload={"round":batch_id,"weights":header["sha256"],"source_inputs":actual["input_digest"],
                     "feedback_seen":len(received),"rule":rule,"predictions":pred}
            if field is not None:
                tools=field.predict(features,alts)
                payload.update(field_mode=field_mode, selected_tool=field.current(),
                               predictions=dict(zip(keys,tools[field.current()].tolist(),strict=True)),
                               candidate_predictions={n:v.tolist() for n,v in tools.items()},
                               source_predictions={n:v.tolist() for n,v in alts.items()}, candidate_ids=keys)
            if source is None:
                snapshot_model=field.snapshot() if field is not None else None
                payload["snapshot"]=gas_observer_predict.freeze(out/"snapshots",blob,header["sha256"],fit_ids,
                                                                 router.counts,field_model=snapshot_model)
            request=digest([identity,payload])
            receipt=endpoint.submit(request,payload)
            if not receipt.get("accepted") or receipt.get("request_id")!=request:
                raise RuntimeError("气体服务输出尚未确认")
            if field_mode == coverage_scale_gas.MODE:
                field.observe_served(features)
            candidates=[k for k,v in samples.items() if v["role"] in ("explore","adopt") and v["available_at"]==j]
            remaining=budget if total_feedback is None else min(budget,max(total_feedback-len(received),0))
            if field is not None and j==9:
                selected=[]
            elif field_mode in (served_scale_gas.MODE, coverage_scale_gas.MODE) and candidates and remaining:
                query_x=np.asarray([samples[k]["values"] for k in candidates])
                selected=field.select(candidates,query_x,bank.predict(current,query_x),remaining,seed)
            else:
                selected=sorted(candidates,key=lambda k:digest([seed,k]))[:remaining]
            if set(selected)&(received|set(keys)):
                raise ValueError("回标编号重复或与服务数据重合")
            if selected:
                returned_x=np.asarray([samples[k]["values"] for k in selected])
                fp=bank.predict(current,returned_x)
                if field is not None:
                    before=field.predict(returned_x,fp)
                    field.update(returned_x,fp,[samples[k]["label"] for k in selected],before)
                else:
                    router.update([samples[k]["label"] for k in selected],fp)
            received.update(selected)
            if field_mode == served_scale_gas.MODE:
                field.observe_served(features)
            saved["counts"]=router.counts.tolist()
            if field is not None:
                saved["field"]=field.state()
                previous=saved.get("field_cost",{"logistic_and_tree_fits":0,"fit_cpu_seconds":0.})
                saved["field_cost"]={"logistic_and_tree_fits":previous["logistic_and_tree_fits"]+field.fits,
                                    "fit_cpu_seconds":previous["fit_cpu_seconds"]+field.cpu}
                field.fits,field.cpu=0,0.
            saved["accepted"].append({"round":batch_id,"request":request,"returned":selected,"seconds":perf_counter()-tic})
            store.save(saved)
        result={"identity":identity,"accepted":saved["accepted"],"seed":seed,"budget":budget}
        if field is not None:
            result.update(field_mode=field_mode,field_cost=saved["field_cost"])
        if total_feedback is not None:
            result["total_feedback"]=total_feedback
        atomic_json(out/"run.json",result)
        return result


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",default=public.ROOT/"data/official",type=Path)
    parser.add_argument("--source-run",type=Path,help="可选的历史气体服务流；缺省从原包逐轮运行")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--feedback",type=int,default=32)
    parser.add_argument("--total-feedback",type=int,help="整条流最多返回的标签数；按到达顺序尽早校准")
    parser.add_argument("--field-mode",choices=(*field_classification.MODES,*site_training.MODES,served_scale_gas.MODE,coverage_scale_gas.MODE))
    parser.add_argument("--through",type=int,default=10)
    args=parser.parse_args(argv)
    out=args.output.resolve()
    inputs=[args.data_root.resolve()]
    source=args.source_run.resolve() if args.source_run else None
    if source is not None:
        inputs.append(source)
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        parser.error("输出与来源须使用独立目录")
    r=run(args.data_root.resolve(),source,out,args.seed,args.feedback,args.through,args.total_feedback,field_mode=args.field_mode)
    print(f"已交接 {len(r['accepted'])} 个公开气体轮次")
