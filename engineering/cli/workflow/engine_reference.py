"""逐批运行现行工业模型，再用当批独立反馈为实际服务预测选择近邻权重。"""
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.models import feedback_neighbors
from harness.models.feedback_neighbors import CurrentNeighbors
from harness.models.official_rul_cpu import _trainer
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json


def run_prepared(data_root,output,engine,batches,through,start=1,*,policy="weight",task="engine"):
    legacy=policy in ("weight","always")
    setting=policy in ("source-setting","source-combinations") and task=="engine"
    phase=policy=="phase-fixed" and task=="bearing"
    native=policy in ("full-condition-native","continued-condition-native") and task=="engine"
    inquiry=policy in ("training-inquiry","training-inquiry-fixed") and task=="engine"
    early=(policy in ("early-reference","source-tool-choice") and task=="engine") or native or inquiry
    if (not 1<=start<=10 or not 1<=through<=len(batches) or task not in ("bearing","engine")
            or (legacy and task!="engine") or (not legacy and not setting and not phase and not early and policy not in ("select","median-confirm","tree-confirm","median","catboost-only"))):
        raise ValueError("无效的起始轮、服务轮或参照策略")
    archive=data_root/task/public.PACKAGES[task][0]
    checksum=sha256(archive.read_bytes()).hexdigest()
    if checksum!=public.PACKAGES[task][1]:
        raise ValueError("寿命任务原包变化")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    train=[r for r in engine.tools.package.train if r.sample_id in ids]
    if legacy:
        implementation=feedback_neighbors
    elif policy=="source-setting":
        from harness.models import engine_setting_reference as implementation
    elif setting:
        from harness.models import source_combinations as implementation
    elif policy=="source-tool-choice":
        from harness.models import engine_source_choice as implementation
    elif policy=="full-condition-native":
        from harness.models import full_condition_engine as implementation
    elif policy=="continued-condition-native":
        from harness.models import continued_condition_engine as implementation
    elif inquiry:
        from harness.models import engine_training_inquiry as implementation
    elif early:
        from harness.models import early_engine_reference as implementation
    elif phase:
        from harness.models import bearing_phase_reference as implementation
    else:
        from harness.models import life_reference as implementation
    implementation_hash=sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    if policy=="source-combinations":
        # 首版实际交接使用的源文件摘要。增加只读状态导出仍可接续原记录；改变预测规则须换版本。
        implementation_hash="9bdcb11ed5d1669c11cc72793542333ad7d4647a76639968b193331573d39660"
    identity=digest([checksum,digest(asdict(engine.config)),engine.tools.identity,engine.planner.identity,start,policy,
                     implementation_hash,
                     [digest(asdict(batch)) for batch in batches]])
    store=LocalStore(output/"checkpoint")
    endpoint=LocalAcceptance()
    endpoint.output=output/"accepted"
    with store.exclusive(), TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C" if task=="engine" else "B") as trainer:
        # 训练库身份参与恢复；构造不访问未来反馈及服务目标。
        if setting:
            reference_type=implementation.EngineSettingReference if policy=="source-setting" else implementation.SourceCombinations
            lookup=reference_type([r.values for r in train])
        elif policy=="source-tool-choice":
            lookup=implementation.SourceSelectedEngine(engine.task,train,engine.tools.package.train,checksum,output/"references")
        elif native:
            cls=implementation.FullConditionEngine if policy=="full-condition-native" else implementation.ContinuedConditionEngine
            lookup=cls(engine.task,train,engine.tools.package,checksum,output/"references",trainer)
        elif inquiry:
            opening=batches[0]
            explore,confirm=([s for s in opening.samples if s.role==role and s.available_at==opening.ordinal]
                             for role in ("explore","adopt"))
            lookup=implementation.EngineTrainingInquiry.from_public_engine(engine.task,train,engine.tools.package,
                checksum,output/"references",trainer,explore,confirm,
                fixed=policy=="training-inquiry-fixed")
        elif early:
            lookup=implementation.EarlyEngineReference(engine.task,train,engine.tools.package.train,checksum,output/"references")
        elif phase:
            lookup=implementation.BearingPhaseReference(engine.task,train,checksum,output/"references")
        elif legacy:
            lookup=CurrentNeighbors(None,checksum,train)
        else:
            lookup=implementation.LifeReference(engine.task,train,checksum,output/"references",policy)
        key=digest([identity,lookup.identity])
        saved=store.load()
        # 历史运行的接收方已经确认原数值；新运行把非负输出约定纳入交接身份。
        nonnegative=early and (saved is None or saved.get("nonnegative_cycles") is True)
        if nonnegative:
            key=digest([key,"nonnegative-cycles-v1"])
        if saved is None:
            saved={"identity":key,"accepted":[]}
            if nonnegative:
                saved["nonnegative_cycles"]=True
        if saved["identity"]!=key or len(saved["accepted"])>through:
            raise ValueError("已有交接属于其它实验或超过当前请求")
        if early:
            atomic_json(output/"source-combinations.json",lookup.repair.state())
        if policy=="source-combinations":
            atomic_json(output/"source-combinations.json",lookup.state())
        p=Path(tmp)/"published.npz"
        for batch in batches[len(saved["accepted"]):through]:
            state=engine.run((batch,))
            regular=state["completed"][batch.batch_id]
            artifact=state["models"][regular["model_id"]]["artifact"]
            header,blob=engine.tools.runner.artifacts.read(artifact)
            p.write_bytes(blob)
            model=trainer.Model.load(p)
            groups=[[asdict(r) for r in batch.samples if r.role==role and r.available_at==batch.ordinal]
                    for role in ("explore","adopt","serve")]
            tic=perf_counter()
            if batch.ordinal+1<start:
                outputs=base=regular["predictions"]
                decision=({"weight":0.,"warmup":True} if legacy else {"chosen":"current","warmup":True})
            elif policy=="always":
                rows,b,ref=lookup.observations(model,groups[2])
                ids_now=[r["sample_id"] for r in rows]
                if any(r["label"] is not None for r in rows):
                    raise ValueError("服务输入含标签")
                base=dict(zip(ids_now,map(float,b),strict=True))
                outputs=dict(zip(ids_now,map(float,.5*(ref+b)),strict=True))
                decision={"weight":.5}
            else:
                outputs,base,decision=lookup.advance_current(model,*groups)
                if nonnegative:
                    outputs={k:max(0.,v) for k,v in outputs.items()}
                    decision["nonnegative_cycles"]=True
            elapsed=perf_counter()-tic
            if base.keys()!=regular["predictions"].keys() or any(abs(v-regular["predictions"][i])>1e-8 for i,v in base.items()):
                raise ValueError("重载权重未复现本轮工作进程交出的预测")
            payload={"batch":batch.batch_id,"input_digest":regular["input_digest"],"selected":regular["model_id"],
                     "weights_sha256":header["sha256"],"bank":lookup.identity,"decision":decision,"predictions":outputs}
            request=digest([key,payload])
            receipt=endpoint.submit(request,payload)
            if receipt.get("request_id")!=request or not receipt.get("accepted"):
                raise RuntimeError("服务交接未确认")
            used={"weight":decision["weight"]} if legacy else {"chosen":decision["chosen"]}
            saved["accepted"].append({"batch":batch.batch_id,"request":request,**used,
                                      "outputs_sha256":digest(outputs),"reference_seconds":elapsed})
            store.save(saved)
        report={"policy":policy,"start":start,"completed_rounds":len(saved["accepted"]),"identity":key,
                "training_rows":len(train),"training_devices":len(set(r.unit_id for r in train)),"accepted":saved["accepted"]}
        if policy=="source-combinations":
            # 单条推理只读模型引用和组合计数，不需打开含回标数据的训练状态。
            regular_state=engine.store.load()
            published={regular_state["completed"][entry["batch"]]["model_id"] for entry in saved["accepted"]}
            atomic_json(output/"issued-models.json",{mid:regular_state["models"][mid]["artifact"] for mid in sorted(published)})
        if setting:
            report.update(task=task,setting_reference=lookup.identity,
                          known_conditions=len(lookup.mapping if policy=="source-setting" else lookup.counts))
        elif inquiry:
            report.update(task=task,inquiry_artifact=lookup.model_ref,reference_identity=lookup.identity,
                          training_inquiry=lookup.trace)
        elif native:
            report.update(task=task,native_artifact=lookup.model_ref,native_sha256=lookup.native_sha256,
                          reference_identity=lookup.identity,training_rows=len(engine.tools.package.train),anchor_rows=len(train))
        elif early:
            report.update(task=task,tree_artifact=lookup.model_ref,reference_identity=lookup.identity)
            if policy=="source-tool-choice":
                report.update(source_choice=lookup.decision["choice"],source_scores=lookup.decision["score"])
        elif phase:
            report.update(task=task,scale_minutes=lookup.scale,tree_artifact=lookup.model_ref,reference_identity=lookup.identity)
        elif not legacy:
            report.update(task=task,train_median=lookup.median,tree_artifact=lookup.model_ref)
        atomic_json(output/"run.json",report)
        return report


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=public.ROOT/"data/official")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",choices=range(1,11),type=int,default=10)
    parser.add_argument("--start",choices=range(1,11),type=int,default=1)
    parser.add_argument("--policy",choices=("weight","always","source-setting","source-combinations"),default="weight")
    args=parser.parse_args(argv)
    data_root,output=args.data_root.resolve(),args.output.resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        parser.error("输入与输出目录须独立")
    engine,batches=public.assemble_public("engine",data_root,output/"regular","first-proposal",{"mode":"rule"})
    result=run_prepared(data_root,output,engine,batches,args.rounds,args.start,policy=args.policy)
    print(f"发动机已交接 {result['completed_rounds']} 轮，记录保存于 {output/'run.json'}")


def life_main(argv):
    parser=ArgumentParser(description="比较当轮普通寿命预测与训练数据提供的固定参照")
    parser.add_argument("task",choices=("engine","bearing"))
    parser.add_argument("--policy",choices=("median-confirm","tree-confirm","select","median","catboost-only","phase-fixed"),required=True)
    parser.add_argument("--data-root",type=Path,default=public.ROOT/"data/official")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rounds",type=int,choices=range(1,11),default=10)
    args=parser.parse_args(argv)
    if args.policy=="phase-fixed" and args.task!="bearing":
        parser.error("本地阶段参照已核对轴承任务")
    root,out=args.data_root.resolve(),args.output.resolve()
    if out.is_relative_to(root) or root.is_relative_to(out):
        parser.error("输入与输出目录须独立")
    ordinary="first-proposal" if args.task=="engine" else "selected"
    harness,batches=public.assemble_public(args.task,root,out/"regular",ordinary,{"mode":"rule"})
    result=run_prepared(root,out,harness,batches,args.rounds,policy=args.policy,task=args.task)
    print(f"{args.task} 已交接 {result['completed_rounds']} 轮，记录保存于 {out/'run.json'}")
