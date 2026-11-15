"""从真正交付过的公开权重继续：每轮独立确认输入校正，成功交接后保存其适用范围。"""
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

import numpy as np

from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.models.gas_views import pair_model
from harness.models.bearing_stumps import stump_model
from harness.models.device_neighbors import DeviceNeighborUpdate
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from harness.models.scoped_calibration import ScopedCalibration, TrainingCalibrationBank
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json


def read(path):
    return json.loads(path.read_bytes())


def run(task, data_root, source_run, output, after, through, policy):
    if policy=="device-neighbor" and task!="engine":
        raise ValueError("设备近邻更新只适用于发动机")
    if not 1<=after<through<=10:
        raise ValueError("须从已经交出的某轮开始，终点在第十轮以内")
    for owner in (source_run.resolve(),data_root.resolve()):
        if output.resolve().is_relative_to(owner) or owner.is_relative_to(output.resolve()):
            raise ValueError("续跑输出、原始运行及官方数据目录须独立")
    source = read(source_run/"run.json")
    if source["task"]!=task or len(source["rounds"])<after:
        raise ValueError("源运行任务或轮数不符")
    if task=="gas":
        file,checksum,reader="cpu-a-guide-20261110-v1.zip",public.PACKAGE_SHA,public.read_official_gas
    else:
        file,checksum,reader=public.PACKAGES[task]
    archive=data_root/task/file
    if sha256(archive.read_bytes()).hexdigest()!=checksum:
        raise ValueError("官方训练包身份改变")
    package=reader(archive)
    if task=="gas":
        batches,held=public.partition_gas(package,read(public.ROOT/"engineering/configs/experiments/official-gas-pilot.json"))
        batches=public.gas_service_windows(batches,held)
    else:
        rows,units=public.partition_rul(package,read(public.PROTOCOL))
        batches=public.windows(package,units,rows["fit"],read(public.SERIES),task)
    prefix=[]
    for r,b in zip(source["rounds"][:after],batches[:after],strict=True):
        ack=read(source_run/r["submitted"])
        payload=ack["payload"]
        if (payload["model_id"]!=r["selected"] or payload["input_digest"]!=digest(asdict(b))
                or payload["batch_id"]!=b.batch_id):
            raise ValueError("历史交接来自另一批输入或权重")
        prefix.append({"batch":b.batch_id,"request":ack["request_id"],"outputs_sha256":digest(payload["predictions"])})
    last=source["rounds"][after-1]
    registered=read(source_run/"state.json")["models"][last["selected"]]["artifact"]
    if registered!=last["model_artifact"]:
        raise ValueError("热身末轮的模型登记与权重文件不同")
    head,blob=ModelArtifacts(source_run/"models").read(last["model_artifact"])
    if head["request"].get("package_sha256")!=checksum:
        raise ValueError("在用权重来自其他原包")
    store=LocalStore(output/"checkpoint")
    with store.exclusive(), TemporaryDirectory(prefix="public-scope-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),{"gas":"A","bearing":"B","engine":"C"}[task]) as trainer:
        file=Path(tmp)/"in-use.npz"
        file.write_bytes(blob)
        model_type=pair_model if task=="gas" else (stump_model if "bearing_stump" in head["request"] else None)
        model=(model_type(trainer.Model) if model_type else trainer.Model).load(file)
        last_inputs=sorted((r for r in batches[after-1].samples if r.role=="serve" and r.available_at==after-1),
                           key=lambda r:r.sample_id)
        actual=model.predict(np.array([r.values for r in last_inputs]))
        if task=="gas":
            actual=actual.argmax(axis=1)+1
        previously_issued=read(source_run/last["submitted"])["payload"]["predictions"]
        if previously_issued.keys()!={r.sample_id for r in last_inputs} or not np.allclose(
                actual,[previously_issued[r.sample_id] for r in last_inputs],rtol=1e-12,atol=1e-8):
            raise ValueError("恢复的底层模型没有复现此前发出的预测")
        fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
        fit_x=np.array([r.values for r in package.train if r.sample_id in fit_ids])
        bank=TrainingCalibrationBank(model,fit_x) if policy.startswith("bank-") else None
        if policy=="device-neighbor":
            action=DeviceNeighborUpdate(model,head["sha256"],[r for r in package.train if r.sample_id in fit_ids])
        else:
            action=ScopedCalibration(model,head["sha256"],task,fit_x,policy,bank)
        identity=digest([action.identity,prefix,checksum,sha256(Path(public.__file__).read_bytes()).hexdigest(),
                         [digest(asdict(b)) for b in batches[after:]]])
        saved=store.load() or {"identity":identity,"layer_state":{},"accepted":[]}
        if saved["identity"]!=identity or after+len(saved["accepted"])>through:
            raise ValueError("续跑路径属于另一协议或已经超过指定轮次")
        endpoint=LocalAcceptance()
        endpoint.output=output/"accepted"
        for b in batches[after+len(saved["accepted"]):through]:
            rows=[[asdict(r) for r in b.samples if r.role==role and r.available_at==b.ordinal]
                  for role in ("explore","adopt","serve")]
            state,decision,predictions=action.advance(saved["layer_state"],b.batch_id,*rows)
            payload={"batch_id":b.batch_id,"input_digest":digest(asdict(b)),"base_sha256":head["sha256"],
                     "calibration_sha256":digest(state["layer"]),"decision":decision,"predictions":predictions}
            request=digest([identity,payload])
            receipt=endpoint.submit(request,payload)
            if receipt.get("request_id")!=request or receipt.get("accepted") is not True:
                raise RuntimeError("接收端尚未确认这批预测")
            saved["accepted"].append({"batch":b.batch_id,"request":request,"outputs_sha256":digest(predictions)})
            saved["layer_state"]=state
            store.save(saved)
        report={"task":task,"policy":policy,"base_sha256":head["sha256"],"prefix":prefix,
                "accepted":saved["accepted"],"rounds":after+len(saved["accepted"]),"identity":identity}
        atomic_json(output/"run.json",report)
        return report


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=public.TASKS)
    parser.add_argument("--data-root",type=Path,default=public.ROOT/"data/official")
    parser.add_argument("--source-run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--after",type=int,default=4)
    parser.add_argument("--through",type=int,default=10)
    parser.add_argument("--policy",required=True,choices=("off","keep","refresh","expiry","row","expiry-row",
                        "bank-keep","bank-refresh","bank-expiry","bank-route","bank-route-refresh","device-neighbor"))
    args=parser.parse_args(argv)
    result=run(args.task,args.data_root,args.source_run,args.output,args.after,args.through,args.policy)
    print(f"{args.task} 已交接至第 {result['rounds']} 轮。报告保存在 {args.output/'run.json'}")
