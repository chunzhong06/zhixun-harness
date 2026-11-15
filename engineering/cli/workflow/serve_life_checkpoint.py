"""用冻结且可按赛方类载入的寿命权重交付各批，无须目标现场反馈。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.models.official_rul_cpu import _trainer
from harness.models.model_artifacts import ModelArtifacts
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json


def confirm_candidate(current, candidate, explore, confirm):
    """只有当前两组已返回的独立标签都改善时使用候选。"""
    checks=[]
    for rows in (explore,confirm):
        if not rows:
            return False,checks
        features=np.asarray([r["values"] for r in rows])
        truth=np.asarray([r["label"] for r in rows],dtype=float)
        if not np.isfinite(truth).all():
            raise ValueError("用于确认的标签无效")
        checks.append([float(np.sqrt(np.mean((model.predict(features)-truth)**2))) for model in (current,candidate)])
    return all(after+1e-12<before for before,after in checks),checks


def run(task,archive,weights,source,output,through,confirm=False):
    def read(path):
        return json.loads(path.read_bytes())
    report=read(source/"run.json")
    state=read(source/"state.json")
    records=report["rounds"]
    if report["task"]!=task or not 1<=through<=len(records):
        raise ValueError("任务或服务轮次与来源不符")
    model_hash=sha256(weights.read_bytes()).hexdigest()
    package_hash=sha256(archive.read_bytes()).hexdigest()
    if package_hash!=public.PACKAGES[task][1]:
        raise ValueError("官方权重加载器来源已变")
    identity=digest([task,package_hash,model_hash,state["fingerprint"],report["receipt_key"],"fixed-checkpoint-v1"])
    if confirm:
        identity=digest([identity,"two-independent-feedback-splits-v1"])
        artifacts=ModelArtifacts(source/"models")
    store=LocalStore(output/"state")
    endpoint=LocalAcceptance()
    endpoint.output=output/"accepted"
    with store.exclusive(),TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        model=trainer.Model.load(weights)
        saved=store.load() or {"identity":identity,"accepted":[]}
        if saved["identity"]!=identity or len(saved["accepted"])>through:
            raise ValueError("续接目录与目标权重、来源或服务轮次不同")
        for j,r in enumerate(records[:through]):
            src=read(source/r["submitted"])
            if Path(r["submitted"]).stem!=src["request_id"] or src["payload"]["model_id"]!=r["selected"]:
                raise ValueError("源服务回执或权重不符")
            keys=sorted(src["payload"]["predictions"])
            rows=[state["samples"][i] for i in keys]
            if not rows or any(v["label"] is not None or v["role"]!="serve" or v["available_at"]!=j for v in rows):
                raise ValueError("需要本轮尚未回标的服务输入")
            x=np.asarray([v["values"] for v in rows])
            pred=model.predict(x)
            details={}
            delivered_hash=model_hash
            if confirm:
                header,blob=artifacts.read(r["model_artifact"])
                old_path=Path(tmp)/"current.npz"
                old_path.write_bytes(blob)
                current=trainer.Model.load(old_path)
                original=current.predict(x)
                if not np.allclose(original,[src["payload"]["predictions"][k] for k in keys],rtol=0,atol=1e-8):
                    raise ValueError("原在用权重未复现实际服务输出")
                known=[v for v in state["samples"].values() if v["available_at"]==j and v["role"] in ("explore","adopt")]
                explore=[v for v in known if v["role"]=="explore"]
                adopt=[v for v in known if v["role"]=="adopt"]
                allowed,checks=confirm_candidate(current,model,explore,adopt)
                if not allowed:
                    pred=original
                    delivered_hash=header["sha256"]
                details={"chosen":"candidate" if allowed else "current","checks":checks,"candidate_sha256":model_hash,
                         "feedback":digest([sorted((v["sample_id"],v["role"],v["label"]) for v in known)])}
            outputs=dict(zip(keys,pred.tolist(),strict=True))
            payload={"batch":r["batch"],"weights_sha256":delivered_hash,"inputs_sha256":digest([keys,x.tolist()]),"predictions":outputs,**details}
            request=digest([identity,payload])
            entry={"batch":r["batch"],"request":request,"rows":len(keys)}
            if j<len(saved["accepted"]):
                if saved["accepted"][j]!=entry or read(endpoint.output/(request+".json"))["payload"]!=payload:
                    raise ValueError("已有交接与冻结权重重新计算的结果不同")
                continue
            receipt=endpoint.submit(request,payload)
            if receipt["request_id"]!=request or not receipt["accepted"]:
                raise ValueError("当前服务预测未确认")
            saved["accepted"].append(entry)
            store.save(saved)
        out={"task":task,"weights_sha256":model_hash,"identity":identity,"accepted":saved["accepted"]}
        atomic_json(output/"run.json",out)
        return out


def main(argv):
    p=ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("bearing","engine"))
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--through",type=int,default=10)
    p.add_argument("--confirm",action="store_true",help="只有当窗探索和确认两份反馈均改善才使用候选权重")
    args=p.parse_args(argv)
    out=args.output.resolve()
    source=args.source_run.resolve()
    if any(out.is_relative_to(f) or f.is_relative_to(out) for f in (source,args.weights.resolve(),public.ROOT/"data")):
        p.error("交付结果与输入须在独立目录")
    archive=public.ROOT/"data/official"/args.task/public.PACKAGES[args.task][0]
    result=run(args.task,archive,args.weights.resolve(),source,out,args.through,args.confirm)
    print(f"已实际交接 {sum(r['rows'] for r in result['accepted'])} 条寿命预测")
