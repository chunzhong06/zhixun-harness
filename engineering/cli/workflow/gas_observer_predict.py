"""从服务交付时封存的状态读取单行 JSON，返回与赛题接口一致的六类分数。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.data.official_gas import LABELS,read_gas_training,prediction_request
from harness.models import observer_errors,gas_feedback_bank,vote_feedback,gas_views
from harness.models import field_classification
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json


def implementation():
    return [sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (observer_errors,gas_feedback_bank,vote_feedback,gas_views)]


def freeze(folder,blob,model_hash,fit_ids,counts,*,field_model=None):
    """记录本次交付前已知的计数。即使回执稍后重试，同一状态也生成同一份快照。"""
    if sha256(blob).hexdigest()!=model_hash:
        raise ValueError("赛方权重内容身份不符")
    store=folder/"models"
    store.mkdir(parents=True,exist_ok=True)
    path=store/(model_hash+".npz")
    if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=model_hash:
        temp=path.with_suffix(".tmp")
        temp.write_bytes(blob)
        temp.replace(path)
    record={"version":1,"package":PACKAGE_SHA,"weights":model_hash,"fit_ids":sorted(fit_ids),
            "counts":counts.tolist(),"modules":implementation()}
    if field_model is not None:
        record.update(version=2, field_model=field_model, field_module=field_classification.implementation())
    key=digest(record)
    atomic_json(folder/(key+".json"),record)
    return key


def infer(snapshot,data_root,requests):
    state=json.loads(snapshot.read_bytes())
    if snapshot.stem!=digest(state) or state["version"] not in (1,2) or state["modules"]!=implementation():
        raise ValueError("快照或推理实现发生变化")
    if state["version"]==2 and state["field_module"]!=field_classification.implementation():
        raise ValueError("现场分类模型版本发生变化")
    archive=data_root/"gas/cpu-a-guide-20261110-v1.zip"
    if state["package"]!=PACKAGE_SHA or sha256(archive.read_bytes()).hexdigest()!=PACKAGE_SHA:
        raise ValueError("官方源训练包不符")
    weights=state["weights"]
    if len(weights)!=64 or any(c not in "0123456789abcdef" for c in weights):
        raise ValueError("无效的权重身份")
    path=snapshot.parent/"models"/(weights+".npz")
    if sha256(path.read_bytes()).hexdigest()!=weights:
        raise ValueError("快照的赛方权重不符")
    ids=set(state["fit_ids"])
    train=[r for r in read_gas_training(archive) if r.sample_id in ids]
    if len(train)!=len(ids) or len(ids)!=len(state["fit_ids"]):
        raise ValueError("快照训练编号不符")
    bank=gas_feedback_bank.GasFeedbackBank(train)
    router=observer_errors.ObserverErrors(bank.names,len(LABELS),state["counts"])
    smooth=router.counts+1+np.eye(len(LABELS),dtype=int)[None,:,:]
    log_rates=np.log(smooth/smooth.sum(axis=2,keepdims=True))
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"A") as trainer:
        current=gas_views.pair_model(trainer.Model).load(path)
        for req in requests:
            if (not isinstance(req,dict) or set(req)-{"inputs","task"} or req.get("task","classification")!="classification"
                    or not isinstance(req.get("inputs"),dict) or set(req["inputs"])!={"features"}):
                raise ValueError("单条预测需要分类任务的 inputs.features")
            values=req["inputs"]["features"]
            prediction_request(values)
            if state["version"]==2:
                probs=field_classification.model_scores(state["field_model"],[values],len(LABELS))[0]
                yield {"outputs":[{"label":k,"score":float(v)} for k,v in zip(LABELS,probs,strict=True)]}
                continue
            outputs=bank.predict(current,np.asarray([values]))
            votes=[int(outputs[k][0])-1 for k in bank.names]
            logits=sum(log_rates[i,:,v] for i,v in enumerate(votes))
            probs=np.exp(logits-logits.max())
            chosen=int(router.predict(outputs,"uniform")[0])-1
            # 选择器已有固定投票优先的平手约定；JSON 接收者则以类别编号破平。
            if probs.argmax()!=chosen:
                probs[chosen]=np.nextafter(probs.max(),np.inf)
            probs/=probs.sum()
            if probs.argmax()!=chosen:
                probs[chosen]=np.nextafter(probs.max(),np.inf)
            yield {"outputs":[{"label":k,"score":float(v)} for k,v in zip(LABELS,probs,strict=True)]}


def main(argv):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",required=True,type=Path)
    parser.add_argument("--data-root",default=public.ROOT/"data/official",type=Path)
    args=parser.parse_args(argv)
    requests=(json.loads(line) for line in sys.stdin if line.strip())
    for response in infer(args.snapshot,args.data_root,requests):
        print(json.dumps(response,allow_nan=False),flush=True)
