"""按编号决定公开任务的角色和轮次，重新运行维护，再比较同条件的交付结果。"""
from argparse import ArgumentParser
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,atomic_json,digest,read,official
from engineering.cli.workflow import public_maintenance as public
from harness.contracts import BatchContext
from harness.models.feedback_routing import apply_choices
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.official_rul_cpu import _trainer
from harness.models.vote_feedback import majority,VoteFeedbackRouter
from harness.models.observer_errors import ObserverErrors


def partition(seed,package,protocol):
    count=protocol["rounds"]
    rows=sorted(package.validation,key=lambda r:digest(["roles",seed,r.sample_id]))
    sets={role:rows[i::3] for i,role in enumerate(("serve","explore","adopt"))}
    rounds=[[] for _ in range(count)]
    for role in ("serve","explore","adopt"):
        ordered=sorted(sets[role],key=lambda r:digest(["round",seed,r.sample_id]))
        for j,row in enumerate(ordered):
            rounds[j%count].append(replace(row,role=role,available_at=j%count,label=None if role=="serve" else row.label))
    return tuple(BatchContext(package.task.task_id,f"round-{j+1}",j,tuple(package.train if j==0 else ())+tuple(r)) for j,r in enumerate(rounds)),()


def main(seed):
    folder=OUT/"observer-errors/label-blind"/str(seed)
    data_root=ROOT/"data/official"
    engine,batches=public.assemble_public("gas",data_root,folder/"regular","selected",{"mode":"rule"},
                                            gas_partition=lambda p,c:partition(seed,p,c))
    pkg,_,_=official("gas")
    bank=GasFeedbackBank([r for r in batches[0].samples if r.role=="fit"])
    router=ObserverErrors(bank.names,6)
    old=VoteFeedbackRouter(bank.names,6)
    released=set()
    groups=[]
    trace=[]
    forecasts={}
    archive=data_root/"gas/cpu-a-guide-20261110-v1.zip"
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"A") as trainer:
        for batch in batches:
            state=engine.run((batch,))
            result=state["completed"][batch.batch_id]
            head,blob=engine.tools.runner.artifacts.read(state["models"][result["model_id"]]["artifact"])
            path=Path(tmp)/"current.npz"
            path.write_bytes(blob)
            model=pair_model(trainer.Model).load(path)
            serve=[r for r in batch.samples if r.role=="serve"]
            if any(r.label is not None for r in serve):
                raise ValueError("服务行不应携带答案")
            keys=[r.sample_id for r in serve]
            p=bank.predict(model,np.asarray([r.values for r in serve]))
            if dict(zip(keys,p["current"].tolist(),strict=True))!=result["predictions"]:
                raise ValueError("当轮模型与底层服务记录不同")
            arms={"current":p["current"],"vote":majority(p,6),"vote-local":apply_choices(old.bank(p),old.choices("vote-local")),
                  "observer":router.predict(p,"uniform")}
            round_path=folder/batch.batch_id/"issued.json"
            atomic_json(round_path,{"keys":keys,"weights":head["sha256"],"seen":len(released),
                                    "predictions":{k:v.tolist() for k,v in arms.items()}})
            groups.append(keys)
            for k,v in read(round_path)["predictions"].items():
                forecasts.setdefault(k,[]).extend(v)
            seen=[r for r in batch.samples if r.role in ("explore","adopt")]
            returned=sorted(seen,key=lambda r:digest([17,r.sample_id]))[:32]
            fb=bank.predict(model,np.asarray([r.values for r in returned]))
            labels=[r.label for r in returned]
            router.update(labels,fb)
            old.update(labels,fb)
            selected={r.sample_id for r in returned}
            if selected&released or selected&set(keys):
                raise ValueError("重复反馈或训练/服务相交")
            released.update(selected)
            trace.append({"batch":batch.batch_id,"returned":sorted(selected),"issued_sha256":sha256(round_path.read_bytes()).hexdigest()})
    keys=[k for group in groups for k in group]
    if set(keys)&released or len(keys)!=len(set(keys)):
        raise ValueError("整流服务编号交叉")
    truth={r.sample_id:r.label for r in pkg.validation}
    y=np.asarray([truth[k] for k in keys])
    blocks=np.split(np.arange(len(y)),np.cumsum([len(g) for g in groups])[:-1])
    table={k:scores(y,np.asarray(v),blocks,6) for k,v in forecasts.items()}
    atomic_json(folder/"scores.json",{"split_seed":seed,"feedback_seed":17,"rows":len(keys),"arms":table,"trace":trace})
    print(seed,{k:round(v["pooled"],5) for k,v in table.items()})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=int,choices=(1113,1114,1115))
    main(parser.parse_args().seed)
