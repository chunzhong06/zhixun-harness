"""从相同的官方寿命检查点以真实标签或固定树的预测续训，按原始输入逐条评估。"""
from argparse import ArgumentParser
from hashlib import sha256
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,frozen,paths,tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, official, read, remaining_cycles, remaining_minutes, summary
from score_tree_reference_transfer import suites


def train(task,trainer,pkg,rows,teacher,folder):
    x=np.asarray([r.values for r in rows])
    y=np.asarray([r.label for r in rows])
    target=tree_values(teacher,task,x)
    shuffle=np.random.default_rng(17).permutation(len(y))
    objectives={"ordinary":y,"mix":.5*(target+y),"teacher":target,"shuffled":.5*(target[shuffle]+y)}
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"original.npz"
    path.write_bytes(pkg.checkpoint)
    base=trainer.Model.load(path)
    models,details={"original":base},{}
    for arm,objective in objectives.items():
        child=trainer.Model.load(path)
        tic=process_time()
        for epoch in range(1,5):
            child.fit(x,objective,1)
            if epoch not in (1,4):
                continue
            name=f"{arm}-{epoch}"
            artifact=folder/(name+".npz")
            child.save(artifact)
            model=trainer.Model.load(artifact)
            if (model.meta["steps"]-base.meta["steps"]!=epoch*ceil(len(x)/base.meta["batch_size"])
                    or model.meta["epochs"]!=base.meta["epochs"]+epoch
                    or not np.array_equal(model.scale,base.scale) or not np.array_equal(model.mean,base.mean)
                    or any(model.meta[k]!=base.meta[k] for k in ("target_mean","target_scale"))):
                raise ValueError("工业模型状态或训练步数不符")
            models[name]=model
            details[name]={"steps":model.meta["steps"]-base.meta["steps"],"seconds":process_time()-tic,
                           "sha256":sha256(artifact.read_bytes()).hexdigest(),"target":digest(objective.tolist())}
    return models,details,float(np.sqrt(np.mean((target-y)**2)))


def train_readout(task,trainer,pkg,rows,teacher,folder):
    from harness.models.readout_training import fit_readout
    x=np.asarray([r.values for r in rows])
    y=np.asarray([r.label for r in rows])
    target=tree_values(teacher,task,x)
    shuffle=np.random.default_rng(17).permutation(len(y))
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"original.npz"
    path.write_bytes(pkg.checkpoint)
    parent=trainer.Model.load(path)
    models={"original":parent}
    details={}
    for name,objective in (("true-ridge",y),("teacher-ridge",target),("mix-ridge",.5*(y+target)),
                           ("shuffled-ridge",.5*(y+target[shuffle]))):
        child=trainer.Model.load(path)
        tic=process_time()
        fit_readout(child,x,objective,"ridge")
        model_path=folder/(name+".npz")
        child.save(model_path)
        models[name]=child=trainer.Model.load(model_path)
        if (child.meta["steps"]!=parent.meta["steps"]+1 or
            any(not np.array_equal(child.params[k],parent.params[k]) for k in ("w1","b1"))):
            raise ValueError("冻结的隐藏层或输出层步数不符")
        details[name]={"sha256":sha256(model_path.read_bytes()).hexdigest(),"seconds":process_time()-tic,
                       "target_sha256":digest(objective.tolist()),
                       "source_teacher_rmse":float(np.sqrt(np.mean((child.predict(x)-target)**2))),
                       "source_original_teacher_rmse":float(np.sqrt(np.mean((parent.predict(x)-target)**2)))}
    return models,details,float(np.sqrt(np.mean((target-y)**2)))


def study(task,dense=False,readout=False):
    pkg,checksum,batches=official(task)
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=[r for r in pkg.train if r.sample_id in ids]
    trees,receipts=frozen(task)
    teacher=trees["catboost"]
    if receipts["catboost"]["fit"]!=digest([r.sample_id for r in rows]):
        raise ValueError("教师训练行与本实验不同")
    if dense:
        if any(r.role!="fit" or r.label is None for r in pkg.train):
            raise ValueError("扩展归属只允许原包完整公开训练集")
        rows=list(pkg.train)
    folder=OUT/("teacher-readout" if readout else ("distillation-full" if dense else "distillation"))/task
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        fit=train_readout if readout else train
        models,details,source_score=fit(task,trainer,pkg,rows,teacher,folder/"models")
        atomic_json(folder/"training.json",{"parent":pkg.checkpoint_sha,"source":checksum,"teacher":receipts["catboost"]["sha256"],
                    "teacher_query_rmse":source_score,"teacher_training_rows":len(ids),
                    "training_samples":len(rows),"fit_digest":digest([r.sample_id for r in rows]),"runs":details})
        def evaluate(suite,keys,x,groups,truth,benchmarks):
            outputs={arm:dict(zip(keys,model.predict(x).tolist(),strict=True)) for arm,model in models.items()}
            outputs["fixed-tree"]=dict(zip(keys,tree_values(teacher,task,x).tolist(),strict=True))
            previous=read(benchmarks)["predictions"]
            if max(abs(v-previous["catboost-only"][i]) for i,v in outputs["fixed-tree"].items())>1e-8:
                raise ValueError("冻结树或目标输入发生变化")
            outputs.update({k:previous[k] for k in ("current","paper","fixed")})
            path=folder/suite/"issued.json"
            atomic_json(path,{"inputs":digest([keys,x.tolist()]),"predictions":outputs})
            p=read(path)["predictions"]
            table=summary(task,truth(),{"parent":p["original"],**p},groups)
            atomic_json(path.with_name("scores.json"),{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
            print(suite,{k:round(table[k]["score"],2) for k in ("original","fixed-tree",*details)},flush=True)
        answers={r.sample_id:r.label for r in pkg.validation}
        for cohort in ((task,"pronostia") if task=="bearing" else (task,"phm08")):
            current,_=paths(cohort,task)
            samples=read(current/"state.json")["samples"]
            serve={k:r for k,r in samples.items() if r["role"]=="serve"}
            if any(r["label"] is not None for r in serve.values()):
                raise ValueError("服务数据携带答案")
            truth=(lambda:answers) if cohort==task else (remaining_cycles if task=="engine" else lambda:remaining_minutes("full-test-manifest.json"))
            evaluate(cohort,list(serve),np.asarray([r["values"] for r in serve.values()]),{k:r["unit_id"] for k,r in serve.items()},truth,OUT/cohort/"issued.json")
        for suite,windows,groups,truth in suites(task,False):
            items=[s for _,block in sorted(windows.items()) for s in block]
            keys,x=zip(*items,strict=True)
            bench=(OUT/"routing" if suite=="new40" else OUT)/"transfer"/suite/"issued.json"
            evaluate(suite,list(keys),np.asarray(x),groups,truth,bench)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("bearing","engine"))
    parser.add_argument("--dense",action="store_true")
    parser.add_argument("--readout",action="store_true",help="在完整公开训练池上闭式拟合输出层")
    args=parser.parse_args()
    study(args.task,args.dense or args.readout,args.readout)
