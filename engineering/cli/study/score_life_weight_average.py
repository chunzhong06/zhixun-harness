"""在相同起点的赛方寿命权重之间插值，保存单模型后与预测平均作配对。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,remaining_cycles,remaining_minutes,summary
from score_tree_reference_transfer import suites


def evaluate(task):
    pkg,_,_=official(task)
    source=OUT/"distillation-full"/task
    target=OUT/"weight-average"/task
    manifest=read(source/"training.json")
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        def load(name):
            path=source/"models"/(name+".npz")
            expected=manifest["parent"] if name=="original" else manifest["runs"][name]["sha256"]
            if sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError("源权重与训练回执不符")
            return trainer.Model.load(path)
        teacher=load("teacher-1")
        variants={}
        receipts={}
        for parent in ("original","ordinary-1"):
            anchor=load(parent)
            child=load("teacher-1")
            if (not np.array_equal(anchor.mean,teacher.mean) or not np.array_equal(anchor.scale,teacher.scale)
                or any(anchor.meta[k]!=teacher.meta[k] for k in ("target_scale","target_mean"))):
                raise ValueError("输入或目标标准化不同")
            tic=perf_counter()
            for k in child.params:
                child.params[k]=.5*(anchor.params[k]+teacher.params[k])
            path=target/"models"/(parent+"-half.npz")
            path.parent.mkdir(parents=True,exist_ok=True)
            child.save(path)
            child=trainer.Model.load(path)
            if any(not np.array_equal(child.params[k],.5*(anchor.params[k]+teacher.params[k])) for k in child.params):
                raise ValueError("保存的参数与等权插值不符")
            variants[parent]=(anchor,child)
            receipts[parent]={"sha256":sha256(path.read_bytes()).hexdigest(),"base":parent,
                "optimizer_source":"teacher-1","seconds":perf_counter()-tic}
        atomic_json(target/"models.json",receipts)
        def score(name,keys,values,groups,answers):
            folder=source/name
            path=folder/"issued.json"
            if sha256(path.read_bytes()).hexdigest()!=read(folder/"scores.json")["issued_sha256"]:
                raise ValueError("旧预测内容不同")
            old=read(path)["predictions"]
            if set(keys)!=set(old["original"]):
                raise ValueError("旧模型与本次服务样本不一致")
            new={k:old[k] for k in ("original","ordinary-1","teacher-1","fixed-tree","current","paper")}
            x=np.asarray(values)
            t=teacher.predict(x)
            if not np.allclose(t,[old["teacher-1"][k] for k in keys],rtol=0,atol=1e-8):
                raise ValueError("继承的权重没有复现原预测")
            for parent,(anchor,model) in variants.items():
                for label,v in (("params",model.predict(x)),("outputs",.5*(anchor.predict(x)+t))):
                    new[f"{label}-{parent}"]=dict(zip(keys,v.tolist(),strict=True))
            out=target/name/"issued.json"
            atomic_json(out,{"source_sha256":sha256(path.read_bytes()).hexdigest(),"predictions":new})
            sealed=read(out)["predictions"]
            table=summary(task,answers(),{"parent":sealed["current"],**sealed},groups)
            atomic_json(out.with_name("scores.json"),{"issued_sha256":sha256(out.read_bytes()).hexdigest(),"arms":table})
            return {k:[r["score"],r["worst_group"]] for k,r in table.items()}
        results={}
        for name in (task,"phm08" if task=="engine" else "pronostia"):
            folder,_=paths(name,task)
            samples=read(folder/"state.json")["samples"]
            keys=[k for k,r in samples.items() if r["role"]=="serve"]
            def truth():
                if name==task:
                    return {r.sample_id:r.label for r in pkg.validation}
                return remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
            results[name]=score(name,keys,[samples[k]["values"] for k in keys],{k:samples[k]["unit_id"] for k in keys},truth)
        for name,rounds,groups,truth in suites(task,False):
            items=[s for _,batch in sorted(rounds.items()) for s in batch]
            keys,values=zip(*items,strict=True)
            results[name]=score(name,keys,values,groups,truth)
        return results


if __name__=="__main__":
    data={task:evaluate(task) for task in ("engine","bearing")}
    atomic_json(OUT/"weight-average/scores.json",data)
    print({n:{k:round(v[0],2) for k,v in r.items()} for t in data.values() for n,r in t.items()},flush=True)
