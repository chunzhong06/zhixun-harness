"""在原回归头精确表达固定训练寿命及其等权融合，保存可重载的赛方单模型。"""
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,remaining_cycles,remaining_minutes,summary
from score_tree_reference_transfer import suites


def study(task):
    pkg,_,batches=official(task)
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    y_fit=[r.label for r in pkg.train if r.sample_id in ids]
    values={"median":float(np.median(y_fit)),"mean":float(np.mean(y_fit))}
    base=OUT/"teacher-readout"/task
    out=OUT/"constant-weights"/task
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        models={}
        receipt={}
        work=out/"models"
        work.mkdir(parents=True,exist_ok=True)
        for source in ("original","teacher-ridge"):
            weights=base/"models"/(source+".npz")
            expected=(pkg.checkpoint_sha if source=="original" else read(base/"training.json")["runs"][source]["sha256"])
            if sha256(weights.read_bytes()).hexdigest()!=expected:
                raise ValueError("起始权重与记录不符")
            for kind,constant in values.items():
                for fraction in ((0.,.5) if source=="original" else (.5,)):
                    name=kind if fraction==0 else f"{kind}-{source}-half"
                    model=trainer.Model.load(weights)
                    target=(constant-model.meta["target_mean"])/model.meta["target_scale"]
                    model.params["w2"]*=fraction
                    model.params["b2"][:]=fraction*model.params["b2"]+(1-fraction)*target
                    path=work/(name+".npz")
                    model.save(path)
                    models[name]=trainer.Model.load(path)
                    receipt[name]={"sha256":sha256(path.read_bytes()).hexdigest(),"source_sha256":expected,
                        "constant":constant,"predictor_fraction":fraction,"fit_rows":len(y_fit),"gradient_steps":0}
        atomic_json(out/"models.json",receipt)
        def evaluate(name,keys,x,groups,truth,refs):
            path=refs/"issued.json"
            if sha256(path.read_bytes()).hexdigest()!=read(refs/"scores.json")["issued_sha256"]:
                raise ValueError("已有对照与封存发放不符")
            previous=read(path)["predictions"]
            retained={k:p for k,p in previous.items() if k in ("current","paper","teacher-ridge","fixed-tree","tree","original")}
            if any(set(keys)!=set(p) for p in retained.values()):
                raise ValueError("对照与新权重的样本不符")
            for arm,model in models.items():
                v=model.predict(np.asarray(x))
                if arm in values and not np.allclose(v,values[arm],rtol=0,atol=1e-10):
                    raise ValueError("赛方类重载后未准确交付常数")
                retained[arm]=dict(zip(keys,v.tolist(),strict=True))
            sealed=out/name/"issued.json"
            atomic_json(sealed,{"predictions":retained,"weights":{k:v["sha256"] for k,v in receipt.items()}})
            p=read(sealed)["predictions"]
            table=summary(task,truth(),{"parent":p["current"],**p},groups)
            atomic_json(sealed.with_name("scores.json"),{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table})
            return {k:(r["score"],r["worst_group"]) for k,r in table.items()}
        result={}
        for name in (task,"phm08" if task=="engine" else "pronostia"):
            folder,_=paths(name,task)
            samples=read(folder/"state.json")["samples"]
            keys=[k for k,v in samples.items() if v["role"]=="serve"]
            def truth():
                if name==task:
                    return {r.sample_id:r.label for r in pkg.validation}
                return remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
            result[name]=evaluate(name,keys,[samples[k]["values"] for k in keys],{k:samples[k]["unit_id"] for k in keys},truth,base/name)
        for name,windows,groups,truth in suites(task,False):
            keys,rows=zip(*(r for _,b in sorted(windows.items()) for r in b),strict=True)
            result[name]=evaluate(name,keys,rows,groups,truth,base/name)
        if task=="bearing":
            source=read(ROOT/"data/public/xjtu-sy/fifth-devices.json")
            ids=[r["member"] for r in source["rows"]]
            g={k:Path(k).parent.name for k in ids}
            result["new-xjtu"]=evaluate("new-xjtu",ids,[r["features"] for r in source["rows"]],g,
                lambda:{k:source["ends"][g[k]]-int(Path(k).stem) for k in ids},base/"new-xjtu")
        return result


if __name__=="__main__":
    results={t:study(t) for t in ("bearing","engine")}
    atomic_json(OUT/"constant-weights/scores.json",results)
    for task,cohorts in results.items():
        for name,r in cohorts.items():
            print(name,{k:round(r[k][0],2) for k in ("current","median","mean","median-original-half","median-teacher-ridge-half","mean-teacher-ridge-half")})
