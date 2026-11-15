"""把绝对寿命目标拆成总时长和余下比例，在相同单条输入与整台留出上比较。"""
from hashlib import sha256
from time import process_time

import numpy as np
from sklearn.base import clone
from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import atomic_json, digest, official, read
from harness.models.training_tools import SklearnBackend
from score_life_training_weights import cohorts, score

DEST=OUT/"life-scale"


def fit(rows,template,save=None):
    if any(r.role!="fit" for r in rows):
        raise ValueError("拟合仅允许源训练行")
    x=np.array([r.values for r in rows])
    y=np.array([r.label for r in rows],dtype=float)
    life=np.array([r.label+r.time_end-1 for r in rows],dtype=float)
    groups=np.array([r.unit_id for r in rows])
    per=[np.unique(life[groups==u]) for u in np.unique(groups)]
    if any(len(q)!=1 or q[0]<=0 for q in per):
        raise ValueError("同一已完设备的原寿命不一致")
    scale=float(np.median(np.ravel(per)))
    med=float(np.median(y))
    models={}
    info={}
    backend=SklearnBackend(execution_enabled=True)
    for name,target in (("direct",y),("second",y),("phase",y/life),("scale",life),("log-scale",np.log(life))):
        obj=clone(template)
        if name=="second":
            obj.set_params(predictor__random_seed=47)
        stamp=process_time()
        obj.fit(x,target)
        cpu=process_time()-stamp
        if save is not None:
            save.mkdir(parents=True,exist_ok=True)
            blob=backend.dump(obj)
            path=save/(name+".joblib")
            path.write_bytes(blob)
            obj=backend.load(path.read_bytes())
            info[name]={"cpu_seconds":cpu,"sha256":sha256(blob).hexdigest()}
        models[name]=obj
    if save is not None:
        atomic_json(save/"training.json",{"fit_ids":digest([r.sample_id for r in rows]),"arms":info})
    def predict(values):
        p={k:tree_values(v,"bearing",values) for k,v in models.items()}
        fraction=np.clip(p["phase"],0,1)
        estimated=p["scale"]
        return {"direct":p["direct"], "two-direct":.5*(p["direct"]+p["second"]), "scale-only":estimated/2,
                "phase-fixed":fraction*scale, "scale-phase":fraction*estimated, "log-scale-phase":fraction*np.exp(p["log-scale"]),
                "median":np.full(len(values),med), "half-life":np.full(len(values),scale/2)}
    return predict


def main():
    pkg,_,batches=official("bearing")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    small=[r for r in pkg.train if r.sample_id in ids]
    tree=frozen("bearing")[0]["catboost"]
    forecast=fit(small,tree,DEST/"bearing/models")
    result={c[0]:score("bearing",c,forecast,DEST,"direct") for c in cohorts("bearing",pkg)}
    atomic_json(DEST/"bearing/scores.json",result)
    for k,t in result.items():
        print(k,{a:round(t[a]["score"],2) for a in ("direct","two-direct","phase-fixed","scale-phase","log-scale-phase","scale-only")})
    issued={}
    for unit in sorted({r.unit_id for r in small}):
        train=[r for r in small if r.unit_id!=unit]
        rows=[r for r in pkg.train if r.unit_id==unit]
        p=fit(train,tree)([r.values for r in rows])
        issued[unit]={"ids":[r.sample_id for r in rows],"predictions":{k:v.tolist() for k,v in p.items()}}
    f=DEST/"bearing/held-source/issued.json"
    atomic_json(f,issued)
    truth={r.sample_id:r for r in pkg.train}
    table={}
    for unit,record in read(f).items():
        p=record["predictions"]
        rows=[truth[k] for k in record["ids"]]
        life=rows[0].label+rows[0].time_end-1
        y=np.array([r.label for r in rows])
        table[unit]={k:float(np.sqrt(np.mean((v-y)**2))) for k,v in p.items()}
        table[unit]["total-predicted-mae"]=float(np.abs(2*np.array(p["scale-only"])-life).mean())
        table[unit]["total-median-mae"]=float(abs(2*p["half-life"][0]-life))
    atomic_json(f.with_name("scores.json"),{"devices":table,"issued_sha256":sha256(f.read_bytes()).hexdigest()})


if __name__=="__main__":
    main()
