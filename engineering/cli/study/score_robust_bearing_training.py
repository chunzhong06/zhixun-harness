"""同一起点与 Adam 历史，只改变逐行寿命损失梯度；整体缩小梯度作为归因对照。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np
from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, official
from score_life_training_weights import cohorts, score, weighted_median


def epoch(model, z, target, threshold, mode):
    m=model.meta
    rng=np.random.default_rng(np.random.SeedSequence([m["seed"],m["epochs"]]))
    order=rng.permutation(len(z))
    for start in range(0,len(z),m["batch_size"]):
        idx=order[start:start+m["batch_size"]]
        batch=z[idx]
        h,logits=model.forward(batch)
        e=logits[:,0]-target[idx]
        bounded=np.clip(e,-threshold,threshold)
        if mode=="whole-batch":
            bounded=e*(np.linalg.norm(bounded)/max(np.linalg.norm(e),1e-12))
        delta=(2*bounded/len(idx))[:,None]
        hidden_delta=(delta@model.params["w2"].T)*(1-h**2)
        gradients={"w1":batch.T@hidden_delta,"b1":hidden_delta.sum(axis=0),"w2":h.T@delta,"b2":delta.sum(axis=0)}
        norm=np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
        m["steps"]+=1
        for name,g in gradients.items():
            g*=min(1.,5./max(norm,1e-12))
            model.first[name]=.9*model.first[name]+.1*g
            model.second[name]=.999*model.second[name]+.001*g*g
            momentum=model.first[name]/(1-.9**m["steps"])
            variance=model.second[name]/(1-.999**m["steps"])
            model.params[name]-=m["learning_rate"]*momentum/(np.sqrt(variance)+1e-8)
    m["epochs"]+=1
    model.validate()


def main():
    task="bearing"
    pkg,_,batches=official(task)
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    little=[r for r in pkg.train if r.sample_id in ids]
    median=float(np.median([r.label for r in little]))
    tree=frozen(task)[0]["catboost"]
    root=OUT/"robust-training"
    models,receipts={},{}
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    out=root/"bearing/models"
    out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"B") as trainer:
        base=Path(tmp)/"parent.npz"
        base.write_bytes(pkg.checkpoint)
        for pool,rows in (("small",little),("full",pkg.train)):
            if any(r.role!="fit" for r in rows):
                raise ValueError("训练池包含非训练行")
            x=np.array([r.values for r in rows])
            y=np.array([r.label for r in rows])
            _,inverse,counts=np.unique([r.unit_id for r in rows],return_inverse=True,return_counts=True)
            parent=trainer.Model.load(base)
            threshold=1.345*weighted_median(np.abs(parent.predict(x)-y),1/counts[inverse])
            z=parent.normalized(x)
            target=(y-parent.meta["target_mean"])/parent.meta["target_scale"]
            for mode in ("direct","huber","whole-batch"):
                m=trainer.Model.load(base)
                cpu=0.
                for n in range(1,41):
                    begin=process_time()
                    if mode=="direct":
                        m.fit(x,y,1)
                    else:
                        epoch(m,z,target,threshold/m.meta["target_scale"],mode)
                    cpu+=process_time()-begin
                    if n not in (5,40):
                        continue
                    arm=f"{pool}-{mode}-{n}"
                    with TemporaryDirectory(dir=out) as temp:
                        path=Path(temp)/(arm+".npz")
                        m.save(path)
                        dest=out/path.name
                        path.replace(dest)
                    models[arm]=trainer.Model.load(dest)
                    receipts[arm]={"cpu_seconds":cpu,"rows":len(rows),"threshold_minutes":threshold,
                                   "steps":m.meta["steps"]-parent.meta["steps"],"fit_ids":digest([r.sample_id for r in rows]),
                                   "weights":sha256(dest.read_bytes()).hexdigest()}
        atomic_json(out/"training.json",{"parent":pkg.checkpoint_sha,"arms":receipts})
        def predict(x):
            return {**{k:m.predict(x) for k,m in models.items()},"source-tree":tree_values(tree,task,x),"median":np.full(len(x),median)}
        result={c[0]:score(task,c,predict,root,"source-tree") for c in cohorts(task,pkg)}
    atomic_json(root/"bearing/scores.json",result)
    for n,t in result.items():
        print(n,{k:round(t[k]["score"],2) for k in ("current","small-direct-40","small-huber-40","full-direct-40","full-huber-40","source-tree","median")})


if __name__=="__main__":
    main()
