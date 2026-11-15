"""仅由官方训练池拟合工况与残差，在可见探索和确认资料上发布同一候选。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, delivered, digest, official, read, reference_folders, remaining_cycles, summary

OUT = BASE/"engine-regime-readout"


def assign(settings, center, scale, clusters):
    z = (settings-center)/scale
    return ((z[:,None,:]-clusters[None,:,:])**2).sum(axis=2).argmin(axis=1)


def design(x, spec):
    n = len(x)
    if spec["recipe"]=="bias":
        return np.ones((n,1))
    idx = np.zeros(n,dtype=int)
    if spec["recipe"]=="regime":
        idx = assign(x[:,21:],spec["setting_mean"],spec["setting_scale"],spec["clusters"])
    z = (x[:,:21]-spec["centers"][idx])/spec["scales"][idx]
    return np.column_stack((np.clip(z,-10,10),np.ones(n)))


def fit(model,x,y,recipe):
    s = {"recipe":recipe}
    if recipe!="bias":
        idx = np.zeros(len(x),dtype=int)
        if recipe=="regime":
            settings = x[:,21:]
            center, scale = settings.mean(0), np.maximum(settings.std(0),1e-12)
            with threadpool_limits(limits=1):
                clusters = KMeans(n_clusters=6,n_init=10,random_state=17).fit((settings-center)/scale).cluster_centers_
            idx = assign(settings,center,scale,clusters)
            s.update(setting_mean=center,setting_scale=scale,clusters=clusters)
        sensors = x[:,:21]
        floor = np.maximum(sensors.std(0)*.001,1e-12)
        s["centers"] = np.array([sensors[idx==j].mean(0) for j in range(int(idx.max())+1)])
        s["scales"] = np.array([np.maximum(sensors[idx==j].std(0),floor) for j in range(len(s["centers"]))])
        s["group_sizes"] = np.bincount(idx)
    bank = design(x,s)
    regularizer = np.eye(bank.shape[1])*len(x)*.1
    regularizer[-1,-1] = 0.
    s["coefficients"] = np.linalg.solve(bank.T@bank+regularizer,bank.T@(y-model.predict(x)))
    return s


def corrected(model, x, archive):
    y = model.predict(x)
    return np.maximum(0,y+design(x,archive)@archive["coefficients"])


def compare(cohort):
    folder=OUT/cohort
    package,checksum,batches = official("engine")
    fixed,paper=reference_folders(cohort,"mature")
    fit_ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    archive, = (ROOT/"data/official/engine").glob("*.zip")
    samples=read(fixed/"state.json")["samples"]
    if {i for i,r in samples.items() if r["role"]=="fit"}!=fit_ids or len(fit_ids)!=3968:
        raise ValueError("公开训练点不同")
    train=[r for r in package.train if r.sample_id in fit_ids]
    x,y=np.array([r.values for r in train]),np.array([r.label for r in train])
    streams,decision,models,elapsed = {},{},{},{}
    with TemporaryDirectory() as tmp, ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
        p=Path(tmp)/"parent.npz"
        p.write_bytes(package.checkpoint)
        model=trainer.Model.load(p)
        for arm in ("parent","bias","global","regime"):
            start=perf_counter()
            spec=None if arm=="parent" else fit(model,x,y,arm)
            elapsed[arm]=perf_counter()-start
            models[arm]=None if spec is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in spec.items()}
            stream,events={},[]
            active=False
            for r in range(10):
                event={"round":r+1}
                def group(role):
                    rows=[s for s in samples.values() if s["role"]==role and s["available_at"]==r]
                    if not rows:
                        raise ValueError("缺共同轮次输入")
                    return rows,np.asarray([s["values"] for s in rows])
                if spec is not None and not active:
                    for role in ("explore","adopt"):
                        rows,data=group(role)
                        labels=np.asarray([s["label"] for s in rows])
                        a,b=[float(np.sqrt(np.mean((pred-labels)**2))) for pred in (model.predict(data),corrected(model,data,spec))]
                        event[role]=[a,b]
                        if b>=a-1e-12:
                            break
                    else:
                        active=True
                rows,data=group("serve")
                if any(s["label"] is not None for s in rows):
                    raise ValueError("服务输入携带答案")
                pred=corrected(model,data,spec) if active else model.predict(data)
                if not np.isfinite(pred).all():
                    raise ValueError("服务输出非有限值")
                stream.update(zip([s["sample_id"] for s in rows],map(float,pred),strict=True))
                events.append({**event,"used":active})
            streams[arm],decision[arm]=stream,events
    paths={"fixed":fixed,"mature":paper,"current":BASE/"live-first"/("public" if cohort=="engine" else "")/cohort}
    if cohort!="engine":
        paths["simple"]=fixed.parent/"simple"
    for arm,p in paths.items():
        streams[arm]=delivered(p)
    sealed=folder/"issued.json"
    atomic_json(sealed,{"source":checksum,"parent":sha256(package.checkpoint).hexdigest(),"fit":digest(sorted(fit_ids)),
                        "models":models,"events":decision,"streams":streams})
    data=read(sealed)["streams"]
    truth=remaining_cycles() if cohort=="phm08" else {r.sample_id:r.label for r in package.validation}
    stats=summary("engine",truth,data,{i:s["unit_id"] for i,s in samples.items()})
    for arm,stream in data.items():
        stats[arm]["rounds"]=[float(np.sqrt(np.mean([(v-truth[i])**2 for i,v in stream.items() if samples[i]["available_at"]==r]))) for r in range(10)]
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":stats,"fit_seconds":elapsed})
    print(cohort,{k:round(s["score"],3) for k,s in stats.items()},flush=True)


if __name__=="__main__":
    for cohort in ("engine","phm08"):
        compare(cohort)
