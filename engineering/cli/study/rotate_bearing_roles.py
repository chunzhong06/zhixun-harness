"""轮换官方验证轴承的三个用途；每个排列独立维护，同一服务设备的结果成对计分。"""
from dataclasses import asdict
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import old_step
from compare_fixed_tree_references import OUT
from compare_group_objectives import ROOT,_trainer,atomic_json,metric,official,read
from engineering.cli.study.run_rul_series import SERIES,windows
from harness.models.life_reference import choose
from bearing_local_evidence import local_medians


def main(local=False):
    pkg,checksum,batches=official("bearing")
    fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    train=[r for r in pkg.train if r.sample_id in fit_ids]
    fit=np.asarray([r.values for r in train]),np.asarray([r.label for r in train])
    median=float(np.median(fit[1]))
    devices=sorted({r.unit_id for r in pkg.validation})
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    original=read(OUT/"routing/bearing/issued.json")["predictions"]
    expected={s.role:s.unit_id for b in batches for s in b.samples if s.role!="fit"}
    outputs={}
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        p=Path(tmp)/"root.npz"
        p.write_bytes(pkg.checkpoint)
        for assignment in permutations(devices):
            role_units=dict(zip(("explore","adopt","serve"),assignment,strict=True))
            replay=windows(pkg,{v:k for k,v in role_units.items()},tuple(train),read(SERIES),"bearing")
            models={k:trainer.Model.load(p) for k in ("current","fixed")}
            cache={k:None for k in models}
            preds={k:{} for k in ("current","fixed","median","mean","median-confirm")}
            if local:
                preds.update({k:{} for k in ("nearby","distance","both")})
            events=[]
            for b in replay:
                e,c,s=([asdict(v) for v in b.samples if v.role==kind] for kind in ("explore","adopt","serve"))
                if not s:
                    events.append({"window":b.ordinal,"service_rows":0})
                    continue
                trace={}
                for k,policy in (("current","old"),("fixed","fixed")):
                    models[k],cache[k],trace[k],v=old_step("bearing",policy,models[k],fit,e,c,s,cache[k])
                    preds[k].update(v)
                def options(rows):
                    base=models["current"].predict(np.array([v["values"] for v in rows]))
                    return {"current":base,"median":np.full(len(rows),median)}
                chosen=choose(options(e),options(c),[v["label"] for v in e],[v["label"] for v in c],"median-confirm")
                if local:
                    for arm,values in local_medians(models["current"],median,e,c,s,chosen["chosen"]=="median").items():
                        preds[arm].update(zip((row["sample_id"] for row in s),values.tolist(),strict=True))
                for row in s:
                    i=row["sample_id"]
                    preds["median"][i]=median
                    preds["mean"][i]=float(fit[1].mean())
                    preds["median-confirm"][i]=median if chosen["chosen"]=="median" else preds["current"][i]
                events.append({"window":b.ordinal,"service_rows":len(s),"decision":chosen,"updates":trace})
            if role_units==expected:
                for name in ("current","fixed","median-confirm"):
                    v=preds[name]
                    if original[name].keys()!=v.keys() or max(abs(y-original[name][i]) for i,y in v.items())>1e-6:
                        raise ValueError(f"原排列未复现 {name}")
            outputs["/".join(assignment)]={"roles":role_units,"predictions":preds,"history":events}
    folder=OUT/"routing"/("bearing-local" if local else "bearing-rotations")
    path=folder/"issued.json"
    atomic_json(path,{"package":checksum,"median":median,"streams":outputs})
    truth={r.sample_id:r.label for r in pkg.validation}
    scores,per_case={},{}
    for name,stream in read(path)["streams"].items():
        records={}
        for arm,p in stream["predictions"].items():
            value=metric("bearing",truth,p)
            records[arm]=value
            scores.setdefault(arm,{}).setdefault(stream["roles"]["serve"],[]).append((len(p),value))
        per_case[name]=records
    metrics={}
    for arm,devices in scores.items():
        dev={k:float(np.sqrt(np.mean([v**2 for _,v in group]))) for k,group in devices.items()}
        metrics[arm]={"pooled":float(np.sqrt(sum(n*v**2 for g in devices.values() for n,v in g)/sum(n for g in devices.values() for n,_ in g))),
                      "device_equal":float(np.sqrt(np.mean([v**2 for v in dev.values()]))),"worst":max(dev.values()),"devices":dev}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":metrics,"cases":per_case})
    print({k:tuple(round(v[i],2) for i in ("pooled","device_equal","worst")) for k,v in metrics.items()})


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--local",action="store_true")
    main(parser.parse_args().local)
