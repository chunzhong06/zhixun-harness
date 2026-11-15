"""用已封存的轴承交接检验支持范围，单列同时维护两个候选的实际构建成本。"""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, delivered, official, read, reference_folders, remaining_minutes, summary
from harness.models.scoped_calibration import closest, split_indices


def choose(model, bank, radius, ids, x, old, new):
    mask = np.zeros(len(ids),dtype=bool) if bank is None else closest(model.normalized(x),bank)<=radius
    return {i:new[i] if yes else old[i] for i,yes in zip(ids,mask,strict=True)},int(mask.sum())


def compare(cohort):
    fixed,_ = reference_folders(cohort,"mature")
    source = BASE/"readout-live"
    path = source/"public"/"bearing" if cohort=="bearing" else source/cohort
    folder = source/"scope"/cohort
    rows = read(fixed/"state.json")["samples"]
    package,_,_ = official("bearing")
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    base, new = delivered(fixed),delivered(path)
    trace = read(path/"run.json")["rounds"]
    protected, events = {},[]
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"B") as trainer:
        p=Path(tmp)/"parent.npz"
        p.write_bytes(package.checkpoint)
        model=trainer.Model.load(p)
        bank, radius=None,None
        for r,record in enumerate(trace):
            if record["current"]!=record["selected"]:
                observations=[s for s in rows.values() if s["available_at"]==r and s["role"]=="explore"]
                a,b=split_indices(observations)
                if not a or not b:
                    raise ValueError("采纳窗口没有可估计半径的独立分片")
                values=model.normalized(np.array([row["values"] for row in observations]))
                bank=values[a]
                radius=float(np.quantile(closest(values[b],bank),.95,method="higher"))
            ids=[i for i,row in rows.items() if row["available_at"]==r and row["role"]=="serve"]
            values=np.array([rows[i]["values"] for i in ids])
            subset,count=choose(model,bank,radius,ids,values,base,new)
            protected.update(subset)
            events.append({"round":r+1,"inside":count,"total":len(ids),"radius":radius})
        streams={"parent":base,"bias":new,"scoped":protected}
        sealed=folder/"issued.json"
        atomic_json(sealed,{"predictions":streams,"range":events})
        truth = ({r.sample_id:r.label for r in package.validation} if cohort=="bearing"
                 else remaining_minutes("full-test-manifest.json"))
        scores=summary("bearing",truth,read(sealed)["predictions"],{i:s["unit_id"] for i,s in rows.items()})
        ref=read(source/"scores"/(cohort+".json"))["arms"]
        builds={k:ref[k]["physical_builds"] for k in ("fixed","bias")}
        atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":scores,
                                         "separate_route_builds":builds})
        print(cohort,{k:round(v["score"],3) for k,v in scores.items()},flush=True)
        if cohort=="pronostia":
            kaist=[r for r in read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"] if "features" in r]
            ids=["KAIST/"+r["file"] for r in kaist]
            sealed_source=source/"transfer"/"bearing"/"issued.json"
            old=read(sealed_source)["predictions"]
            chosen,count=choose(model,bank,radius,ids,np.array([r["features"] for r in kaist]),old["fixed"],old["readout-bias"])
            sealed=folder/"kaist-issued.json"
            atomic_json(sealed,{"predictions":{"parent":old["fixed"],"bias":old["readout-bias"],"scoped":chosen},
                                "upstream_sha256":sha256(sealed_source.read_bytes()).hexdigest(),"inside":count})
            raw=read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
            end=datetime.fromisoformat(raw[-1]["time"])
            truth={i:(end-datetime.fromisoformat(r["time"])).total_seconds()/60 for i,r in zip(ids,kaist,strict=True)}
            scores=summary("bearing",truth,read(sealed)["predictions"],dict.fromkeys(ids,"KAIST"))
            atomic_json(folder/"kaist-scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":scores})
            print("kaist",count,{k:round(v["score"],3) for k,v in scores.items()},flush=True)


if __name__=="__main__":
    for cohort in ("bearing","pronostia"):
        compare(cohort)
