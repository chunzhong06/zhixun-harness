"""在预选的新波形上从实际部署的单条 JSON 入口发放，并用设备档案终点复算分钟。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,summary
from engineering.cli.workflow.bearing_phase_predict import infer
from harness.data.official_bearing import CHANNELS,MEASURES
from score_unseen_xjtu_devices import collect


def main():
    data=collect(second_offset=True)
    output=OUT/"life-scale/fresh-windows"
    folder=OUT/"life-scale/live/pronostia"
    keys=[r["member"] for r in data["rows"]]
    x=np.array([r["features"] for r in data["rows"]])
    def requests():
        for v in x:
            yield {"task":"regression","inputs":{"features":{c:dict(zip(MEASURES,v[i*17:(i+1)*17].tolist(),strict=True))
                                                            for i,c in enumerate(CHANNELS)}}}
    p={"phase":{k:r["outputs"]["value"] for k,r in zip(keys,infer(folder,requests()),strict=True)}}
    pkg,_,batches=official("bearing")
    median=float(np.median([r.label for r in batches[0].samples if r.role=="fit"]))
    source_tree=frozen("bearing")[0]["catboost"]
    root=OUT/"completed-calibration/femto/all/models"
    weights=root/"hidden-ridge.npz"
    if sha256(weights.read_bytes()).hexdigest()!=read(root/"training.json")["models"]["hidden-ridge"]:
        raise ValueError("校准权重与原封存不同")
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        path=Path(tmp)/"original.npz"
        path.write_bytes(pkg.checkpoint)
        q={"original":trainer.Model.load(path).predict(x),"tree":tree_values(source_tree,"bearing",x),
           "median":np.full(len(x),median),"femto-calibrated":trainer.Model.load(weights).predict(x)}
        p.update({name:dict(zip(keys,values.tolist(),strict=True)) for name,values in q.items()})
    file=output/"issued.json"
    atomic_json(file,{"predictions":p,"run_sha256":sha256((folder/"run.json").read_bytes()).hexdigest(),
                "source_sha256":sha256((ROOT/"data/public/xjtu-sy/fifth-offset6.json").read_bytes()).hexdigest()})
    actual=read(file)["predictions"]
    units={k:Path(k).parent.name for k in keys}
    truth={k:data["ends"][u]-int(Path(k).stem) for k,u in units.items()}
    result=summary("bearing",truth,{"parent":actual["original"],**actual},units)
    atomic_json(file.with_name("scores.json"),{"arms":result,"issued_sha256":sha256(file.read_bytes()).hexdigest()})
    print(len(keys),{k:(round(t["score"],2),round(t["worst_group"],2)) for k,t in result.items()})


if __name__=="__main__":
    main()
