"""在原 PRONOSTIA 流上核对同一单条确认规则与整窗确认的差异。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from bearing_local_evidence import local_medians
from compare_calibration_expiry import source_window
from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,remaining_minutes,summary
from harness.models.model_artifacts import ModelArtifacts


def main():
    pkg,_,batches=official("bearing")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    median=float(np.median([r.label for r in pkg.train if r.sample_id in ids]))
    prior=read(OUT/"routing/pronostia/issued.json")
    folder,_=paths("pronostia","bearing")
    samples=read(folder/"state.json")["samples"]
    weights=read(folder/"run.json")["rounds"]
    outputs={k:{} for k in ("nearby","distance","both")}
    versions=ModelArtifacts(folder/"models")
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"B") as trainer:
        p=Path(tmp)/"published.npz"
        for r,rec in enumerate(weights):
            info,blob=versions.read(rec["model_artifact"])
            if info["sha256"]!=prior["history"][r]["weights"]:
                raise ValueError("参照父版权重不同")
            p.write_bytes(blob)
            model=trainer.Model.load(p)
            e,c,s=source_window(samples,r)
            use=prior["history"][r]["choices"]["median-confirm"]["chosen"]=="median"
            for arm,v in local_medians(model,median,e,c,s,use).items():
                outputs[arm].update(zip((row["sample_id"] for row in s),v.tolist(),strict=True))
    outputs.update({k:prior["predictions"][k] for k in ("current","median","median-confirm","paper","stump")})
    path=OUT/"routing/pronostia-local/issued.json"
    atomic_json(path,{"predictions":outputs,"source_sha256":sha256((OUT/"routing/pronostia/issued.json").read_bytes()).hexdigest()})
    preds=read(path)["predictions"]
    table=summary("bearing",remaining_minutes("full-test-manifest.json"),{"parent":preds["current"],**preds},
                  {k:samples[k]["unit_id"] for k in preds["current"]})
    atomic_json(path.with_name("scores.json"),{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
    print({k:round(v["score"],2) for k,v in table.items()})


if __name__=="__main__":
    main()
