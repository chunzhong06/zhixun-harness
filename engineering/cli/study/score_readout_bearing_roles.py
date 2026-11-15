"""在三台公开轴承的全部角色分配上核查冻结输出层的实际误差。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,frozen,tree_values
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,summary


def main():
    pkg,_,_=official("bearing")
    base=OUT/"teacher-readout/bearing"
    path=base/"models/teacher-ridge.npz"
    if sha256(path.read_bytes()).hexdigest()!=read(base/"training.json")["runs"]["teacher-ridge"]["sha256"]:
        raise ValueError("蒸馏权重变化")
    history=OUT/"routing/bearing-rotations"
    source=history/"issued.json"
    if sha256(source.read_bytes()).hexdigest()!=read(history/"scores.json")["issued_sha256"]:
        raise ValueError("角色分配参照内容变化")
    folds=read(source)["streams"]
    rows={r.sample_id:r for r in pkg.validation}
    ids=list(rows)
    x=np.asarray([rows[i].values for i in ids])
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"B") as trainer:
        result=trainer.Model.load(path).predict(x)
        original=trainer.Model.load(base/"models/original.npz").predict(x)
    trees,_=frozen("bearing")
    fixed=tree_values(trees["catboost"],"bearing",x)
    pool={n:dict(zip(ids,v.tolist(),strict=True)) for n,v in
          (("teacher-ridge",result),("original",original),("tree",fixed),("half",.5*(result+original)))}
    out=base/"roles"
    sealed=out/"issued.json"
    forecast={}
    units={}
    source_ids={}
    for role_map,fold in folds.items():
        old=fold["predictions"]
        for k in old["current"]:
            key=f"{role_map}/{k}"
            units[key]=rows[k].unit_id
            source_ids[key]=k
            if units[key]!=fold["roles"]["serve"]:
                raise ValueError("服务设备身份不符")
            for arm,p in {**old,**pool}.items():
                forecast.setdefault(arm,{})[key]=p[k]
    if any(list(source_ids.values()).count(k)!=2 for k in ids):
        raise ValueError("每台轴承应在两种反馈排列中提供服务")
    atomic_json(sealed,{"weights":sha256(path.read_bytes()).hexdigest(),"predictions":forecast})
    saved=read(sealed)["predictions"]
    truth={key:rows[k].label for key,k in source_ids.items()}
    table=summary("bearing",truth,{"parent":saved["current"],**saved},units)
    atomic_json(out/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table})
    print({k:(round(r["score"],2),round(r["worst_group"],2)) for k,r in table.items()})


if __name__=="__main__":
    main()
