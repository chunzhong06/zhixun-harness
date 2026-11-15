"""用冻结的工况刻度变换训练/服务输入，各臂从同一官方 Adam 状态独立续训。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from score_engine_early_reference import DEST as EARLIER, ROOT, official, read, tree_values, summary, cohorts
from harness.contracts import digest
from harness.models.official_rul_cpu import _trainer
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = EARLIER.parent/"engine-native-condition"
PRIOR = EARLIER.parent/"engine-fixed-comparison"
RECIPES = {"raw-truth":("raw","truth"),"uniform-truth":("uniform","truth"),
           "early-truth":("early","truth"),"early-tree":("early","teacher"),
           "early-half":("early","half")}


def model_input(x,condition,global_scale,parent):
    if condition is None:
        return x
    z = global_scale.transform(condition.transform(x))
    return z*parent.scale+parent.mean


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此条件续训已有完整得分")
        pkg,checksum,batches = official("engine")
        ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        rows = [r for r in pkg.train if r.sample_id in ids]
        record = read(EARLIER/"training.json")
        if [r.sample_id for r in rows]!=record["fit_ids"]:
            raise ValueError("续训行与固定树不一致")
        x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows])
        repair = SourceCombinations(x)
        transforms = {"raw":(None,None)}
        for kind in ("early","uniform"):
            file = EARLIER/(kind+".joblib")
            if sha256(file.read_bytes()).hexdigest()!=record["models"][kind]["sha256"]:
                raise ValueError("源工况刻度与固定实验不一致")
            pipeline = joblib.load(file)
            condition = pipeline.named_steps["condition"]
            z = condition.transform(x)
            transforms[kind] = (condition,StandardScaler().fit(z))
            if kind=="early":
                teacher_y = np.maximum(0,tree_values(pipeline.named_steps["tree"],"engine",z))
        path = DEST/"transforms.joblib"
        joblib.dump(transforms,path)
        transforms = joblib.load(path)
        objectives = {"truth":y,"teacher":teacher_y,"half":.5*(y+teacher_y)}
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        if sha256(archive.read_bytes()).hexdigest()!=checksum:
            raise ValueError("源包身份不符")
        prior = read(PRIOR/"scores.json")
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        with TemporaryDirectory(dir=DEST) as temp, ZipFile(archive) as pack, _trainer(pack,Path(temp),"C") as trainer:
            initial = Path(temp)/"parent.npz"
            initial.write_bytes(pkg.checkpoint)
            parent = trainer.Model.load(initial)
            if any(r.role!="fit" for r in rows):
                raise ValueError("源训练含非训练用途样本")
            states,models = {},{}
            for name,(kind,target) in RECIPES.items():
                model = trainer.Model.load(initial)
                values = model_input(x,*transforms[kind],parent)
                model.fit(values,objectives[target],10)
                file = DEST/(name+".npz")
                model.save(file)
                models[name] = trainer.Model.load(file)
                np.testing.assert_allclose(model.predict(values),models[name].predict(values),rtol=0,atol=1e-9)
                states[name] = {"sha256":sha256(file.read_bytes()).hexdigest(),"steps":model.meta["steps"]}
            atomic_json(DEST/"training.json",{"parent":sha256(pkg.checkpoint).hexdigest(),"models":states,
                        "source_order":digest([r.sample_id for r in rows]),"transform":sha256(path.read_bytes()).hexdigest()})
            results = {}
            for site,keys,features,groups,answers,_ in cohorts("engine",pkg):
                req = prior[site]["request"]
                old = read(PRIOR/"accepted"/(req+".json"))["payload"]
                if digest([site,old])!=req or old["ids"]!=list(keys):
                    raise ValueError("强固定交付身份或服务行序变化")
                mapped = repair.transform(np.asarray(features))
                transformed = {k:model_input(mapped,*t,parent) for k,t in transforms.items()}
                previous = old["predictions"]
                pred = {k:previous[k] for k in ("current-auto","paper-auto","raw-tree","early-forest-half")}
                pred["strong-early-tree"] = previous["early-tree"]
                for name,(kind,_) in RECIPES.items():
                    pred[name] = dict(zip(keys,models[name].predict(transformed[kind]).tolist(),strict=True))
                pred["early-parent"] = dict(zip(keys,parent.predict(transformed["early"]).tolist(),strict=True))
                payload = {"parent_receipt":req,"models":states,"ids":list(keys),"predictions":pred}
                rid = digest([site,payload])
                ack = receiver.submit(rid,payload)
                if not ack["accepted"] or ack["request_id"]!=rid:
                    raise RuntimeError("官方续训权重交接失败")
                sealed = read(receiver.output/(rid+".json"))["payload"]["predictions"]
                scores = summary("engine",answers(),{"parent":sealed["current-auto"],**sealed},groups)
                results[site] = {"request":rid,"scores":scores}
                print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in scores.items()}},ensure_ascii=False),flush=True)
            atomic_json(DEST/"scores.json",results)


if __name__=="__main__":
    main()
