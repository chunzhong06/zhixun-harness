"""在同一全量发动机训练集上比较赛方原权重续训、源工况处理和树参照。"""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.base import clone

from score_engine_native_condition import (
    DEST as NATIVE, EARLIER, PRIOR, ROOT, _trainer, cohorts, model_input, official, read, summary, tree_values,
)
from harness.contracts import digest
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = NATIVE.parent/"engine-full-condition"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("完整训练池比较已有结果")
        pkg,checksum,batches = official("engine")
        rows = pkg.train
        if len(rows)!=113089 or any(r.role!="fit" for r in rows):
            raise ValueError("全部训练设备的公开范围发生变化")
        small = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        repair = SourceCombinations([r.values for r in rows if r.sample_id in small])
        x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows])
        prior = read(PRIOR/"scores.json")
        small_file = EARLIER/"early.joblib"
        transform_file = NATIVE/"transforms.joblib"
        if (sha256(small_file.read_bytes()).hexdigest()!=read(EARLIER/"training.json")["models"]["early"]["sha256"]
                or sha256(transform_file.read_bytes()).hexdigest()!=read(NATIVE/"training.json")["transform"]):
            raise ValueError("开工参照或原模型输入变换与小池发生变化")
        transforms = joblib.load(transform_file)
        condition = transforms["early"][0]
        template = joblib.load(small_file).named_steps["tree"]
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        if sha256(archive.read_bytes()).hexdigest()!=checksum:
            raise ValueError("赛方训练包身份改变")
        models,states = {},{}
        endpoint = LocalAcceptance()
        endpoint.output = DEST/"accepted"
        with TemporaryDirectory(dir=DEST) as temp, ZipFile(archive) as pack, _trainer(pack,Path(temp),"C") as trainer:
            checkpoint = Path(temp)/"parent.npz"
            checkpoint.write_bytes(pkg.checkpoint)
            parent = trainer.Model.load(checkpoint)
            for representation in ("raw","early"):
                transformed = x if representation=="raw" else condition.transform(x)
                name = representation+"-tree"
                tic = process_time()
                candidate = clone(template).fit(transformed,y)
                elapsed = process_time()-tic
                file = DEST/(name+".joblib")
                joblib.dump(candidate,file)
                models[name] = joblib.load(file)
                states[name] = {"sha256":sha256(file.read_bytes()).hexdigest(),"cpu_seconds":elapsed}
                name = representation+"-native"
                values = model_input(x,*transforms[representation],parent)
                candidate = trainer.Model.load(checkpoint)
                tic = process_time()
                candidate.fit(values,y,10)
                elapsed = process_time()-tic
                file = DEST/(name+".npz")
                candidate.save(file)
                models[name] = trainer.Model.load(file)
                states[name] = {"sha256":sha256(file.read_bytes()).hexdigest(),"cpu_seconds":elapsed,"steps":candidate.meta["steps"]}
                np.testing.assert_allclose(candidate.predict(values[:16]),models[name].predict(values[:16]),rtol=0,atol=1e-9)
            atomic_json(DEST/"training.json",{"source":checksum,"fit":digest([r.sample_id for r in rows]),
                                              "rows":len(rows),"repair":repair.identity,"states":states})
            results = {}
            for site,keys,inputs,owners,answers,_ in cohorts("engine",pkg):
                req = prior[site]["request"]
                old = read(PRIOR/"accepted"/(req+".json"))["payload"]
                if digest([site,old])!=req or list(keys)!=old["ids"]:
                    raise ValueError("既有比较测量发生变化")
                mapped = repair.transform(np.asarray(inputs))
                pred = {k:old["predictions"][k] for k in ("current-auto","paper-auto","early-tree","early-forest-half")}
                for rep in ("raw","early"):
                    z = mapped if rep=="raw" else condition.transform(mapped)
                    v = model_input(mapped,*transforms[rep],parent)
                    for kind,output in (("native",models[rep+"-native"].predict(v)),
                                        ("tree",tree_values(models[rep+"-tree"],"engine",z))):
                        pred["full-"+rep+"-"+kind] = dict(zip(keys,np.maximum(0,output).tolist(),strict=True))
                payload = {"ids":list(keys),"prior":req,"states":states,"predictions":pred}
                request = digest([site,payload])
                ack = endpoint.submit(request,payload)
                if not ack["accepted"] or ack["request_id"]!=request:
                    raise RuntimeError("全量训练候选交接失败")
                issued = read(endpoint.output/(request+".json"))["payload"]["predictions"]
                scores = summary("engine",answers(),{"parent":issued["current-auto"],**issued},owners)
                results[site] = {"request":request,"scores":scores}
                print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in scores.items()}},ensure_ascii=False),flush=True)
            atomic_json(DEST/"scores.json",results)


if __name__=="__main__":
    main()
