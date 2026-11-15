"""固定寿命树的容量，只更换其目标为源设备上实际的官方预测误差。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.base import clone

from score_engine_native_condition import DEST as NATIVE, EARLIER, ROOT, official, read, tree_values, cohorts, summary, _trainer
from harness.contracts import digest
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = NATIVE.parent/"engine-residual-condition"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此协议已有完整结果")
        pkg,checksum,batches = official("engine")
        ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        rows = [r for r in pkg.train if r.sample_id in ids]
        source = read(EARLIER/"training.json")
        if source["fit_ids"] != [r.sample_id for r in rows]:
            raise ValueError("源训练清单改变")
        tree_file = EARLIER/"early.joblib"
        trained_file = NATIVE/"raw-truth.npz"
        if (sha256(tree_file.read_bytes()).hexdigest()!=source["models"]["early"]["sha256"]
                or sha256(trained_file.read_bytes()).hexdigest()!=read(NATIVE/"training.json")["models"]["raw-truth"]["sha256"]):
            raise ValueError("树或普通续训模型身份改变")
        pipeline = joblib.load(tree_file)
        condition = pipeline.named_steps["condition"]
        tree = pipeline.named_steps["tree"]
        x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows])
        repair = SourceCombinations(x)
        z = condition.transform(x)
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        if sha256(archive.read_bytes()).hexdigest()!=checksum:
            raise ValueError("官方源包身份改变")
        original_receipts = read(NATIVE/"scores.json")
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        results = {}
        with TemporaryDirectory(dir=DEST) as temp, ZipFile(archive) as pack, _trainer(pack,Path(temp),"C") as trainer:
            path = Path(temp)/"original.npz"
            path.write_bytes(pkg.checkpoint)
            models = {"parent":trainer.Model.load(path),"trained":trainer.Model.load(trained_file)}
            correction,means,identities = {},{},{}
            for key,model in models.items():
                residual = y-model.predict(x)
                learner = clone(tree).fit(z,residual)
                file = DEST/(key+".joblib")
                joblib.dump(learner,file)
                correction[key] = joblib.load(file)
                identities[key] = sha256(file.read_bytes()).hexdigest()
                means[key] = float(residual.mean())
            atomic_json(DEST/"training.json",{"source":digest(source["fit_ids"]),"corrections":identities,"means":means})
            for site,keys,features,groups,answers,_ in cohorts("engine",pkg):
                prior = original_receipts[site]["request"]
                old = read(NATIVE/"accepted"/(prior+".json"))["payload"]
                if digest([site,old])!=prior or old["ids"]!=list(keys):
                    raise ValueError("服务来源或顺序不符")
                pred = {k:old["predictions"][k] for k in ("current-auto","paper-auto","strong-early-tree")}
                mapped = repair.transform(np.asarray(features))
                field = condition.transform(mapped)
                direct = np.maximum(0,tree_values(tree,"engine",field))
                np.testing.assert_allclose(direct,[pred["strong-early-tree"][i] for i in keys],rtol=0,atol=1e-7)
                for key,model in models.items():
                    base = model.predict(mapped)
                    if key=="trained":
                        np.testing.assert_allclose(base,[old["predictions"]["raw-truth"][i] for i in keys],rtol=0,atol=1e-7)
                    delta = tree_values(correction[key],"engine",field)
                    values = {"base":base,"blend":.5*(base+direct),"constant":base+means[key],
                              "residual":base+delta,"partial":base+.5*delta}
                    for label,v in values.items():
                        pred[key+"-"+label] = dict(zip(keys,np.maximum(0,v).tolist(),strict=True))
                payload = {"ids":list(keys),"prior":prior,"weights":identities,"predictions":pred}
                rid = digest([site,payload])
                ack = receiver.submit(rid,payload)
                if not ack["accepted"] or ack["request_id"]!=rid:
                    raise RuntimeError("修正后的寿命预测未交接")
                sealed = read(receiver.output/(rid+".json"))["payload"]["predictions"]
                scores = summary("engine",answers(),{"parent":sealed["current-auto"],**sealed},groups)
                results[site] = {"request":rid,"scores":scores}
                print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in scores.items()}},ensure_ascii=False),flush=True)
            atomic_json(DEST/"scores.json",results)


if __name__=="__main__":
    main()
