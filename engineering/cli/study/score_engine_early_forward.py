"""在截断设备上把末次未知寿命保留为变量，不补造失效标签。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np

from score_engine_early_reference import DEST as RESEARCH, ROOT, OUT, frozen, tree_values, read, digest, LocalStore, LocalAcceptance, atomic_json
from score_censored_engines import available_tracks,paths,ModelArtifacts,_trainer
from harness.models.early_engine_reference import EarlyEngineReference
from harness.models.source_combinations import SourceCombinations
from harness.models.training_tools import SklearnBackend

DEST = RESEARCH/"forward"


def compare(records):
    result = {}
    for name in ("current","paper","tree","uniform"):
        counts = {"better":0,"worse":0,"equal":0,"unknown":0}
        for row in records:
            gap = row["last"]-np.asarray(row["cycles"])
            a = np.asarray(row["predictions"]["early"])-gap
            b = np.asarray(row["predictions"][name])-gap
            intercept = np.mean(a*a-b*b)
            slope = -2*np.mean(a-b)
            if intercept==0 and slope==0:
                counts["equal"]+=1
            elif intercept<0 and slope<=0:
                counts["better"]+=1
            elif intercept>0 and slope>=0:
                counts["worse"]+=1
            else:
                counts["unknown"]+=1
        result[name] = counts
    fractions = {}
    for arm in records[0]["predictions"]:
        deficits = np.concatenate([r["last"]-np.asarray(r["cycles"])-r["predictions"][arm] for r in records])
        fractions[arm] = float(np.mean(deficits>0))
    return {"paired":result,"below_observed_fraction":fractions}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("冻结后的复核已有结果")
        delivered = RESEARCH/"integrated/engine"
        report = read(delivered/"run.json")
        header,blob = ModelArtifacts(delivered/"references").read(report["tree_artifact"])
        repair = SourceCombinations.restore(read(delivered/"source-combinations.json"))
        if digest([header["request"],header["sha256"]])!=report["reference_identity"] or header["request"]["repair"]!=repair.identity:
            raise ValueError("交付参照与修复身份不配套")
        early = SklearnBackend(execution_enabled=True).load(blob)
        file = RESEARCH/"uniform.joblib"
        if sha256(file.read_bytes()).hexdigest()!=read(RESEARCH/"training.json")["models"]["uniform"]["sha256"]:
            raise ValueError("全寿命参照权重已变")
        uniform = joblib.load(file)
        tree = frozen("engine")[0]["catboost"]
        current,other = paths("engine","engine")
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        summary = {}
        with TemporaryDirectory(dir=DEST) as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
            models = {}
            for arm,folder in (("current",current),("paper",other["paper"])):
                ref = read(folder/"run.json")["rounds"][-1]["model_artifact"]
                _,weights = ModelArtifacts(folder/"models").read(ref)
                p = Path(tmp)/(arm+".npz")
                p.write_bytes(weights)
                models[arm] = trainer.Model.load(p)
            for name,tracks,excluded in available_tracks():
                if name=="train.txt":
                    continue
                site = Path(name).stem
                old = read(OUT/"setting-reference/automatic/censored"/site/"issued.json")
                indexed = {(r["device"],r["cycle"]):r["pred"] for r in old["predictions"]}
                records = []
                for uid,v in tracks.items():
                    check = sorted({*range(0,len(v),20),len(v)-1})
                    new = [i for i in range(9,len(v),20) if (uid,i+1) not in indexed]
                    if not new:
                        raise ValueError("设备没有协议要求的新测点")
                    idx = check+new
                    x = np.column_stack((v[idx,3:],v[idx,:3]))
                    mapped = repair.transform(x)
                    pred = {k:m.predict(mapped) for k,m in models.items()}
                    pred["tree"] = tree_values(tree,"engine",mapped)
                    for k in ("tree","paper","current"):
                        np.testing.assert_allclose(pred[k][:len(check)],[indexed[uid,i+1][k+"-mapped"] for i in check],rtol=0,atol=1e-7)
                    pred["early"] = EarlyEngineReference.predict(early,repair,x)
                    pred["uniform"] = EarlyEngineReference.predict(uniform,repair,x)
                    # 末次时钟只供下方评估使用；各模型仅接收上述测量矩阵。
                    records.append({"device":uid,"last":len(v),"cycles":[i+1 for i in new],
                                    "predictions":{k:a[len(check):].tolist() for k,a in pred.items()}})
                payload = {"site":site,"reference":report["reference_identity"],"rows":records}
                request = digest(payload)
                ack = receiver.submit(request,payload)
                if not ack["accepted"] or ack["request_id"]!=request:
                    raise RuntimeError("新测量交付失败")
                sealed = read(receiver.output/(request+".json"))["payload"]["rows"]
                metrics = compare(sealed)
                summary[site] = {"request":request,"devices":len(sealed),"excluded":excluded,
                                 "new_measurements":sum(len(r["cycles"]) for r in sealed),**metrics}
                print(json.dumps({site:metrics},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",summary)


if __name__=="__main__":
    main()
