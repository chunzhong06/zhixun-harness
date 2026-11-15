"""在同一批源设备上替换工况参照的采样时期，再用同一树配置评估。"""
import json
from hashlib import sha256

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, official, read, summary
from score_life_training_weights import cohorts
from harness.contracts import digest
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = OUT.parent/"whole-maintenance/engine-early-reference"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("此协议已有结果")
        pkg,_,batches = official("engine")
        keys = {r.sample_id for r in batches[0].samples if r.role=="fit"}
        rows = [r for r in pkg.train if r.sample_id in keys]
        by_device = {}
        for r in pkg.train:
            by_device.setdefault(r.unit_id,[]).append(r)
        if len(rows)!=3968 or len(by_device)!=496 or any(len(v)<8 for v in by_device.values()):
            raise ValueError("源训练规模改变")
        ordered = [sorted(v,key=lambda r:r.time_end) for v in by_device.values()]
        references = {"uniform":rows,"early":[r for unit in ordered for r in unit[:8]],
                      "late":[r for unit in ordered for r in unit[-8:]]}
        x = np.asarray([r.values for r in rows])
        y = np.asarray([r.label for r in rows])
        repair = SourceCombinations(x)
        original = frozen("engine")[0]["catboost"]
        models = {"fixed-tree":original}
        identities = {}
        for name,ref in references.items():
            raw = np.asarray([r.values for r in ref])
            scaler = ConditionStandardScaler((21,22,23),tuple(pkg.task.condition_decimals)).fit(raw)
            # 树的训练清单、标签及超参数都相同，参照只决定逐行的单位换算。
            tree = clone(original)
            tree.fit(scaler.transform(x),y)
            model = Pipeline([("condition",scaler),("tree",tree)])
            file = DEST/(name+".joblib")
            joblib.dump(model,file)
            models[name] = joblib.load(file)
            identities[name] = {"sha256":sha256(file.read_bytes()).hexdigest(),"reference_ids":[r.sample_id for r in ref],
                                "known_groups":len(scaler.groups_)}
        atomic_json(DEST/"training.json",{"fit_ids":[r.sample_id for r in rows],"models":identities})
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        scores = {}
        for site,ids,features,groups,answers,_ in cohorts("engine",pkg):
            old_folder = OUT/"setting-reference/automatic"/site
            path = old_folder/"issued.json"
            if sha256(path.read_bytes()).hexdigest()!=read(old_folder/"scores.json")["issued_sha256"]:
                raise ValueError("既有对照身份改变")
            old = read(path)["predictions"]
            mapped = repair.transform(np.asarray(features))
            fixed = tree_values(original,"engine",mapped)
            np.testing.assert_allclose(fixed,[old["tree-auto"][k] for k in ids],rtol=0,atol=1e-7)
            pred = {k:old[k] for k in ("current-auto","paper-auto","tree-auto")}
            for name in references:
                model = models[name]
                out = tree_values(model.named_steps["tree"],"engine",model.named_steps["condition"].transform(mapped))
                pred[name] = dict(zip(ids,out.tolist(),strict=True))
            payload = {"ids":list(ids),"models":{k:v["sha256"] for k,v in identities.items()},"predictions":pred}
            req = digest([site,payload])
            ack = receiver.submit(req,payload)
            if not ack["accepted"] or ack["request_id"]!=req:
                raise RuntimeError("发动机预测交接未完成")
            issued = read(receiver.output/(req+".json"))["payload"]["predictions"]
            table = summary("engine",answers(),{"parent":issued["current-auto"],**issued},groups)
            scores[site] = {"request":req,"scores":{k:{m:v[m] for m in ("score","worst_group","group_scores")} for k,v in table.items()}}
            print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in table.items()}},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",scores)


if __name__=="__main__":
    main()
