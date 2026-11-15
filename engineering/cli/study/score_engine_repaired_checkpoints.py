"""在相同原始与修复输入上重载已冻结的单权重，检验上游修复是否改变候选取舍。"""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,frozen,tree_values
from compare_group_objectives import ROOT,_trainer,official,read,summary
from score_life_training_weights import cohorts
from harness.models.source_combinations import SourceCombinations
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json
from harness.contracts import digest

DEST=OUT.parent/"whole-maintenance/engine-repaired-checkpoints"
PROTOCOL=ROOT/"engineering/configs/experiments/engine-repaired-checkpoints.json"


def load_models(trainer,pkg):
    head=OUT/"teacher-readout/engine";hidden=OUT/"tree-hidden/engine/models"
    a=read(head/"training.json");b=read(hidden/"training.json")
    if a["parent"]!=pkg.checkpoint_sha or b["parent"]!=pkg.checkpoint_sha:
        raise ValueError("候选不来自同一官方起点")
    files={**{k:(head/"models"/(k+".npz"),a["runs"][k]["sha256"]) for k in ("true-ridge","teacher-ridge")},
           **{k:(hidden/(k+".npz"),b["arms"][k]["sha256"]) for k in ("frozen-truth","half-truth")}}
    models={}
    for k,(p,h) in files.items():
        if sha256(p.read_bytes()).hexdigest()!=h:raise ValueError("历史候选权重身份变化")
        models[k]=trainer.Model.load(p)
    return models,{k:h for k,(_,h) in files.items()}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("已完成相同对照，请复用")
        pkg,_,batches=official("engine");fit={r.sample_id for r in batches[0].samples if r.role=="fit"}
        source=np.asarray([r.values for r in pkg.train if r.sample_id in fit])
        if len(source)!=3968:raise ValueError("输入修复源库不同")
        repair=SourceCombinations(source);tree=frozen("engine")[0]["catboost"]
        archive,=(ROOT/"data/official/engine").glob("*.zip")
        all_scores={}
        with TemporaryDirectory(dir=DEST) as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
            models,identities=load_models(trainer,pkg)
            root=Path(tmp)/"original.npz";root.write_bytes(pkg.checkpoint);original=trainer.Model.load(root)
            models["original"]=original;identities["original"]=pkg.checkpoint_sha
            for name,keys,x,groups,answers,_ in cohorts("engine",pkg):
                prior=OUT/"setting-reference/automatic"/name
                old_path=prior/"issued.json"
                if sha256(old_path.read_bytes()).hexdigest()!=read(prior/"scores.json")["issued_sha256"]:
                    raise ValueError("现行修复分数与封存预测不符")
                saved=read(old_path);old=saved["predictions"]
                if any(set(v)!=set(keys) for v in old.values()):
                    raise ValueError("历史交付与当前服务行集合不同")
                x=np.asarray(x);mapped=repair.transform(x)
                if any(not np.array_equal(m.mean,original.mean) or not np.array_equal(m.scale,original.scale) for m in models.values()):
                    raise ValueError("候选改变了官方输入尺度")
                pred={k:old[k] for k in ("current-raw","current-auto","paper-auto","tree-auto")}
                tree_p=tree_values(tree,"engine",mapped)
                np.testing.assert_allclose(tree_p,[old["tree-auto"][k] for k in keys],rtol=0,atol=1e-7)
                np.testing.assert_allclose(tree_values(tree,"engine",x),[old["tree-raw"][k] for k in keys],rtol=0,atol=1e-7)
                for arm,model in models.items():
                    for treatment,values in (("raw",x),("auto",mapped)):
                        p=model.predict(values)
                        pred[arm+"/"+treatment]=dict(zip(keys,p.tolist(),strict=True))
                # 原初始化逐条复现，确认新候选与旧证据使用同一特征次序和修复。
                np.testing.assert_allclose([pred["original/auto"][k] for k in keys],[old["original-auto"][k] for k in keys],rtol=0,atol=1e-7)
                np.testing.assert_allclose([pred["original/raw"][k] for k in keys],[old["original-raw"][k] for k in keys],rtol=0,atol=1e-7)
                # 既有动态维护与论文臂复用已封存交付；本次新推理为固定树与单权重。
                payload={"ids":list(keys),"inputs_sha256":digest([list(keys),x.tolist()]),
                         "models":identities,"source_repair":repair.identity,"predictions":pred}
                request=digest([digest(read(PROTOCOL)),name,payload]);endpoint=LocalAcceptance();endpoint.output=DEST/name/"accepted"
                ack=endpoint.submit(request,payload)
                if not ack.get("accepted") or ack["request_id"]!=request:raise RuntimeError("候选预测尚未交接")
                issued=read(endpoint.output/(request+".json"))["payload"]
                table=summary("engine",answers(),{"parent":issued["predictions"]["current-auto"],**issued["predictions"]},groups)
                record={"request":request,"scores":{a:{k:v[k] for k in ("score","worst_group","group_scores")} for a,v in table.items()}}
                atomic_json(DEST/name/"scores.json",record)
                all_scores[name]=record
                print(json.dumps({"case":name,"scores":{a:[round(v["score"],3),round(v["worst_group"],3)] for a,v in table.items()}}),flush=True)
        atomic_json(DEST/"scores.json",{"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"models":identities,
                                       "cases":all_scores,"new_training":False,"development_only":True})


if __name__=="__main__":main()
