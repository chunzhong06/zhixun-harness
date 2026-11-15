"""在旧测试机队此前未逐行评分的中途循环上检验已交付的赛方原模型。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import joblib
import numpy as np

from compare_fixed_tree_references import BASE, paths, tree_values
from engine_endpoint_check import RAW, SUBSETS, read_cmapss, test_answers
from score_engine_full_condition import DEST as STUDY, ROOT, _trainer, read, summary
from harness.contracts import digest
from harness.models.early_engine_reference import EarlyEngineReference
from harness.models.full_condition_engine import FullConditionEngine
from harness.models.model_artifacts import ModelArtifacts
from harness.models.source_combinations import SourceCombinations
from harness.models.training_tools import SklearnBackend
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent/"engine-full-condition-forward"


def released(folder,field):
    report = read(folder/"run.json")
    manifest,blob = ModelArtifacts(folder/"references").read(report[field])
    source = SourceCombinations.restore(read(folder/"source-combinations.json"))
    if (report["completed_rounds"]!=10 or manifest["request"]["repair"]!=source.identity
            or digest([manifest["request"],manifest["sha256"]])!=report["reference_identity"]):
        raise ValueError("已交付参照、输入修复或完整轮数不符")
    return report,source,blob


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("本批测量已有结果")
        public,repair,blob = released(STUDY.parent/"engine-full-condition-integrated/engine","native_artifact")
        _,old_repair,old_blob = released(STUDY.parent/"engine-nonnegative-integrated/engine","tree_artifact")
        if repair.identity!=old_repair.identity:
            raise ValueError("比较工具的输入修复不同")
        small = SklearnBackend(execution_enabled=True).load(old_blob)
        state = read(STUDY/"training.json")
        tree_path = STUDY/"early-tree.joblib"
        if sha256(tree_path.read_bytes()).hexdigest()!=state["states"]["early-tree"]["sha256"]:
            raise ValueError("全量强固定树改变")
        full_tree = joblib.load(tree_path)
        known = set()
        for suffix in ("","-offset6","-offset11","-offset16"):
            known.update(read(BASE/"live-first"/("cross-stream"+suffix)/"issued.json")["predictions"]["fixed"])
        series = {s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
        owner,subset,stage,gaps,obs = {},{},{},{},[]
        for s,tracks in series.items():
            for t in tracks:
                for c in range(13,len(t.values),20):
                    k = f"{s}/{t.unit}/{c}"
                    if k in known or (c-9)%20==0:
                        raise ValueError("新循环与已登记的研究测点重合")
                    owner[k],subset[k],stage[k],gaps[k] = t.unit_id,s,c,len(t.values)-c
                    row = t.values[c-1]
                    obs.append([*row[3:],*row[:3]])
        keys = list(owner)
        x = np.asarray(obs)
        mapped = repair.transform(x)
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        if sha256(archive.read_bytes()).hexdigest()!=state["source"]:
            raise ValueError("赛方原包身份改变")
        current,others = paths("engine","engine")
        with TemporaryDirectory(dir=DEST) as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
            cond,center,model,weight = FullConditionEngine.load_bundle(blob,trainer)
            if weight!=public["native_sha256"] or weight!=state["states"]["early-native"]["sha256"]:
                raise ValueError("赛方续训权重不能复现")
            out = {"candidate":model.predict(FullConditionEngine.inputs(cond,center,model,mapped)),
                   "full-early-tree":tree_values(full_tree,"engine",cond.transform(mapped)),
                   "small-early-tree":EarlyEngineReference.predict(small,repair,x)}
            for name,folder in (("current",current),("paper",others["paper"])):
                ref = read(folder/"run.json")["rounds"][-1]["model_artifact"]
                _,weights = ModelArtifacts(folder/"models").read(ref)
                path = Path(tmp)/(name+".npz")
                path.write_bytes(weights)
                out[name] = trainer.Model.load(path).predict(mapped)
        payload = {"ids":keys,"inputs":digest(x.tolist()),"reference":public["reference_identity"],
                   "predictions":{k:dict(zip(keys,np.maximum(0,v).tolist(),strict=True)) for k,v in out.items()}}
        request = digest(["cycles13",payload])
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        ack = receiver.submit(request,payload)
        if not ack["accepted"] or ack["request_id"]!=request:
            raise RuntimeError("新测点预测未交接")
        issued = read(receiver.output/(request+".json"))["payload"]["predictions"]
        # 同一设备末端答案此前已用于其他研究，仅本批中途预测是冻结后第一次发放。
        ends = {u:e for s,tracks in series.items() for u,e in test_answers(RAW,s,tracks).items()}
        truth = {k:ends[owner[k]]+gaps[k] for k in keys}
        masks = {"全部":keys,**{s:[k for k in keys if subset[k]==s] for s in SUBSETS},
                 "前100循环":[k for k in keys if stage[k]<=100],"100循环后":[k for k in keys if stage[k]>100]}
        result = {}
        for g,ids in masks.items():
            score = summary("engine",truth,{"parent":{k:issued["current"][k] for k in ids},
                            **{n:{k:values[k] for k in ids} for n,values in issued.items()}},owner)
            result[g] = {n:v for n,v in score.items() if n!="parent"}
            print(json.dumps({g:{n:{"误差":v["score"],"最差设备":v["worst_group"]} for n,v in result[g].items()}},ensure_ascii=False),flush=True)
        atomic_json(DEST/"scores.json",{"request":request,"rows":len(keys),"devices":len(set(owner.values())),"scores":result})


if __name__=="__main__":
    main()
