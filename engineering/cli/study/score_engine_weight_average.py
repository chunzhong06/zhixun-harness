"""复现旧的十遍续训，再公平比较继续训练的末版权重、平均权重及多模型平均输出。"""
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np

from score_engine_full_condition import DEST as PARENT, ROOT, cohorts, read, summary, official, _trainer
from score_engine_full_condition_forward import released
from harness.contracts import digest
from harness.models.full_condition_engine import FullConditionEngine
from harness.runtime.storage import LocalStore,atomic_json
from harness.runtime.handoff import LocalAcceptance

DEST = PARENT.parent/"engine-weight-average"
SNAPSHOTS = (20,30,40,50,60)


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已有训练轨迹平均结果")
        pkg,checksum,_ = official("engine")
        initial = pkg.checkpoint
        old = read(PARENT/"scores.json")
        report,repair,bundle = released(PARENT.parent/"engine-full-condition-integrated/engine","native_artifact")
        archive, = (ROOT/"data/official/engine").glob("*.zip")
        if sha256(archive.read_bytes()).hexdigest()!=checksum:
            raise ValueError("官方包校验失败")
        endpoint = LocalAcceptance()
        endpoint.output = DEST/"accepted"
        with TemporaryDirectory(dir=DEST) as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
            cond,center,_,identity = FullConditionEngine.load_bundle(bundle,trainer)
            p = Path(tmp)/"parent.npz"
            p.write_bytes(initial)
            model = trainer.Model.load(p)
            if any(r.role!="fit" for r in pkg.train):
                raise ValueError("源训练池用途不符")
            x = FullConditionEngine.inputs(cond,center,model,[r.values for r in pkg.train])
            y = np.asarray([r.label for r in pkg.train])
            tic = process_time()
            nets,hashes = {},{}
            for epoch in (10,*SNAPSHOTS):
                model.fit(x,y,10)
                file = DEST/f"epoch-{epoch}.npz"
                model.save(file)
                key = f"epoch-{epoch}"
                hashes[key] = sha256(file.read_bytes()).hexdigest()
                nets[key] = trainer.Model.load(file)
                if epoch==10 and (hashes[key]!=identity or identity!=report["native_sha256"]):
                    raise ValueError("分段训练没有复现已交付十遍权重")
            elapsed = process_time()-tic
            # 保留最后一遍可续训状态；参数平均版是独立推理产物，不作为原优化轨迹的接续点。
            average = trainer.Model.load(DEST/"epoch-60.npz")
            for k in average.params:
                average.params[k] = np.mean([nets[f"epoch-{e}"].params[k] for e in SNAPSHOTS],axis=0)
            file = DEST/"average.npz"
            average.save(file)
            nets["average"] = trainer.Model.load(file)
            hashes["average"] = sha256(file.read_bytes()).hexdigest()
            atomic_json(DEST/"training.json",{"source":checksum,"parent":sha256(initial).hexdigest(),"weights":hashes,
                                              "averaged_epochs":SNAPSHOTS,"cpu_seconds":elapsed})
            result = {}
            for site,keys,values,groups,answers,_ in cohorts("engine",pkg):
                req = old[site]["request"]
                issued = read(PARENT/"accepted"/(req+".json"))["payload"]
                if issued["ids"]!=list(keys) or digest([site,issued])!=req:
                    raise ValueError("与冻结服务清单不一致")
                mapped = FullConditionEngine.inputs(cond,center,model,repair.transform(values))
                pred = {k:np.maximum(0,v.predict(mapped)) for k,v in nets.items()}
                pred["output-average"] = np.mean([pred[f"epoch-{e}"] for e in SNAPSHOTS],axis=0)
                np.testing.assert_allclose(pred["epoch-10"],[issued["predictions"]["full-early-native"][k] for k in keys],
                                           rtol=0,atol=1e-9)
                scores = {k:dict(zip(keys,v.tolist(),strict=True)) for k,v in pred.items()}
                scores.update({k:issued["predictions"][k] for k in ("full-early-tree","early-tree","current-auto","paper-auto")})
                payload = {"ids":list(keys),"predictions":scores,"weight_hashes":hashes,"prior":req}
                key = digest([site,payload]); ack=endpoint.submit(key,payload)
                if not ack["accepted"] or ack["request_id"]!=key:
                    raise ValueError("预测交接失败")
                stored=read(endpoint.output/(key+".json"))["payload"]["predictions"]
                table=summary("engine",answers(),{"parent":stored["epoch-10"],**stored},groups)
                del table["parent"]
                result[site]={"request":key,"scores":table}
                print(json.dumps({site:{k:{"误差":v["score"],"最差设备":v["worst_group"]} for k,v in table.items()}},ensure_ascii=False),flush=True)
            atomic_json(DEST/"scores.json",result)


if __name__=="__main__":
    main()
