"""只在源训练资料上学习投票者的软目标，保存可被赛方原模型类单独加载的气体权重。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from compare_calibration_expiry import source_window
from gas_complementary_feedback import ROOT,BASE,OUT,REGISTERED_SHA256,atomic_json,official,read,read_gas_batches
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from harness.models.vote_feedback import majority


def centered(v):
    return v-v.mean(axis=1,keepdims=True)


def fit(trainer,blob,folder,current,bank,x,y):
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/"source.npz"
    source.write_bytes(blob)
    original=trainer.Model.load(source)
    outputs=bank.predict(current,x)
    members=np.stack(list(outputs.values()))
    vote=np.stack([(members==k).sum(0) for k in range(1,7)],axis=1)
    true=np.eye(6)[y-1]*len(outputs)
    targets={k:centered(np.log(v+1)) for k,v in (("vote",vote),("true",true))}
    targets["mix"]=.5*(targets["vote"]+targets["true"])
    targets["shuffle"]=targets["vote"][np.random.default_rng(17).permutation(len(x))]
    h,logits=original.forward(original.normalized(x))
    design=np.column_stack((h,np.ones(len(x))))
    gram=design.T@design+.1*len(x)*np.eye(design.shape[1])
    models={"raw":original}
    for name,target in targets.items():
        candidate=trainer.Model.load(source)
        delta=np.linalg.solve(gram,design.T@(target-centered(logits)))
        candidate.params["w2"]+=delta[:-1]
        candidate.params["b2"]+=delta[-1]
        candidate.meta["steps"]+=1
        candidate.meta["epochs"]+=1
        path=folder/(name+".npz")
        candidate.save(path)
        models[name]=trainer.Model.load(path)
    return models


def main():
    pkg,_,batches=official("gas")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows=[r for r in pkg.train if r.sample_id in ids]
    x=np.asarray([r.values for r in rows])
    y=np.asarray([r.label for r in rows])
    bank=GasFeedbackBank(rows)
    src=BASE/"selected/gas"
    store=ModelArtifacts(src/"models")
    records=read(src/"run.json")["rounds"]
    state=read(src/"state.json")["samples"]
    archive,=(ROOT/"data/official/gas").glob("*.zip")
    later=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    folder=OUT/"vote-readout"
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"A") as trainer:
        cache={}
        def predict(record,values):
            header,blob=store.read(record["model_artifact"])
            key=header["sha256"]
            if key not in cache:
                path=Path(tmp)/"current.npz"
                path.write_bytes(blob)
                model=pair_model(trainer.Model).load(path)
                options=fit(trainer,blob,folder/"models"/key,model,bank,x,y)
                cache[key]=(model,options)
            current,options=cache[key]
            fixed=bank.predict(current,values)
            return {"current":fixed["current"],"fixed-vote":majority(fixed,6),
                    **{name:model.predict(values).argmax(1)+1 for name,model in options.items()}}
        for name in ("public","late"):
            streams={}
            groups=[]
            for j in range(len(records) if name=="public" else len(later)):
                if name=="public":
                    _,_,serve=source_window(state,j)
                    keys=[s["sample_id"] for s in serve]
                    inputs=[s["values"] for s in serve]
                    record=records[j]
                else:
                    keys=[s.sample_id for s in later[j].records]
                    inputs=[s.values for s in later[j].records]
                    record=records[3]
                outputs=predict(record,np.asarray(inputs))
                if name=="public" and dict(zip(keys,outputs["current"].tolist(),strict=True))!=read(src/record["submitted"])["payload"]["predictions"]:
                    raise ValueError("父模型的预测不符")
                for k,v in outputs.items():
                    streams.setdefault(k,[]).extend(v.tolist())
                groups.append(keys)
            sealed=folder/name/"issued.json"
            atomic_json(sealed,{"keys":groups,"predictions":streams})
            truth=({r.sample_id:r.label for r in pkg.validation} if name=="public" else
                   {r.sample_id:r.label for b in later for r in b.records})
            target=np.array([truth[k] for block in groups for k in block])
            blocks=np.split(np.arange(len(target)),np.cumsum([len(g) for g in groups])[:-1])
            t={k:scores(target,np.array(v),blocks,6) for k,v in read(sealed)["predictions"].items()}
            atomic_json(sealed.with_name("scores.json"),{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":t})
            print(name,{k:round(r["pooled"],5) for k,r in t.items()})


if __name__=="__main__":
    main()
