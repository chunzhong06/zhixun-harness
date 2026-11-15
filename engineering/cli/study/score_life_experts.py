"""在真实回标时钟上对照固定各半和按历史误差加权的两份寿命预测。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import source_window
from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import ROOT,atomic_json,read,official,_trainer,summary,remaining_minutes,remaining_cycles
from harness.models.model_artifacts import ModelArtifacts


class LifeExperts:
    def __init__(self):
        self.n=0
        self.loss=np.zeros(2)
        self.raw=np.zeros(2)
        self.last=np.zeros(2)

    def prior(self):
        if not self.n:
            return .5
        scores=-np.sqrt(8*np.log(2)/self.n)*self.loss
        prob=np.exp(scores-scores.max())
        return float(prob[1]/prob.sum())

    def advance(self,current,candidate,feedback,service):
        x=np.asarray([r["values"] for r in service])
        a,b=current.predict(x),candidate.predict(x)
        def winner(loss):
            return .5 if loss[0]==loss[1] else float(loss[1]<loss[0])
        weights={"current":0.,"candidate":1.,"half":.5,"earlier":self.prior(),
                 "last-winner":winner(self.last),"cumulative-winner":winner(self.raw)}
        if feedback:
            x_fb=np.asarray([r["values"] for r in feedback])
            y=np.asarray([r["label"] for r in feedback],dtype=float)
            error=(np.stack([current.predict(x_fb),candidate.predict(x_fb)])-y)**2
            self.last=error.sum(1)
            self.raw+=self.last
            denom=error.max(0)
            self.loss+=np.divide(error,denom,out=np.zeros_like(error),where=denom>0).sum(1)
            self.n+=len(y)
        weights["current-window"]=self.prior()
        return {k:a+(b-a)*v for k,v in weights.items()},weights


def standard(task,name):
    folder,_=paths(name,task)
    pkg,_,_=official(task)
    regular=read(folder/"run.json")["rounds"]
    samples=read(folder/"state.json")["samples"]
    artifacts=ModelArtifacts(folder/"models")
    root=OUT/"tree-hidden"/task
    cand=root/"models/whole-teacher.npz"
    gold=read(root/name/"issued.json")["predictions"]
    bank=LifeExperts()
    predictions={}
    trace=[]
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        candidate=trainer.Model.load(cand)
        for j,r in enumerate(regular):
            _,blob=artifacts.read(r["model_artifact"])
            p=Path(tmp)/"current.npz"
            p.write_bytes(blob)
            current=trainer.Model.load(p)
            e,c,s=source_window(samples,j)
            if any(v["label"] is not None for v in s):
                raise ValueError("服务样本不应携带真实寿命")
            old_seen=bank.n
            outputs,weights=bank.advance(current,candidate,e+c,s)
            ids=[v["sample_id"] for v in s]
            for model,v in outputs.items():
                predictions.setdefault(model,{}).update(zip(ids,v.tolist(),strict=True))
            for key,arm in (("current","current"),("candidate","whole-teacher")):
                if max(abs(gold[arm][k]-predictions[key][k]) for k in ids)>1e-8:
                    raise ValueError("重载的寿命专家与先前预测不同")
            trace.append({"batch":r["batch"],"seen_earlier":old_seen,"current_labels":len(e+c),"weights":weights})
    dest=root/"expert-weights"/name
    path=dest/"issued.json"
    atomic_json(path,{"predictions":predictions,"trace":trace})
    issued=read(path)["predictions"]
    y=({r.sample_id:r.label for r in pkg.validation} if name==task else
       remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json"))
    groups={k:v["unit_id"] for k,v in samples.items() if v["role"]=="serve"}
    table=summary(task,y,{"parent":issued["current"],**issued},groups)
    atomic_json(dest/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
    print(name,{k:round(v["score"],3) for k,v in table.items()})


if __name__=="__main__":
    for task,name in (("engine","engine"),("engine","phm08"),("bearing","bearing"),("bearing","pronostia")):
        standard(task,name)
