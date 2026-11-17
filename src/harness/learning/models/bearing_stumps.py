"""在随包轴承权重上续接一层浅树；训练、恢复与单条预测都读取同一检查点里的修正。"""
from hashlib import sha256
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from harness.core.contracts import digest
from harness.adapters.datasets.official_rul import ARCHIVE_SHA
from harness.learning.models.official_rul_cpu import RulPilotTools, _subset, _trainer, read_parent
from harness.adapters.storage.local import atomic_json


def stump_model(base):
    class StumpModel(base):
        def predict(self,x):
            value=super().predict(x)
            for stump in self.meta.get("bearing_residual_stumps",[]):
                if stump["feature"]==-1:
                    delta=stump["left"]
                else:
                    column=np.asarray(x[:,stump["feature"]],dtype=np.float32).astype(float)
                    delta=np.where(column<=stump["threshold"],stump["left"],stump["right"])
                value=np.maximum(0,value+.5*delta)
            return value

        def fit(self,x,y,epochs):
            from sklearn.tree import DecisionTreeRegressor
            if epochs!=1 or self.meta["profile"]!="B":
                raise ValueError("每遍仅对轴承父版本增加一棵树")
            est=DecisionTreeRegressor(max_depth=1,min_samples_leaf=max(5,ceil(.02*len(x))),random_state=17)
            tree=est.fit(x,y-self.predict(x)).tree_
            if tree.node_count==1:
                c=float(tree.value[0,0,0])
                stump={"feature":-1,"threshold":0.,"left":c,"right":c}
            else:
                stump={"feature":int(tree.feature[0]),"threshold":float(tree.threshold[0]),
                       "left":float(tree.value[tree.children_left[0],0,0]),"right":float(tree.value[tree.children_right[0],0,0])}
            self.meta.setdefault("bearing_residual_stumps",[]).append(stump)
            self.meta["steps"]+=1
            self.meta["epochs"]+=1
            self.validate()
            return [float(np.mean(((y-self.predict(x))/self.meta["target_scale"])**2))]
    return StumpModel


class StumpTools(RulPilotTools):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from importlib.metadata import version
        self.recipe={"implementation":sha256(Path(__file__).read_bytes()).hexdigest(),
                     "numpy":np.__version__,"sklearn":version("scikit-learn")}
        self.identity=digest([self.identity,self.recipe])

    def _fit_request(self,task,action,rows,parent):
        if action.fit_manifest not in (None,"full"):
            raise ValueError("单层树使用完整冻结清单")
        return {**super()._fit_request(task,action,rows,parent),"bearing_stump":self.recipe}

    def _prediction_request(self,artifact,inputs):
        return {**super()._prediction_request(artifact,inputs),"bearing_stump":self.recipe}


def execute_stump_job(folder,payload,artifacts):
    """预测从父链读嵌入的树；临时原包训练器仅绑定本次隔离工作进程。"""
    from importlib.metadata import version
    request,backend=payload["request"],payload["backend"]
    origin=Path(backend["archive"])
    recipe={"implementation":sha256(Path(__file__).read_bytes()).hexdigest(),"numpy":np.__version__,"sklearn":version("scikit-learn")}
    if (request["bearing_stump"]!=recipe or backend["task"]!="bearing" or request["package_sha256"]!=ARCHIVE_SHA["bearing"]
            or sha256(origin.read_bytes()).hexdigest()!=request["package_sha256"]):
        raise ValueError("轴承候选程序或原包变化")
    with TemporaryDirectory(dir=folder) as tmp, ZipFile(origin) as z,_trainer(z,Path(tmp),"B") as trainer:
        work=Path(tmp)
        root={"package_sha256":request["package_sha256"],"checkpoint_sha256":sha256(z.read("checkpoint.npz")).hexdigest()}
        predicting=request["kind"]=="predict"
        head,blob,_=read_parent(artifacts,request["artifact"] if predicting else request["parent"],root)
        if request["checkpoint_sha256"]!=root["checkpoint_sha256"]:
            raise ValueError("初始模型身份不符")
        p=work/"parent.npz"
        p.write_bytes(blob)
        trainer.Model=stump_model(trainer.Model)
        parent=trainer.Model.load(p)
        if predicting:
            rows=payload["rows"]
            if digest(rows)!=request["inputs_digest"] or any(set(r)!={"sample_id","values"} for r in rows):
                raise ValueError("单条服务请求与摘要不同")
            values=parent.predict(np.array([r["values"] for r in rows]))
            if len(values)!=len(rows) or not np.isfinite(values).all():
                raise ValueError("寿命输出无效")
            return {"value":dict(zip((r["sample_id"] for r in rows),map(float,values),strict=True))}
        if (request["kind"]!="continue" or request["epochs"]!=1 or payload["rows"] or request["parent_sha256"]!=head["sha256"]
                or request["worker_sha256"]!=sha256(Path(__file__).with_name("official_rul_cpu.py").read_bytes()).hexdigest()):
            raise ValueError("续训父版、原包工作器或请求不符")
        diag=backend.get("training_diagnostics",request)
        train,valid=work/"train.jsonl",work/"explore.jsonl"
        _subset(z,"train",request["fit_ids"],train)
        _subset(z,"validation",diag["exploration_ids"],valid)
        report=trainer.train_run(str(work/"baselines"/"configs"/"B.json"),train,valid,work/"candidate",
                                 resume=p,epochs=1,stage_id=diag["stage"])
        output=work/"candidate"/"checkpoint.npz"
        child=trainer.Model.load(output)
        content=output.read_bytes()
        if (len(child.meta["bearing_residual_stumps"])!=1+len(parent.meta.get("bearing_residual_stumps",[]))
                or child.meta["steps"]!=parent.meta["steps"]+1 or child.meta["epochs"]!=parent.meta["epochs"]+1
                or report["parent_checkpoint_sha256"]!="sha256:"+head["sha256"]
                or report["checkpoint_sha256"]!="sha256:"+sha256(content).hexdigest()):
            raise ValueError("增量训练或发布权重不符")
        atomic_json(folder/"training.json",{**report,"bearing_stump":recipe,"added":child.meta["bearing_residual_stumps"][-1]})
        (folder/"history.jsonl").write_bytes((work/"candidate"/"history.jsonl").read_bytes())
        return {"artifact":artifacts.create(request,lambda:content)}
