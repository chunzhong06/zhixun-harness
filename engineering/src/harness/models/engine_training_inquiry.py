"""从当轮合法探索、确认设备启动一次真实训练研究；后续仅加载已交付的模型。"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler

from harness.contracts import digest
from harness.methods import training_inquiry
from harness.methods.training_inquiry import LabeledBatch, investigate
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.continued_condition_engine import ContinuedConditionEngine
from harness.models.full_condition_engine import FullConditionEngine
from harness.models.model_artifacts import ModelArtifacts
from harness.models.source_combinations import SourceCombinations
from harness.models.training_tools import SklearnBackend


def input_view(reading, x, condition, center, native=None):
    if reading=="raw":
        return x
    if native is None:
        return condition.transform(x)
    return FullConditionEngine.inputs(condition,center,native,x)


def predict(bundle, x):
    learner,reading,model,condition,center = bundle
    native = model if learner=="native" else None
    values = input_view(reading,x,condition,center,native)
    output = model.predict(values) if native is not None else model.predict(values,thread_count=1)
    return np.maximum(0,output)


def load_bundle(content, trainer):
    bundle,trace = SklearnBackend(execution_enabled=True).load(content)
    learner,reading,weights,condition,center = bundle
    if learner=="native":
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/"published.npz"
            p.write_bytes(weights)
            model = trainer.Model.load(p)
    elif learner=="tree":
        model = weights
    else:
        raise ValueError("未登记的预测器")
    return (learner,reading,model,condition,center),trace


class EngineTrainingInquiry:
    def __init__(self,task,sample,source,parent,checksum,folder,trainer,explore,confirm,*,fixed=False,
                 incumbent=None):
        self.repair = SourceCombinations([r.values for r in sample])
        if (not sample or any(r.role!="fit" for r in source)
                or not {r.sample_id for r in sample}.issubset({r.sample_id for r in source})):
            raise ValueError("训练操作只能读取正式训练设备")
        owners = {r.unit_id for r in source}
        for batch,role in ((explore,"explore"),(confirm,"adopt")):
            if not batch or any(r.role!=role or r.unit_id in owners or r.label is None for r in batch):
                raise ValueError("探索、确认和训练设备必须分开")
        backend = SklearnBackend(execution_enabled=True)
        self.policy = "training-inquiry-fixed" if fixed else "training-inquiry"
        spec = {"policy":self.policy,"package":checksum,"parent":sha256(parent).hexdigest(),
                "repair":self.repair.identity,"sample":digest([r.sample_id for r in sample]),
                "explore":digest([asdict(r) for r in explore]),"confirm":digest([asdict(r) for r in confirm]),
                "implementation":[sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,training_inquiry.__file__)],
                "conditions":list(zip(task.condition_features,task.condition_decimals,strict=True)),
                "incumbent":incumbent.identity if incumbent else sha256(parent).hexdigest(),
                "backend":backend.specification}

        def run():
            by_unit = {}
            for r in source:
                by_unit.setdefault(r.unit_id,[]).append(r)
            opening = [r.values for rows in by_unit.values() for r in sorted(rows,key=lambda s:s.time_end)[:8]]
            cols = tuple(task.features.index(name) for name in task.condition_features)
            condition = ConditionStandardScaler(cols,tuple(task.condition_decimals)).fit(opening)
            x = {"sample":np.asarray([r.values for r in sample]),"full":np.asarray([r.values for r in source])}
            y = {"sample":np.asarray([r.label for r in sample]),"full":np.asarray([r.label for r in source])}
            center = StandardScaler().fit(condition.transform(x["sample"]))
            def feedback(rows):
                return LabeledBatch(self.repair.transform([r.values for r in rows]),np.array([r.label for r in rows]),
                                    tuple(r.unit_id for r in rows))
            with TemporaryDirectory() as tmp:
                p = Path(tmp)/"parent.npz"
                p.write_bytes(parent)
                initial = (("native","reference",incumbent.model,incumbent.condition,incumbent.center)
                           if incumbent else ("native","raw",trainer.Model.load(p),None,None))
                def build(trial):
                    if trial.learner=="native":
                        model = trainer.Model.load(p)
                        model.fit(input_view(trial.reading,x[trial.records],condition,center,model),y[trial.records],10)
                    else:
                        model = CatBoostRegressor(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1,
                                                  random_seed=17,thread_count=1,verbose=False,allow_writing_files=False)
                        model.fit(input_view(trial.reading,x[trial.records],condition,center),y[trial.records])
                    return trial.learner,trial.reading,model,condition,center
                chosen,trace = investigate(build,predict,initial,feedback(explore),feedback(confirm),
                                           len(sample),len(source),fixed=fixed)
                learner,reading,model,used_condition,used_center = chosen
                if learner=="native":
                    file = Path(tmp)/"published.npz"
                    model.save(file)
                    model = file.read_bytes()
                trace["evidence"] = {"explore_units":sorted({r.unit_id for r in explore}),
                                     "confirm_units":sorted({r.unit_id for r in confirm})}
                trace["incumbent_identity"] = spec["incumbent"]
                return backend.dump(((learner,reading,model,used_condition,used_center),trace))

        store = ModelArtifacts(folder)
        self.model_ref = store.create(spec,run)
        header,content = store.read(self.model_ref)
        self.identity = digest([spec,header["sha256"]])
        self.bundle,self.trace = load_bundle(content,trainer)

    @classmethod
    def from_public_engine(cls,task,sample,package,checksum,folder,trainer,explore,confirm,*,fixed):
        # 以此前可实际交付的六十遍模型为守擂版；挑战者拟合仍从同一官方原权重出发。
        incumbent = ContinuedConditionEngine(task,sample,package,checksum,folder,trainer)
        return cls(task,sample,package.train,package.checkpoint,checksum,folder,trainer,explore,confirm,
                   fixed=fixed,incumbent=incumbent)

    def advance_current(self,model,explore,confirm,serve):
        ids = [r["sample_id"] for r in serve]
        if not ids or len(ids)!=len(set(ids)) or any(r["label"] is not None for r in serve):
            raise ValueError("服务输入需要无答案的唯一记录")
        x = np.asarray([r["values"] for r in serve])
        result = predict(self.bundle,self.repair.transform(x))
        decision = {"chosen":self.trace["published"],"trace_sha256":digest(self.trace)}
        return dict(zip(ids,result.tolist(),strict=True)),dict(zip(ids,model.predict(x).tolist(),strict=True)),decision
