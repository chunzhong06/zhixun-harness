"""发动机固定参照。源设备前八个循环估计工况尺度，逐条服务只读传感器与设置。"""
from dataclasses import replace

import numpy as np
from sklearn.pipeline import Pipeline

from harness.core.contracts import ActionPlan, digest
from harness.learning.models.condition_processing import ConditionStandardScaler
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.core.recipe import PipelineRecipe
from harness.learning.models.source_combinations import SourceCombinations
from harness.learning.models.training_tools import SklearnBackend


class EarlyEngineReference:
    def __init__(self,task,training,source,checksum,folder):
        self.repair = SourceCombinations([r.values for r in training])
        groups = {}
        for r in source:
            if r.role!="fit":
                raise ValueError("参照需要源端训练观测")
            groups.setdefault(r.unit_id,[]).append(r)
        if set(groups)!={r.unit_id for r in training} or any(len(rows)<8 for rows in groups.values()):
            raise ValueError("训练与早期参照需要相同源设备及完整起始观测")
        early = [r for rows in groups.values() for r in sorted(rows,key=lambda v:v.time_end)[:8]]
        scaler = ConditionStandardScaler((21,22,23),tuple(task.condition_decimals)).fit([r.values for r in early])
        backend = SklearnBackend(execution_enabled=True)
        recipe = PipelineRecipe(predictor="catboost",processor="none",max_depth=6)
        spec = {"policy":"early-reference","package":checksum,"repair":self.repair.identity,
                "training":digest([r.sample_id for r in training]),"early":digest([r.sample_id for r in early]),
                "recipe":recipe.key,"condition_decimals":list(task.condition_decimals),"backend":backend.specification}

        def build():
            x = scaler.transform([r.values for r in training])
            rows = [replace(r,values=tuple(v)) for r,v in zip(training,x.tolist(),strict=True)]
            tree = backend.fit(task,"source",ActionPlan("balanced",pipeline=recipe),rows)
            return backend.dump(Pipeline([("condition",scaler),("tree",tree)]))

        store = ModelArtifacts(folder)
        self.model_ref = store.create(spec,build)
        header,blob = store.read(self.model_ref)
        self.pipeline = backend.load(blob)
        self.identity = digest([spec,header["sha256"]])

    @staticmethod
    def predict(pipeline,repair,x):
        normalized = pipeline.named_steps["condition"].transform(repair.transform(x))
        return pipeline.named_steps["tree"].predict(normalized,thread_count=1)

    def advance_current(self,model,explore,confirm,serve):
        ids = [r["sample_id"] for r in serve]
        if not ids or len(set(ids))!=len(ids) or any(r["label"] is not None for r in serve):
            raise ValueError("发动机预测需要无标签的唯一测量")
        x = np.asarray([r["values"] for r in serve])
        p = self.predict(self.pipeline,self.repair,x)
        return (dict(zip(ids,p.tolist(),strict=True)),dict(zip(ids,model.predict(x).tolist(),strict=True)),
                {"chosen":"early-reference"})
