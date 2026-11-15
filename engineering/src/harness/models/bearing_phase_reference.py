"""用冻结源设备总寿命将成熟树的阶段预测换算为分钟。服务仅需本条 34 项测量。"""
from dataclasses import replace

import numpy as np
from harness.contracts import ActionPlan, digest
from harness.models.model_artifacts import ModelArtifacts
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend


class BearingPhaseReference:
    def __init__(self,task,training,checksum,folder):
        if task.target_unit!="minutes" or not training or any(r.role!="fit" for r in training):
            raise ValueError("阶段参照需要以分钟计的轴承源训练行")
        total={}
        endpoints=set()
        starts=set()
        for r in training:
            duration=r.label+r.time_end-1
            if duration<=0 or total.setdefault(r.unit_id,duration)!=duration:
                raise ValueError("训练设备的总时长无效")
            if r.label==0:
                endpoints.add(r.unit_id)
            if r.time_end==1:
                starts.add(r.unit_id)
        if endpoints!=set(total) or starts!=set(total):
            raise ValueError("阶段参照需要源设备的起止观测")
        self.scale=float(np.median(list(total.values())))
        fit=[replace(r,label=r.label/total[r.unit_id]) for r in training]
        recipe=PipelineRecipe(predictor="catboost",processor="none",max_depth=6)
        backend=SklearnBackend(execution_enabled=True)
        spec={"policy":"phase-fixed","package":checksum,"source":digest([(r.sample_id,r.label,r.values) for r in fit]),
              "scale":self.scale,"recipe":recipe.key,"backend":backend.specification}
        store=ModelArtifacts(folder)
        self.model_ref=store.create(spec,lambda:backend.dump(backend.fit(task,"source",ActionPlan("balanced",pipeline=recipe),fit)))
        header,blob=store.read(self.model_ref)
        self.tree=backend.load(blob)
        self.identity=digest([spec,header["sha256"]])

    def advance_current(self,model,explore,confirm,serve):
        ids=[r["sample_id"] for r in serve]
        if not ids or len(ids)!=len(set(ids)) or any(r["label"] is not None for r in serve):
            raise ValueError("阶段预测需要本轮唯一无标签的轴承测量")
        x=np.asarray([r["values"] for r in serve])
        pred=np.clip(self.tree.predict(x,thread_count=1),0.,1.)*self.scale
        return (dict(zip(ids,pred.tolist(),strict=True)),dict(zip(ids,model.predict(x).tolist(),strict=True)),
                {"chosen":"phase-fixed","scale_minutes":self.scale})
