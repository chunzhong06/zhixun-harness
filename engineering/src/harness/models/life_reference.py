"""在相同工业预测上比较固定寿命参照。只允许探索赢家进入本窗确认。"""
import numpy as np

from harness.contracts import ActionPlan,digest
from harness.models.model_artifacts import ModelArtifacts
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend


MODES = {"select":("current","median","catboost-only","catboost-half"),
         "median-confirm":("current","median"),"tree-confirm":("current","catboost-only")}


def alternatives(base,tree,median):
    base = np.asarray(base,dtype=float)
    tree = np.asarray(tree,dtype=float)
    if base.ndim!=1 or base.shape!=tree.shape or not np.isfinite(base).all() or not np.isfinite(tree).all() or not np.isfinite(median):
        raise ValueError("寿命参照需要同一组有限一维预测")
    return {"current":base,"median":np.full(base.shape,median),"catboost-only":tree,"catboost-half":.5*(base+tree)}


def choose(explore,confirm,truth_explore,truth_confirm,mode):
    names=MODES[mode]
    if not len(truth_explore) or not len(truth_confirm):
        return {"chosen":"current","proposed":None,"reason":"缺少独立反馈"}
    def rmse(y,v):
        y=np.asarray(y,dtype=float)
        if y.shape!=v.shape or not np.isfinite(y).all():
            raise ValueError("反馈标签未与此分片的预测对齐")
        return float(np.sqrt(np.mean((y-v)**2)))
    scores={k:rmse(truth_explore,explore[k]) for k in names}
    best=min(names,key=scores.get)
    if best=="current" or scores[best]+1e-12>=scores["current"]:
        return {"chosen":"current","proposed":None,"explore":scores,"reason":"探索保留原版"}
    before,after=[rmse(truth_confirm,confirm[k]) for k in ("current",best)]
    return {"chosen":best if after+1e-12<before else "current","proposed":best,
            "explore":scores,"confirmation":[before,after],"reason":"独立反馈比较"}


class LifeReference:
    """训练库只读；常数分支无需训练第三方模型，树模型保存为可信本地产物。"""
    def __init__(self,task,training,checksum,folder,policy):
        if not training or any(r.role!="fit" or r.label is None for r in training):
            raise ValueError("参照只能读取官方 fit")
        self.median=float(np.median([r.label for r in training]))
        self.policy=policy
        self.tree=None
        recipe=PipelineRecipe(processor="none",predictor="catboost",max_depth=6)
        bank=digest([(r.sample_id,r.label,r.values) for r in training])
        spec={"package":checksum,"fit":bank,"median":self.median,"policy":policy}
        if policy in ("tree-confirm","catboost-only","select"):
            backend=SklearnBackend(execution_enabled=True)
            spec.update(recipe=recipe.key,backend=backend.specification)
            store=ModelArtifacts(folder)
            ref=store.create(spec,lambda:backend.dump(backend.fit(task,"source",ActionPlan("balanced",pipeline=recipe),training)))
            manifest,blob=store.read(ref)
            self.tree=backend.load(blob)
            self.model_ref=ref
            spec["model_sha256"]=manifest["sha256"]
        else:
            self.model_ref=None
        self.identity=digest(spec)

    def advance_current(self,model,explore,confirm,serve):
        keys=[r["sample_id"] for group in (explore,confirm,serve) for r in group]
        if not serve or len(keys)!=len(set(keys)) or any(r["label"] is not None for r in serve):
            raise ValueError("服务数据有标签或分片有重叠")
        def predictions(rows):
            if not rows:
                return {}
            x=np.asarray([r["values"] for r in rows])
            base=model.predict(x)
            if not np.isfinite(base).all():
                raise ValueError("工业模型返回非有限预测")
            if self.tree is None:
                return {"current":base,"median":np.full(base.shape,self.median)}
            return alternatives(base,self.tree.predict(x,thread_count=1),self.median)
        if self.policy in MODES:
            decision=choose(predictions(explore),predictions(confirm),[r["label"] for r in explore],
                            [r["label"] for r in confirm],self.policy)
        else:
            decision={"chosen":self.policy,"reason":"固定参照"}
        options=predictions(serve)
        ids=[r["sample_id"] for r in serve]
        return (dict(zip(ids,options[decision["chosen"]].tolist(),strict=True)),
                dict(zip(ids,options["current"].tolist(),strict=True)),decision)
