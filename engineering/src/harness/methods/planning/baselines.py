"""固定维护对照使用与其他计划器相同的观察和动作接口。"""
from harness.contracts import ActionPlan
from harness.models.pipeline_recipe import PipelineRecipe


class FixedMaintenancePlanner:
    """研究对照：保持原模型，或每批提出一次固定近期重训。"""
    def __init__(self, retrain: bool):
        self.retrain = retrain
        self.identity = f"fixed-maintenance-v1:{retrain}"

    def propose(self, context):
        if not self.retrain or context["trials"]:
            return {"actions": [], "stop": True}
        ref = next(item["id"] for item in context["facts"]["evidence"] if item["kind"] == "performance")
        return {"actions": [{"recipe": "recent", "ratio": 0., "reason": "固定对照：每批近期重训，再按共同评价决定采用",
                             "evidence_refs": [ref]}], "stop": False}


class FixedComponentPlanner:
    """固定顺序参照拥有相同新工具；预算限制实际能完成的子集，不读结果改顺序。"""
    identity = "fixed-component-search-v2"

    def propose(self, context):
        ref = next(item["id"] for item in context["facts"]["evidence"] if item["kind"]=="performance")
        header = {"reason":"固定目录中的可执行改法，按相同探索资料比较","evidence_refs":[ref]}
        tried = {trial["action"] for trial in context["trials"]}
        pipes = [None, PipelineRecipe(processor="robust"), PipelineRecipe(predictor="forest", trees=32),
                 PipelineRecipe(processor="none", predictor="catboost", trees=32)]
        if context["objective"]["task_type"] == "classification":
            pipes += [PipelineRecipe(class_weight="balanced"),
                      PipelineRecipe(processor="none", predictor="catboost", trees=32, class_weight="balanced")]
        else:
            pipes += [PipelineRecipe(processor="none", predictor="catboost", trees=32, regression_loss="mae")]
        remaining = context["maintenance"]["decision"]["remaining_trials"]
        actions = [ActionPlan("recent",reason=header["reason"],evidence_refs=(ref,),pipeline=pipe)
                   for pipe in pipes]
        actions = [a.to_record() for a in actions if a.key not in tried][:min(3, remaining)]
        if not actions:
            return {"operation":"finish",**header}
        return {"operation":"experiment",**header,"actions":actions}


class FixedHistoryPlanner:
    """同一预测器固定比较三种资料配方；提名与采用仍由公共控制器完成。"""
    identity = "fixed-history-maintenance-v1"

    def propose(self, context):
        if context["trials"]:
            return {"actions": [], "stop": True}
        ref = next(item["id"] for item in context["facts"]["evidence"] if item["kind"] == "performance")
        actions = [ActionPlan(recipe, ratio, "固定比较近期与两种历史回放", (ref,)).to_record()
                   for recipe, ratio in (("recent", 0.), ("uniform", 1.), ("coverage", 1.))]
        return {"actions": actions, "stop": False}
