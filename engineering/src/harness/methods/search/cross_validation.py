"""探索资料很少时，在拟合用途内部比较配方，再生成一个完整候选。"""
from dataclasses import replace

from harness.contracts import ActionPlan
from harness.evaluation.metrics import prediction_score
from harness.models.pipeline_recipe import PipelineRecipe
from harness.data.splitting import validation_folds


def nominate_with_cv(engine, batch, rows, current):
    """每折重新构造处理器/预测器；采用资料始终由原控制器保存。

    默认按独立单位划折，也可显式选择带间隔的时间前缀；比较普通重训与稳健缩放。
    必须完成所有同样的折才参加排序；折内训练及评价经过共同预算账目。
    选中配方重新在全部合法fit资料拟合，再由新采用资料比较原模型。
    """
    try:
        folds = validation_folds(rows, folds=engine.config.fit_cv_folds,task=engine.task,
                                 mode=engine.config.fit_cv_mode,gap=engine.config.fit_cv_gap,
                                 action=engine._effective_action(ActionPlan("recent"), current.model_id))
    except ValueError as exc:
        engine._event(f"{batch.batch_id}:cv_unavailable", "cv_unavailable", reason=str(exc))
        return None, []
    robust = replace(current.pipeline or PipelineRecipe(), processor="robust")
    actions = [ActionPlan("recent",reason="小样本分组比较：普通重训"),
               ActionPlan("recent",reason="小样本分组比较：稳健缩放",pipeline=robust)]
    if current.pipeline == robust:
        actions = actions[:1]
    records = []
    for action in actions:
        effective = engine._effective_action(action, current.model_id)
        scores = []
        for i,(train,valid) in enumerate(folds):
            name = f"cv:{action.key}:{i}"
            artifact = engine._operation(batch,name+":build",3,5,
                lambda:engine.tools.build(engine.task,f"{batch.batch_id}/cv/{action.key}/{i}",effective,train),
                recovery=(lambda:engine.tools.recover_build(engine.task,f"{batch.batch_id}/cv/{action.key}/{i}",effective,train))
                    if hasattr(engine.tools,"recover_build") else None,
                requested_action=action.to_record(), effective_action=effective.to_record())
            if artifact is None:
                break
            def evaluate(predictions=None):
                inputs=tuple({"sample_id":s.sample_id,"values":list(s.values)} for s in valid)
                predictions=engine.tools.predict(artifact,inputs) if predictions is None else predictions
                if set(predictions)!={s.sample_id for s in valid} or any(not engine.task.accepts_target(v) for v in predictions.values()):
                    raise ValueError("折内预测无效")
                return prediction_score(engine.task,tuple(replace(s,role="explore") for s in valid),predictions)
            def recover_score():
                inputs=tuple({"sample_id":s.sample_id,"values":list(s.values)} for s in valid)
                receipt=engine.tools.recover_predict(artifact,inputs)
                if receipt and receipt["status"]=="completed":
                    return {**receipt,"artifact":evaluate(receipt["artifact"])}
                return receipt
            score=engine._operation(batch,name+":score",1,5,evaluate,
                recovery=recover_score if hasattr(engine.tools,"recover_predict") else None)
            if score is None:
                break
            scores.append({"score":score,"train":[s.sample_id for s in train],"valid":[s.sample_id for s in valid]})
        record={"action":action.to_record(),"effective_action":effective.to_record(),
                "folds":scores,"complete":len(scores)==len(folds)}
        if record["complete"]:
            record["mean_utility"]=sum(engine.task.utility(s["score"]) for s in scores)/len(scores)
        records.append(record)
    complete=[r for r in records if r["complete"]]
    engine.state.setdefault("cross_validation",{})[batch.batch_id]=records
    engine._event(f"{batch.batch_id}:cv_compared","cv_compared",records=records)
    if not complete:
        return None, records
    best=max(complete,key=lambda r:r["mean_utility"])
    action=ActionPlan.parse(best["action"])
    bundle=engine._build(batch,action,rows,current.model_id)
    return bundle, records
