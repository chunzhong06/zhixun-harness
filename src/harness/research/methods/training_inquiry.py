"""用实际试验结果组合下一对训练操作；探索数据提名，另一组数据仅验一次。"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Trial:
    learner: str
    reading: str
    records: str


@dataclass(frozen=True)
class LabeledBatch:
    inputs: np.ndarray
    labels: np.ndarray
    groups: tuple

    def score(self, values):
        target = np.asarray(self.labels, dtype=float)
        values = np.asarray(values, dtype=float)
        groups = np.asarray(self.groups)
        if (target.ndim!=1 or not len(target) or values.shape!=target.shape or groups.shape!=target.shape
                or len(self.inputs)!=len(target) or not np.isfinite(values).all() or not np.isfinite(target).all()):
            raise ValueError("评价需要逐行对应的有限预测、答案和设备或采集组")
        squared = (values-target)**2
        return {"error":float(np.sqrt(squared.mean())),
                "worst":max(float(np.sqrt(squared[groups==u].mean())) for u in np.unique(groups))}


def propose(pilot_scores, sampled_rows, available_rows, fixed=False):
    """输入只有先导分数和可用训练量；确认组及服务组均不参与提出解释。"""
    gains = {}
    for learner in ("native","tree"):
        raw = pilot_scores[Trial(learner,"raw","sample")]["error"]
        ref = pilot_scores[Trial(learner,"reference","sample")]["error"]
        gains[learner] = raw-ref
    best = min(pilot_scores,key=lambda trial:pilot_scores[trial]["error"])
    disagree = (gains["native"]>0)!=(gains["tree"]>0)
    if fixed or (disagree and available_rows>sampled_rows):
        reason = ("固定顺序优先比较原模型的两种读法" if fixed else
                  "两种预测器对参照的反应方向不同，扩大训练记录后复查原模型")
        pair = [Trial("native",reading,"full") for reading in ("raw","reference")]
        factor = "reading"
    elif gains["native"]>0 and gains["tree"]>0:
        reason = "两个预测器都从参照受益，在完整训练记录上检验哪个预测器更适合这种读法"
        pair = [Trial(learner,"reference","full") for learner in ("native","tree")]
        factor = "learner"
    else:
        reason = "参照没有产生共同收益，保留先导中误差较低的读法，在全部记录上比较预测器"
        pair = [Trial(learner,best.reading,"full") for learner in ("native","tree")]
        factor = "learner"
    return pair,{"hypothesis":reason,"vary":factor,"reference_gain":gains,
                 "sampled_rows":sampled_rows,"available_rows":available_rows,
                 "proposed_trials":[asdict(t) for t in pair]}


def investigate(build, predict, incumbent, explore, confirm, sampled_rows, available_rows, *, fixed=False):
    """先真实构建四个小模型，再执行提案中的两个对照；确认失败原样返回在用模型。"""
    if set(explore.groups)&set(confirm.groups):
        raise ValueError("探索和确认必须来自不同设备或采集组")
    pilot = [Trial(l,r,"sample") for l in ("native","tree") for r in ("raw","reference")]
    scored,models,events = {},{},[]
    def execute(trials):
        for trial in trials:
            fitted = build(trial)
            metrics = explore.score(predict(fitted,explore.inputs))
            models[trial],scored[trial] = fitted,metrics
            events.append({"trial":asdict(trial),"exploration":metrics})

    execute(pilot)
    trials,proposal = propose(scored,sampled_rows,available_rows,fixed)
    execute(trials)
    first,second = trials
    proposal["followup_difference"] = scored[first]["error"]-scored[second]["error"]
    proposal["lower_error_trial"] = asdict(min(trials,key=lambda t:scored[t]["error"]))
    if first.learner==second.learner:
        proposal["corresponding_pilot_difference"] = proposal["reference_gain"][first.learner]
    else:
        proposal["corresponding_pilot_difference"] = (
            scored[Trial(first.learner,first.reading,"sample")]["error"]-
            scored[Trial(second.learner,second.reading,"sample")]["error"])
    best = min(scored,key=lambda trial:scored[trial]["error"])
    incumbent_score = confirm.score(predict(incumbent,confirm.inputs))
    challenger_score = confirm.score(predict(models[best],confirm.inputs))
    accepted = (challenger_score["error"]<incumbent_score["error"] and
                challenger_score["worst"]<=1.01*incumbent_score["worst"])
    trace = {"pilot":events[:len(pilot)],"proposal":proposal,"followup":events[len(pilot):],
             "nomination":asdict(best),"confirmation":{"incumbent":incumbent_score,"challenger":challenger_score},
             "accepted":accepted,"published":asdict(best) if accepted else "incumbent",
             "fit_count":len(events),"fixed_order":fixed}
    return (models[best] if accepted else incumbent),trace
