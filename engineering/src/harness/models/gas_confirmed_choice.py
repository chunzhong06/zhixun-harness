"""用当窗探索行选择分类修正，再用独立确认行决定哪些修正可以实际交付。"""
import numpy as np

from harness.models.feedback_routing import FeedbackRouter, apply_choices
from harness.models.vote_feedback import majority


MODES=("fixed-vote","vote-confirmed","vote-by-class","explore-only","class-confirmed","score-confirmed")


def candidates(predictions):
    return {**predictions, "vote": majority(predictions, 6)}


def macro_f1(truth, pred):
    matrix=np.zeros((6,6),int)
    np.add.at(matrix,(np.asarray(truth,dtype=int)-1,np.asarray(pred,dtype=int)-1),1)
    support=matrix.sum(0)+matrix.sum(1)
    return np.mean(2*matrix.diagonal()[support>0]/support[support>0])


def choose(mode, explore, y_explore, confirm, y_confirm):
    base=["current"]*6
    if mode not in MODES:
        raise ValueError("未知气体确认策略")
    if mode=="fixed-vote":
        return ["vote"]*6
    if not len(y_explore) or not len(y_confirm):
        return base
    if mode=="vote-confirmed":
        wins=all(macro_f1(y,p["vote"])>macro_f1(y,p["current"]) for p,y in
                 ((explore,y_explore),(confirm,y_confirm)))
        return ["vote"]*6 if wins else base
    names=("current","vote") if mode=="vote-by-class" else tuple(explore)
    selector=FeedbackRouter(names,6)
    selector.update(y_explore,explore)
    chosen=selector.choices("local")
    if mode=="explore-only":
        return chosen
    y=np.asarray(y_confirm)
    for label,name in enumerate(chosen,1):
        mask=confirm["current"]==label
        gain=int(np.sum(confirm[name][mask]==y[mask]))-int(np.sum(confirm["current"][mask]==y[mask]))
        if gain<1:
            chosen[label-1]="current"
    if mode=="score-confirmed" and macro_f1(y,apply_choices(confirm,chosen))<=macro_f1(y,confirm["current"]):
        return base
    return chosen
