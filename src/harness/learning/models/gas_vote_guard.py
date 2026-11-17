"""按过去已经交付的预测检验投票纠错，累计真实回标决定何时启用。"""
import numpy as np

from harness.learning.models.feedback_routing import apply_choices
from harness.learning.models.vote_feedback import VoteFeedbackRouter


class GuardedVote:
    def __init__(self, names, classes, state=None):
        self.router = VoteFeedbackRouter(names, classes, state["router"] if state else None)
        self.net = state["net"] if state else 0
        self.disagreements = state["disagreements"] if state else 0

    def choices(self, mode):
        if mode != "guarded-vote":
            raise ValueError("未知投票启用方式")
        if self.net >= 2 and self.disagreements >= 4:
            return self.router.choices("vote-local")
        return ["original"] * self.router.classes

    def bank(self, outputs):
        return self.router.bank(outputs)

    @property
    def seen(self):
        return self.router.seen

    def update(self, answers, outputs):
        # 先用当时的规则复原竞争预测，再让这一批答案进入下一批的规则。
        routed = apply_choices(self.bank(outputs), self.router.choices("vote-local"))
        base = outputs["current"]
        truth = np.asarray(answers)
        self.net += int((routed == truth).sum()) - int((base == truth).sum())
        self.disagreements += int((routed != base).sum())
        self.router.update(truth, outputs)

    def state(self):
        return {"net": self.net, "disagreements": self.disagreements, "router": self.router.state()}
