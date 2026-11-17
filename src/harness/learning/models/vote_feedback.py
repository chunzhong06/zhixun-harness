"""用固定投票启动，并在收到旧批反馈后修正某些类别。"""
import numpy as np

from harness.learning.models.feedback_routing import FeedbackRouter


def majority(predictions,classes):
    base=np.asarray(predictions["current"],dtype=int)
    voters=np.stack(list(predictions.values()))
    votes=np.stack([(voters==g).sum(axis=0) for g in range(1,classes+1)],axis=1)
    tied=votes[np.arange(len(base)),base-1]==votes.max(axis=1)
    return np.where(tied,base,votes.argmax(axis=1)+1)


class VoteFeedbackRouter:
    """以固定多数票开始，通过已回标记录修正它。按到达序号隔开探索和确认。"""
    def __init__(self,names,classes,state=None):
        if names[0]!="current":
            raise ValueError("第一项须为原始分类器")
        self.classes=classes
        self.names=("current","original",*names[1:])
        self.seen=state["seen"] if state else 0
        self.halves=[FeedbackRouter(self.names,classes,c) for c in (state["counts"] if state else (None,None))]
        if sum(s.counts[0].sum() for s in self.halves)!=self.seen:
            raise ValueError("反馈记录缺失或重复")

    def bank(self,outputs):
        return {"current":majority(outputs,self.classes),"original":outputs["current"],
                **{k:v for k,v in outputs.items() if k!="current"}}

    def choices(self,mode):
        left,right=self.halves
        if mode=="vote":
            return ["current"]*self.classes
        if mode=="vote-local":
            return FeedbackRouter(self.names,self.classes,left.counts+right.counts).choices("local")
        if mode!="vote-confirm":
            raise ValueError("未知投票确认规则")
        choices=left.choices("local")
        for label,name in enumerate(choices):
            counts=right.counts[self.names.index(name),label]
            net=np.trace(counts)-counts[:,label].sum()
            if net<1:
                choices[label]="current"
        return choices

    def update(self,y,candidates):
        y=np.asarray(y,dtype=int)
        bank=self.bank(candidates)
        for j,router in enumerate(self.halves):
            select=(np.arange(len(y))+self.seen)%2==j
            router.update(y[select],{k:v[select] for k,v in bank.items()})
        self.seen+=len(y)

    def state(self):
        return {"seen":self.seen,"counts":[r.counts.tolist() for r in self.halves]}
