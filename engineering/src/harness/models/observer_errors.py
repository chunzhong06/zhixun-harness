"""按已返回的真类统计各分类器的混淆率。每个服务批使用上一批结束时的计数。"""
import numpy as np

from harness.models.vote_feedback import majority


class ObserverErrors:
    def __init__(self,names,classes,counts=None):
        self.names=tuple(names)
        self.classes=classes
        shape=(len(names),classes,classes)
        self.counts=np.zeros(shape,dtype=np.int64) if counts is None else np.array(counts,dtype=np.int64)
        if self.counts.shape!=shape or (self.counts<0).any():
            raise ValueError("已有分类器的混淆计数无效")
        if not np.all(self.counts.sum(2)==self.counts[0].sum(1)):
            raise ValueError("各分类器返回的真类数不同")

    def predict(self,outputs,mode):
        if mode not in ("uniform","frequency","accuracy"):
            raise ValueError("未知的观察者计分方式")
        base=majority(outputs,self.classes)
        if not self.counts.any():
            return base
        votes=np.stack([outputs[k] for k in self.names]).astype(int)-1
        smooth=self.counts+1+np.eye(self.classes,dtype=int)[None,:,:]
        if mode=="accuracy":
            weights=(np.trace(self.counts,axis1=1,axis2=2)+1)/(self.counts.sum(axis=(1,2))+2)
            table=np.stack([np.sum(weights[:,None]*(votes==g),axis=0) for g in range(self.classes)],axis=1)
        else:
            logp=np.log(smooth/smooth.sum(2,keepdims=True))
            table=sum(logp[j].T[v] for j,v in enumerate(votes))
            if mode=="frequency":
                table+=np.log(self.counts[0].sum(1)+1)
        maximum=table.max(1)
        ties=np.isclose(table,maximum[:,None],atol=1e-12,rtol=0)
        return np.where(ties[np.arange(len(base)),base-1],base,ties.argmax(1)+1)

    def update(self,answers,outputs):
        y=np.asarray(answers,dtype=int)-1
        votes=[np.asarray(outputs[k],dtype=int)-1 for k in self.names]
        if y.ndim!=1 or any(v.shape!=y.shape for v in votes) or any(np.any((v<0)|(v>=self.classes)) for v in [y,*votes]):
            raise ValueError("回标数、类别或行序不符")
        for j,v in enumerate(votes):
            np.add.at(self.counts[j],(y,v),1)
