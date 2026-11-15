"""依据此前回标的预测选择分类候选，持久状态只需要小型分类混淆计数。"""
import numpy as np


class FeedbackRouter:
    def __init__(self,names,classes,counts=None):
        self.names=tuple(names)
        self.classes=int(classes)
        if not names or names[0]!="current" or len(set(names))!=len(names) or classes<2:
            raise ValueError("分类路由需要约定顺序的候选和基础版")
        shape=(len(names),classes,classes,classes)
        self.counts=np.zeros(shape,dtype=np.int64) if counts is None else np.asarray(counts,dtype=np.int64)
        if self.counts.shape!=shape or np.any(self.counts<0):
            raise ValueError("持久分类计数无效")

    def update(self,answers,outputs):
        y=np.asarray(answers,dtype=int)-1
        base=np.asarray(outputs["current"],dtype=int)-1
        for j,k in enumerate(self.names):
            pred=np.asarray(outputs[k],dtype=int)-1
            if (y.ndim!=1 or base.shape!=y.shape or pred.shape!=y.shape or
                    any(np.any((v<0)|(v>=self.classes)) for v in (base,pred,y))):
                raise ValueError("回标行与已交付各候选的类别或顺序不符")
        for j,k in enumerate(self.names):
            np.add.at(self.counts[j],(base,np.asarray(outputs[k],dtype=int)-1,y),1)

    def choices(self,mode):
        if mode=="global":
            matrix=self.counts.sum(axis=1)
            den=matrix.sum(axis=1)+matrix.sum(axis=2)
            f1=np.divide(2*np.diagonal(matrix,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0)
            scores=f1.sum(axis=1)/np.maximum(1,(den>0).sum(axis=1))
            return [self.names[int(np.argmax(scores))]]*self.classes
        if mode not in ("local","relabel"):
            raise ValueError("未知候选选择方式")
        result=[]
        for g in range(self.classes):
            counts=self.counts[:,g,:,:]
            if mode=="relabel":
                returned=counts[0].sum(axis=0)
                dest=int(np.argmax(returned))
                allow=(dest!=g and returned.sum()>=4 and returned[dest]-returned[g]>=2)
                result.append(f"label-{dest+1}" if allow else "current")
                continue
            benefit=np.trace(counts,axis1=1,axis2=2)-counts[:,:,g].sum(axis=1)
            disagreements=counts.sum(axis=(1,2))-counts[:,g,:].sum(axis=1)
            gain=np.where((benefit>=2)&(disagreements>=4),benefit,0)
            result.append(self.names[int(np.argmax(gain))])
        return result


def apply_choices(predictions,choices):
    base=np.asarray(predictions["current"],dtype=int)
    selected=base.copy()
    for label,choice in enumerate(choices,1):
        mask=base==label
        if choice.startswith("label-"):
            selected[mask]=int(choice.split("-")[1])
        else:
            selected[mask]=np.asarray(predictions[choice])[mask]
    return selected
