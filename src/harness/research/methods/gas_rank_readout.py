"""用单传感器读数在其自身分布中的位置估计各类概率，再用历史事前误差选读法。"""
import numpy as np


def ranks(reference, values):
    result=np.empty(values.shape,dtype=float)
    for j in range(values.shape[1]):
        sorted_values=np.sort(reference[:,j])
        lo=np.searchsorted(sorted_values,values[:,j],side="left")
        hi=np.searchsorted(sorted_values,values[:,j],side="right")
        result[:,j]=(lo+hi+1)/(2*(len(reference)+1))
    return result


def probability_bank(features, labels, query, classes):
    """每个通道把源端或现场的已标样本压缩为32个有序区段，保留每类的实测分布。"""
    grid=np.linspace(1/64,1-1/64,32)
    indices=np.minimum((features*32).astype(int),31)
    bank=[]
    for col in range(features.shape[1]):
        table=np.full((32,classes),.05)
        np.add.at(table,(indices[:,col],labels-1),1)
        # 让各类拥有相同的先验权重，避免以查询样本的类别比例当作现场发生率。
        table/=table.sum(axis=0,keepdims=True)
        weights=np.exp(-.5*((query[:,col,None]-grid)/.09)**2)
        p=weights@table
        bank.append(p/p.sum(axis=1,keepdims=True))
    return np.stack(bank,axis=1)


class RankReadout:
    def __init__(self,source,labels):
        self.classes=len(np.unique(labels))
        self.source=ranks(source,source)
        self.labels=labels

    def predict(self, query, observed, history, answers):
        qr=ranks(observed,query)
        old=probability_bank(self.source,self.labels,qr,self.classes)
        local=old
        if len(history) and len(np.unique(answers))==self.classes:
            local=probability_bank(ranks(observed,history),answers,qr,self.classes)
        return np.concatenate((old,local),axis=1)


def aggregate(bank, past, answers):
    half=bank.shape[1]//2
    result={"源秩等权":bank[:,:half].mean(1),"现场秩等权":bank[:,half:].mean(1),
            "两域秩固定各半":bank.mean(1)}
    prior=np.full(bank.shape[1],1/bank.shape[1])
    if len(answers):
        hist=np.concatenate(past,axis=0)
        true=np.asarray(answers)
        cls=np.unique(true)
        losses=-np.log(np.take_along_axis(hist,true[:,None,None]-1,axis=2).squeeze(2).clip(1e-5))
        score=np.mean([losses[true==c].mean(0) for c in cls],axis=0)
        evidence=len(true)/(len(true)+24.)
        strength=10*evidence
        prior=np.exp(-strength*(score-score.min()))
        prior/=prior.sum()
    result["按现场通道与域的可靠性"]=np.einsum("ntc,t->nc",bank,prior)
    return result,prior
