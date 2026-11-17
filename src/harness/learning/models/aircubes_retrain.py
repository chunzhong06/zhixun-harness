"""在源期训练两只电极的非线性校准曲线，部署后只凭已返回的同仪器参考重估原始读数。"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.learning.models.aircubes_pair import PairedMaintenance

PRIMARY="按本机近期分歧与回标在原始电极上重新标定"
INCUMBENT="固定现场配对记忆0.95"
VIEWS=("电极一","电极二","双电极")
MEMORIES=(.88,.97,1.)


def features(z,j):
    v=z[:,j]
    temp,humidity,drh=z[:,2:].T
    env=[temp,humidity,drh,temp**2,humidity**2,temp*humidity,drh*temp]
    terms=[np.ones(len(z)),v,v**2,v*temp,v*humidity,v*drh,*env]
    if j==2:
        # 两路都可见；工况与共模响应、差模响应的交互分别表示灵敏度变化。
        v=(z[:,0]+z[:,1])/2
        gap=z[:,0]-z[:,1]
        terms=[np.ones(len(z)),v,v**2,v*temp,v*humidity,v*drh,gap,gap*temp,gap*humidity,*env]
    return np.column_stack(terms)


class ElectrodeRelease:
    def fit(self,x,y):
        self.scaler=StandardScaler().fit(x)
        z=np.clip(self.scaler.transform(x),-7,7)
        self.source=[]
        for i in range(3):
            a=features(z,i)
            gram=a.T@a/len(y)
            rhs=a.T@y/len(y)
            reg=np.eye(a.shape[1])*.002
            coef=np.linalg.solve(gram+reg,rhs)
            self.source.append((gram+reg,coef))
        return self


class ElectrodeMaintenance:
    def __init__(self,release):
        self.release=release
        self.old=PairedMaintenance()
        self.grams=[np.zeros((3,g.shape[0],g.shape[1])) for g,_ in release.source]
        self.rhs=[np.zeros((3,len(c))) for _,c in release.source]
        self.recent=[]
        self.global_loss=np.full(10,12.)
        self.names=[f"{v}原始重训记忆{d:g}" for v in VIEWS for d in MEMORIES]

    def predict(self,raw,baseline):
        previous=self.old.predict(baseline,raw[:,2])
        peer=previous[INCUMBENT]
        z=np.clip(self.release.scaler.transform(raw),-7,7)
        predictions=[];self.last=[]
        for j,(g,initial) in enumerate(self.release.source):
            f=features(z,j)
            # 相当于用十二个源期样本的二阶矩保留校准起点；现场只加入合法真值。
            coeff=np.linalg.solve(self.grams[j]+12*g,self.rhs[j][...,None])[...,0]
            pred=np.maximum(0,f@(initial+coeff).T)
            predictions.append(pred)
            self.last.append((f[-1],float(f[-1]@initial)))
        bank=np.column_stack((peer,*predictions))
        self.ctx=np.column_stack(((baseline[:,0]-baseline[:,1])/6,
            (baseline[:,2]-baseline[:,3])/6,(peer-15)/15,z[:,2],z[:,4])).clip(-5,5)
        if self.recent:
            ref,loss=zip(*self.recent,strict=True)
            dist=np.mean((self.ctx[:,None,:]-np.array(ref)[None,:,:])**2,axis=2)
            recency=.97**np.arange(len(ref)-1,-1,-1)
            w=np.exp(-2*dist)*recency
            expected=(w@np.array(loss)+5*self.global_loss)/(w.sum(1)[:,None]+5)
        else:
            expected=np.broadcast_to(self.global_loss,bank.shape)
        self.last_bank=bank[-1].copy()
        weight=(1+expected)**-2
        global_w=(1+self.global_loss)**-2
        out={**dict(zip(self.names,bank[:,1:].T,strict=True)),
             PRIMARY:(weight*bank).sum(1)/weight.sum(1),
             "原始重训只按本机平均误差选":bank@global_w/global_w.sum(),
             "原始重训固定等权":bank.mean(1)}
        for i,name in enumerate(self.names,1):
            out["同工具论文与"+name+"各半"]=(peer+bank[:,i])/2
        return {**previous,**out}

    def update(self,y):
        self.old.update(y)
        loss=np.minimum((self.last_bank-y)**2,625)
        self.global_loss=.96*self.global_loss+.04*loss
        self.recent.append((self.ctx[-1].copy(),loss))
        self.recent=self.recent[-100:]
        decay=np.array(MEMORIES)
        for j,(x,initial) in enumerate(self.last):
            self.grams[j]=decay[:,None,None]*self.grams[j]+np.outer(x,x)
            self.rhs[j]=decay[:,None]*self.rhs[j]+x*(y-initial)
