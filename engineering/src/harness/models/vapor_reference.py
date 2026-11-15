"""以最后已返回的参考读数为起点，学习慢探头后来看到的变化。"""
import numpy as np
from scipy.linalg import solve
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness.models.vapor_field import OnlineVoltage

PRIMARY="依上次参照与当下变化的装机误差决定发布"
ANCHORS=("通道相对变化","论文状态相对变化","两者相对变化合用")
READERS=tuple(f"{k}/{p}" for k in ANCHORS for p in ("源端","累计回标","近期回标"))


def relative(views,indices,previous,anchor):
    # 服务窗口的前一批参考身份恒早于当前窗口，不用当前批其他窗的PID。
    if np.any(indices<=previous):raise ValueError("参考必须早于当前输出")
    anchor=np.broadcast_to(anchor,len(indices))
    lag=(indices-previous)*.5
    components=[]
    for field in (4,5):
        x=views[field][indices]
        delta=x-views[field][previous]
        components.append(np.column_stack((delta,delta/lag[:,None],x,anchor,lag)))
    return [*components,np.column_stack((components[0],components[1]))]


class ReferenceModels:
    def fit(self,rows,labels,starts):
        self.trained=[OnlineVoltage().fit(x,labels-starts) for x in rows]
        self.scale=StandardScaler().fit(rows[-1])
        z=self.scale.transform(rows[-1])
        self.linear=LogisticRegression(C=.18,class_weight="balanced",max_iter=500).fit(z,labels>=1)
        self.tree=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=9,l2_regularization=3,
                                                  min_samples_leaf=20,random_state=73).fit(z,labels>=1)
        return self

    def predict(self,x,anchor,straight):
        analog=[anchor+model.predict(v) for model,v in zip(self.trained,x,strict=True)]
        bank=np.column_stack(analog)
        z=self.scale.transform(x[-1])
        # 分类器转换为以1为判定线的无量纲读数，供固定/现场端以相同方式合用。
        binary=np.column_stack((2*self.linear.predict_proba(z)[:,1],2*self.tree.predict_proba(z)[:,1]))
        out=dict(zip(READERS,bank.T,strict=True))
        out["以已回参考训练线性分类"]=binary[:,0]
        out["以已回参考训练树分类"]=binary[:,1]
        out["保持上次参考"]=np.full(len(z),anchor)
        out["原论文近期读法"]=straight
        return out

    def update(self,rows,target,anchor):
        for model,x in zip(self.trained,rows,strict=True):model.update(x,target-anchor)


class ReferenceTrust:
    def fit(self,ctx,bank,y):
        self.scaler=StandardScaler().fit(ctx)
        self.ctx=self.scaler.transform(ctx)
        self.src_loss=self.loss(bank,y)
        self.local=[];self.error=[]
        return self

    @staticmethod
    def loss(bank,y):
        labels=y[:,None]>=1
        return ((bank>=1)!=labels).astype(float)+.15*(expit(2*(bank-1))-labels)**2

    def predict(self,ctx,bank):
        z=self.scaler.transform(ctx)
        def risk(distance,loss,count):
            n=min(len(loss),count)
            ix=np.argpartition(distance,n-1,axis=1)[:,:n]
            w=np.exp(-np.take_along_axis(distance,ix,axis=1))
            return w,loss[ix]
        d=np.mean((z[:,None]-self.ctx[None])**2,axis=2)
        w,cost=risk(d,self.src_loss,40)
        errors=((cost*w[:,:,None]).sum(1)+2*self.src_loss.mean(0))/(w.sum(1)[:,None]+2)
        if self.local:
            d=np.mean((z[:,None]-np.asarray(self.local)[None])**2,axis=2)
            w,observed=risk(d,np.array(self.error),20)
            errors=((w[:,:,None]*observed).sum(1)+4*errors)/(w.sum(1)[:,None]+4)
        weights=np.exp(-18*(errors-errors.min(1)[:,None]))
        # 各回归器有不同的电压幅度，以平滑的分类票限制大幅度外推的影响。
        return 2*(expit(3*(bank-1))*weights).sum(1)/weights.sum(1)

    def update(self,ctx,bank,y):
        self.local.extend(self.scaler.transform(ctx));self.error.extend(self.loss(bank,y))


def context(views,c,idx,prev,anchor):
    delta=np.abs(views[3][idx]-views[3][prev])
    return np.column_stack((c[idx],np.log1p(delta),np.full(len(idx),np.clip(anchor,-1,5)),(idx-prev)*.5))
