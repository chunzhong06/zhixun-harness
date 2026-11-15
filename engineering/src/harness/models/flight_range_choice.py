"""源端轮流留出机队估计各段寿命误差，在新航班仅用实际测量决定信任哪段。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor

from harness.methods.flight_disagreement import fit_weights


def range_views(prep,rows):
    scaler,healthy,_,_ = prep
    residual=rows[:,:14]-healthy.predict(scaler.transform(rows[:,14:]))
    parts=np.array_split(np.argsort(rows[:,16],kind="stable"),min(3,len(rows)))
    features=[]
    settings=[]
    for i in range(3):
        ix=parts[min(i,len(parts)-1)]
        r=residual[ix]
        q=np.percentile(r,[25,50,75],axis=0)
        features.append(np.r_[q[1],q[2]-q[0],np.abs(r).mean(0)])
        settings.extend((np.mean(rows[ix,16]), np.std(rows[ix,16])))
    signed=residual.mean(0)
    fast=np.abs(np.diff(residual,axis=0)).mean(0) if len(rows)>1 else np.zeros(14)
    context=np.r_[rows[:,14:].mean(0),settings]
    return (*features,np.concatenate(features),np.r_[signed,residual.std(0),fast],context)


NAMES=("低油门森林","中油门森林","高油门森林","三段联合森林","偏差波动森林")
CHOICES=("估错最小","估错软组合","公共与估错各半")


def train_heads(x,y):
    cfg=dict(n_estimators=100,max_depth=6,random_state=17,n_jobs=1)
    return [RandomForestRegressor(**cfg).fit(a,y) for a in x[:len(NAMES)]]


def gate_inputs(pred,context):
    return np.column_stack((context,pred,pred-pred.mean(axis=1,keepdims=True)))


class FlightRangeChoice:
    def __init__(self,prep,training,target,held_pred,held_context,sites,units):
        self.prep=prep
        self.heads=train_heads(training,target)
        self.shared=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=17,n_jobs=1).fit(
            np.vstack(training[:3]),np.tile(target,3))
        weights=np.zeros(len(target))
        for site in np.unique(sites):
            local=sites==site
            ids=np.unique(units[local])
            for uid in ids:
                mask=local&(units==uid)
                weights[mask]=1/(mask.sum()*len(ids))
        self.fixed=fit_weights(held_pred,target,weights)
        self.error=ExtraTreesRegressor(n_estimators=100,max_depth=4,min_samples_leaf=12,random_state=17,n_jobs=1)
        self.error.fit(gate_inputs(held_pred,held_context),np.abs(held_pred-target[:,None]),sample_weight=weights)
        self.rule=None

    def predictions(self,x):
        pred=np.column_stack([model.predict(values) for model,values in zip(self.heads,x[:5],strict=True)])
        err=np.maximum(1.,self.error.predict(gate_inputs(pred,x[5])))
        weights=1/err**2
        soft=(weights*pred).sum(1)/weights.sum(1)
        fixed=pred@self.fixed
        out=dict(zip(NAMES,pred.T,strict=True))
        shared=np.column_stack([self.shared.predict(segment) for segment in x[:3]])
        out.update({"五头等权":pred.mean(1),"三段等权":pred[:,:3].mean(1),"公共固定权重":fixed,
                    "估错最小":pred[np.arange(len(pred)),err.argmin(1)],"估错软组合":soft,
                    "公共与估错各半":(soft+fixed)/2,"共享森林等权":shared.mean(1)})
        for i,col in enumerate(shared.T):
            out[f"共享森林第{i+1}段"]=col
        return out

    def forecast(self,rows):
        if self.rule not in CHOICES:
            raise ValueError("该机队模型没有封存现场决定")
        x=[a[None,:] for a in range_views(self.prep,rows)]
        return float(self.predictions(x)[self.rule][0])
