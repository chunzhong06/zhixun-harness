"""NO₂ 共址标定与迟到的参考仪读数。当前设备从同一套源训练工具中选择校正。"""
from collections import deque

import numpy as np
from sklearn.ensemble import RandomForestRegressor

RAW=("NO2_00","NO2_01","T_01","RH_01")
BASE=("论文稳健回归探头一","论文稳健回归探头二","论文随机森林探头一","论文随机森林探头二","四模型固定平均")
MODES=("保持","长时偏差","近期偏差","校准斜率","近期温度校准")
PRIMARY="用实发回标选择双探头的校准模型"
FROZEN="保持装机时的均衡配比"


class HumidityHistory:
    def __init__(self):
        self.delta=deque(maxlen=51)
        self.last=np.nan
        self.weights=np.exp(-np.arange(51)/15)/600

    def next(self,humidity):
        change=humidity-self.last if np.isfinite(humidity) and np.isfinite(self.last) else 0.
        self.delta.appendleft(change)
        self.last=humidity
        return np.dot(self.weights[:len(self.delta)],self.delta)


def design(rows,j):
    voltage=rows[:,j];temperature=rows[:,2];drh=rows[:,4]
    return np.column_stack((np.ones(len(rows)),voltage,temperature,voltage*temperature,drh)),np.column_stack((voltage,temperature,drh))


def huber(x,y):
    # 与作者 RLM_22 一致的Huber阈值，用迭代加权最小二乘及MAD尺度。
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    for _ in range(1000):
        err=y-x@beta
        scale=max(np.median(abs(err))/.67448975,1e-8)
        sw=np.sqrt(np.minimum(1.,1.345*scale/np.maximum(abs(err),1e-10)))
        nxt=np.linalg.lstsq(x*sw[:,None],y*sw,rcond=None)[0]
        stop=np.linalg.norm(x@(nxt-beta))<=1e-4*max(np.linalg.norm(err),1.)
        beta=nxt
        if stop:break
    return beta


class AirCalibration:
    def fit(self,x,y):
        self.linear=[];self.forest=[]
        for j in (0,1):
            lin,tree=design(x,j)
            self.linear.append(huber(lin,y))
            # 与R的回归随机森林相同，三个字段每次随机取一个。叶大小在sklearn下为适配。
            model=RandomForestRegressor(n_estimators=1000,min_samples_leaf=100,max_features=1,n_jobs=1,random_state=72+j)
            self.forest.append(model.fit(tree,y))
        return self

    def predict(self,x):
        lin=[];tree=[]
        for j in (0,1):
            a,b=design(x,j)
            lin.append(a@self.linear[j]);tree.append(self.forest[j].predict(b))
        bank=np.column_stack((*lin,*tree))
        return np.column_stack((bank,bank.mean(1)))


class StationMaintenance:
    def __init__(self):
        self.precision=np.zeros((3,len(BASE),3,3))
        self.rhs=np.zeros((3,len(BASE),3))
        self.recent_sum=np.zeros(len(BASE))
        self.recent_count=0.
        self.cost=np.ones((len(BASE),len(MODES)))*16.
        self.count=0

    def coefficients(self):
        regularizer=np.diag([2.,12.,12.])
        return np.linalg.solve(self.precision+regularizer,self.rhs[...,None])[...,0]

    def predict(self,base,temp):
        design=np.stack((np.ones_like(base),(base-15)/15,np.broadcast_to((temp[:,None]-15)/12,base.shape)),axis=2)
        coef=self.coefficients()
        bank=np.stack((base,base+coef[0,:,0],base+self.recent_sum/(2+self.recent_count),
            base+np.sum(coef[2,:,:2]*design[:,:,:2],axis=2),
            base+np.sum(coef[1]*design,axis=2)),axis=2)
        # 实发历史上的误差决定哪些电极及工具可继续用于本台。各固定对手正常接收回标。
        w=(1.0+np.maximum(0,self.cost))**-2.0
        corrected=np.maximum(0,bank)
        out={f"{a}/{b}":corrected[:,i,j] for i,a in enumerate(BASE) for j,b in enumerate(MODES)}
        out[PRIMARY]=(corrected*w).sum((1,2))/w.sum()
        out[FROZEN]=corrected.mean((1,2))
        return out,design,corrected

    def feedback(self,features,bank,y):
        err=y-bank
        self.cost=.90*self.cost+.10*np.minimum(err**2,900)
        # 基准四类输出可能在零以下；原读法通过features中保存的预测恢复。
        raw=features[:,1]*15+15
        target=y-raw
        self.recent_sum=.85*self.recent_sum+target
        self.recent_count=.85*self.recent_count+1
        phi=np.stack((np.column_stack((np.ones(len(BASE)),np.zeros((len(BASE),2)))),
                      features,features*np.array([1,1,0.])))
        decay=np.array([1.,.85,.98])[:,None,None]
        self.precision=decay[:,:,:,None]*self.precision+phi[:,:,:,None]*phi[:,:,None,:]
        self.rhs=decay*self.rhs+phi*target[None,:,None]
        self.count+=1
