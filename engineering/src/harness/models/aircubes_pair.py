"""监测两个电极的实际分歧，以已交接的参考值校准四个论文估计的组合。"""
import numpy as np
from harness.models.aircubes import BASE,StationMaintenance

PRIMARY="据本台已返误差和两电极分歧校准浓度"
CONSTANT="用已返参考拟合全局模型分量"
ANCHOR=BASE[-1]+"/近期温度校准"


class PairedMaintenance:
    def __init__(self):
        self.old=StationMaintenance()
        self.params=(.85,.95,1.)
        self.mass=np.zeros((len(self.params),6,6))
        self.target=np.zeros((len(self.params),6))
        self.risk=np.full(4,16.)

    def predict(self,base,temp):
        old,features,bank=self.old.predict(base,temp)
        anchor=old[ANCHOR]
        diff=(base[:,:4]-base[:,4:5])/5
        # 只通过模型预测之间的差异以及现场可测温度解释需要额外修正的部分。
        design=np.column_stack((np.ones(len(base)),diff,(temp-15)/12))
        coef=np.linalg.solve(self.mass+np.diag([10.,10.,10.,10.,10.,16.]),self.target[...,None])[...,0]
        options=np.maximum(0.,anchor[:,None]+design@coef.T)
        names=[f"固定现场配对记忆{r:g}" for r in self.params]
        new=dict(zip(names,options.T,strict=True))
        complete=np.column_stack((anchor,options))
        weight=(1+self.risk)**-2
        new[PRIMARY]=complete@weight/weight.sum()
        new["现场误差组合初值冻结"]=complete.mean(1)
        new[CONSTANT]=options[:,-1]
        for i,n in enumerate(names):
            new["论文原校正与"+n+"各半"]=(options[:,i]+anchor)/2
        self.pending=(features[-1],bank[-1],design[-1],anchor[-1],complete[-1])
        return {**old,**new}

    def update(self,y):
        features,bank,z,anchor,options=self.pending
        self.old.feedback(features,bank,y)
        self.mass=np.array(self.params)[:,None,None]*self.mass+np.outer(z,z)
        self.target=np.array(self.params)[:,None]*self.target+z*(y-anchor)
        self.risk=.92*self.risk+.08*np.minimum((y-options)**2,900)
