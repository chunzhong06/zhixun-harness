"""用同一时点多个电极的测量关系观察偏移；有限真实参考用于区分偏移与灵敏度变化。"""
import numpy as np
from harness.learning.models.aircubes_bias import DriftMaintenance
from harness.learning.models.aircubes_joint import JointProbes

PRIMARY="按回标检验无标签电极关系后决定是否校正"
BASELINE="论文强校正与合用当前及上一窗变化重训0.82各半"


class AgreementRelease:
    def fit(self,t,x,y,joint):
        self.scale=joint.scale
        z=np.clip(self.scale.transform(x),-7,7)
        context=self.covariates(z)
        # 拟合两路仪器对同一浓度的正常关系。训练目标不用当前环境的参考浓度。
        self.mapping=[]
        for j in (0,1):
            a=np.column_stack((context,z[:,1-j],z[:,1-j]*z[:,2]))
            w=np.linalg.solve(a.T@a+3*np.eye(a.shape[1]),a.T@z[:,j])
            self.mapping.append(w)
        return self

    @staticmethod
    def covariates(z):
        temp=z[:,2];rh=z[:,3]
        return np.column_stack((np.ones(len(temp)),temp,rh,z[:,4],temp*rh,temp**2,z[:,5:8]))


class AgreementMaintenance:
    def __init__(self,tools):
        self.old=DriftMaintenance(tools)
        self.joint=tools["joint"]
        self.agree=tools["agreement"]
        self.offset=np.zeros((2,2));self.last=None
        self.stat=np.zeros((2,2));self.rhs=np.zeros(2)
        self.squared=np.ones(5)*6.

    def predict(self,t,x,bank):
        old=self.old.predict(t,x,bank)
        incumbent=old[BASELINE]
        z=np.clip(self.agree.scale.transform(x),-7,7)
        c=self.agree.covariates(z)
        err=np.column_stack([z[:,j]-np.column_stack((c,z[:,1-j],z[:,1-j]*z[:,2]))@self.agree.mapping[j] for j in (0,1)])
        smoothed=[]
        for stamp,sample in zip(t,err,strict=True):
            hours=1/6 if self.last is None else min(100,(stamp-self.last)/3600)
            if hours<=0:raise ValueError("电极偏移监测时间回退")
            w=-np.expm1(-hours/np.array([2.,24.]))
            self.offset+=w[:,None]*(np.clip(sample,-3,3)-self.offset)
            smoothed.append(self.offset.copy());self.last=stamp
        smooth=np.stack(smoothed)
        # 由源双电极曲线估计本次电压变化换算成多少ppb，另外的原始通道保持同步实测。
        joint_coef=self.joint.models[0][1]
        orig=JointProbes().design(t,z)[0]@joint_coef
        values=[]
        for h in range(2):
            for j in (0,1):
                adjusted=z.copy();adjusted[:,j]-=smooth[:,h,j]
                decoded=JointProbes().design(t,adjusted)[0]@joint_coef
                values.append(np.clip(decoded-orig,-15,15))
        delta=np.column_stack(values)
        out={};bank2=[incumbent]
        for j,v in enumerate(delta.T):
            label=("两小时","一日")[j//2]+("第一电极","第二电极")[j%2]
            for k in (-.25,.25,.5,1.):
                out["固定无标签偏移/"+label+f"/{k:g}"]=np.maximum(0,incumbent+k*v)
            bank2.append(out["固定无标签偏移/"+label+"/0.5"])
        options=np.column_stack(bank2)
        w=(1+self.squared)**-2
        out["按已回平均误差合用偏移"]=options@w/w.sum()
        context=delta[:,2:]/5
        coeff=np.linalg.solve(self.stat+10*np.eye(2),self.rhs)
        adjusted=np.clip(context@coeff,-10,10)
        out["按已回参考拟合无标签偏移"]=np.maximum(0,incumbent+adjusted)
        out[PRIMARY]=(.5*out["按已回参考拟合无标签偏移"]+.5*out["按已回平均误差合用偏移"])
        out["固定无标签初始配比"]=(incumbent+options.mean(1))/2
        self.pending=(context[-1],options[-1],incumbent[-1])
        return {**old,**out}

    def update(self,y):
        c,options,p=self.pending
        self.old.update(y)
        self.squared=.93*self.squared+.07*np.minimum((y-options)**2,625)
        self.stat=.96*self.stat+np.outer(c,c)
        self.rhs=.96*self.rhs+c*(y-p)
