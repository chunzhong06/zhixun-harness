"""从源期逐段留出试验学互补误差，在相同专家库上比较全局组合和当前读数条件组合。"""
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from harness.learning.models.air_quality.aircubes import AirCalibration
from harness.learning.models.air_quality.aircubes_joint import JointRelease
from harness.learning.models.air_quality.aircubes_retrain import ElectrodeRelease
from harness.learning.models.air_quality.aircubes_innovation import TransferRelease
from harness.learning.models.air_quality.aircubes_bias import DriftMaintenance

EXPERTS=("论文强校正与合用当前及上一窗变化重训0.82各半","按本机连续校准误差识别漂移速度",
         "论文强校正与加入同机NO读数重训0.82各半","本机混合上次误差6小时乘0.5",
         "双电极原始重训记忆0.88","论文和漂移固定一比三/两种源曲线固定各半/偏移与灵敏度均快速")
PRIMARY="根据当前模型分歧和装机前向试验学习互补误差"


def context(x,bank):
    return np.column_stack((x[:,2],x[:,3],x[:,4],x[:,7],bank.mean(1),bank.std(1),
                            bank[:,0]-bank[:,1],bank[:,2]-bank[:,3]))


def moments(design,target):
    return design.T@design/len(target),design.T@target/len(target)


def simplex(gram,rhs):
    n=len(rhs);one=np.ones(n)/n
    res=minimize(lambda w:float(w@gram@w-2*w@rhs),one,jac=lambda w:2*(gram@w-rhs),
        method="SLSQP",bounds=[(0.,1.)]*n,constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
        options={"maxiter":100,"ftol":1e-8})
    return res.x if res.success else one


class StackRelease:
    def fit(self,time,x,y):
        start=time[0];stop=time[-1]
        breaks=np.linspace(start,stop+1,5)
        banks=[];contexts=[];targets=[]
        for j in (1,2,3):
            cutoff=breaks[j];train=time<cutoff;test=(time>=cutoff)&(time<breaks[j+1])
            if not test.any():continue
            tick=time[train];raw=x[train];truth=y[train]
            paper=AirCalibration().fit(raw[:,:5],truth)
            electrode=ElectrodeRelease().fit(raw[:,:5],truth)
            tools={"paper":paper,"electrode":electrode,"joint":JointRelease().fit(tick,raw,truth),
                   "transfer":TransferRelease().fit(tick,raw[:,:5],truth,electrode)}
            controller=DriftMaintenance(tools)
            clock=time[test];rows=x[test];labels=y[test]
            pp=paper.predict(rows[:,:5])
            bounds=np.r_[0,np.flatnonzero(np.diff(clock//21600))+1,len(clock)]
            for a,b in zip(bounds[:-1],bounds[1:],strict=True):
                published=controller.predict(clock[a:b],rows[a:b],pp[a:b])
                bank=np.column_stack([published[n] for n in EXPERTS])
                banks.append(bank);contexts.append(context(rows[a:b],bank));targets.extend(labels[a:b])
                if (clock[b-1]+600)%21600==0 and b<len(clock):
                    controller.update(labels[b-1])
        b=np.vstack(banks);c=np.vstack(contexts);truth=np.array(targets)
        self.scale=StandardScaler().fit(c)
        # 优化等价于绝对浓度的平方误差，先扣同一个共同预测改善数值条件。
        d=b-b[:,:1];target=truth-b[:,0]
        self.g,self.r=moments(d,target)
        self.weights=simplex(self.g+.02*np.eye(len(EXPERTS)),self.r)
        z=self.scale.transform(c).clip(-3,3)
        f=self.interaction(z,d)
        self.cond_g,self.cond_r=moments(f,truth-b@self.weights)
        return self

    @staticmethod
    def interaction(z,difference):
        return (difference[:,1:,None]*np.column_stack((np.ones(len(z)),z))[:,None,:]).reshape(len(z),-1)


class StackMaintenance:
    def __init__(self,tools):
        self.old=DriftMaintenance(tools);self.install=tools["stack"]
        self.g=np.zeros_like(self.install.g);self.r=np.zeros_like(self.install.r)
        self.cg=np.zeros_like(self.install.cond_g);self.cr=np.zeros_like(self.install.cond_r)

    def predict(self,t,x,base):
        out=self.old.predict(t,x,base);q=self.install
        bank=np.column_stack([out[n] for n in EXPERTS]);anchor=bank@q.weights
        d=bank-bank[:,:1];z=q.scale.transform(context(x,bank)).clip(-3,3)
        f=q.interaction(z,d)
        w=simplex(18*q.g+self.g+.5*np.eye(len(EXPERTS)),18*q.r+self.r)
        pred=bank@w
        out.update({"装机试验固定六工具权重":anchor,"六工具固定平均":bank.mean(1),"现场统一重估六工具配比":pred})
        for reg in (.15,1.,6.):
            prior=np.linalg.solve(q.cond_g+reg*np.eye(len(q.cond_r)),q.cond_r)
            theta=np.linalg.solve(24*q.cond_g+self.cg+24*reg*np.eye(len(prior)),24*q.cond_r+self.cr)
            out[f"源端条件组合/{reg:g}"]=np.maximum(0,anchor+(f@prior).clip(-6,6))
            out[f"现场重训条件组合/{reg:g}"]=np.maximum(0,anchor+(f@theta).clip(-6,6))
        out[PRIMARY]=(out["现场重训条件组合/1"]+pred)/2
        out["源端条件与全局固定各半"]=(out["源端条件组合/1"]+anchor)/2
        self.pending=d[-1],bank[-1,0],f[-1],anchor[-1]
        return out

    def update(self,y):
        self.old.update(y)
        d,first,f,anchor=self.pending
        self.g=.96*self.g+np.outer(d,d);self.r=.96*self.r+d*(y-first)
        self.cg=.96*self.cg+np.outer(f,f);self.cr=.96*self.cr+f*(y-anchor)
