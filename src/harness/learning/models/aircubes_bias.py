"""利用历史真实参考估计电极增益和偏移的不同变化速度。新参考到达后才更新后验。"""
import numpy as np
from harness.learning.models.aircubes_joint import JointMaintenance

PRIMARY="按本机连续校准误差识别漂移速度"
LABELS=("慢偏移","快偏移","灵敏度漂移","偏移与灵敏度均缓慢","偏移与灵敏度均快速")


class DriftFilter:
    def __init__(self,dimension):
        d=dimension
        # 偏移的量纲是ppb，其他维度使用已标准化的观测。
        self.noise=np.stack([np.r_[offset,np.full(d-1,slope)] for offset,slope in
                             ((.06,.002),(1.,.002),(.06,.08),(.2,.012),(.8,.08))])
        self.cov=np.broadcast_to(np.diag(np.r_[2.,np.full(d-1,.45)]),(5,d,d)).copy()
        self.beta=np.zeros((5,d))
        self.last=None

    def predict(self,measurement,base):
        return np.maximum(0,base[:,None]+measurement@self.beta.T)

    def update(self,t,x,residual,sigma2):
        elapsed=1 if self.last is None else max(.01,(t-self.last)/21600)
        self.last=t
        idx=np.arange(len(x))
        self.cov[:,idx,idx]+=self.noise*min(elapsed,100)
        projection=self.cov@x
        variance=projection@x+sigma2
        err=residual-self.beta@x
        self.beta+=(err/variance)[:,None]*projection
        self.cov-=projection[:,:,None]*projection[:,None,:]/variance[:,None,None]
        return -.5*(np.log(variance)+np.minimum(err**2/variance,25))


class DriftMaintenance:
    def __init__(self,tools):
        self.old=JointMaintenance(tools)
        self.joint=tools["joint"]
        self.filters=[DriftFilter(6),DriftFilter(6)]
        self.mass=np.ones((2,5))/10
        self.noise=5.

    def predict(self,t,x,paper):
        old=self.old.predict(t,x,paper)
        z=np.clip(self.joint.scale.transform(x),-6,6)
        # 不做新一轮现场回归的两个源起点，避免把同一参考余差在两套维护里重复相加。
        source=[]
        from harness.learning.models.aircubes_joint import JointProbes
        static=JointProbes().design(t,z)
        for j in (0,2):
            source.append(static[j]@self.joint.models[j][1])
        readings=[];out={};self.pending=[]
        for j,(model,p) in enumerate(zip(self.filters,source,strict=True)):
            f=np.column_stack((np.ones(len(t)),(p-18)/12,z[:,2],z[:,4],z[:,0]-z[:,1],
                               (z[:,5]+z[:,6])/2 if j else z[:,3]))
            vals=model.predict(f,p);readings.append(vals)
            name=("原双电极","含辅助读数")[j]
            for i,label in enumerate(LABELS):out[name+"固定"+label]=vals[:,i]
            self.pending.append((f[-1],p[-1]))
        bank=np.stack(readings,axis=1)
        for i,label in enumerate(LABELS):
            out["两种源曲线固定各半/"+label]=bank[:,:,i].mean(1)
        direct=(bank*self.mass).sum((1,2))
        ref=old["本机混合上次误差6小时乘0.5"]
        out["后验漂移模型"]=direct
        out[PRIMARY]=(direct*3+ref)/4
        out["冻结漂移模型初始配比与论文"]=(bank.mean((1,2))*3+ref)/4
        for k in list(out):
            if k not in (PRIMARY,"后验漂移模型","冻结漂移模型初始配比与论文"):
                out["论文和漂移固定各半/"+k]=(ref+out[k])/2
                out["论文和漂移固定一比三/"+k]=(ref+3*out[k])/4
        self.last=t[-1]
        return {**old,**out}

    def update(self,y):
        self.old.update(y)
        likelihood=[]
        disagreement=[]
        for model,(f,p) in zip(self.filters,self.pending,strict=True):
            disagreement.extend(y-(p+model.beta@f))
            likelihood.append(model.update(self.last,f,y-p,max(self.noise,2.)))
        prior=.97*self.mass+.03/self.mass.size
        log=np.array(likelihood)
        w=prior*np.exp(log-log.max())
        self.mass=w/w.sum()
        self.noise=.95*self.noise+.05*min(64,float(np.median(np.square(disagreement))))
