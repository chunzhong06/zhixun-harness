"""在同一仪器上从已发布的NO和NO₂电极读数训练交叉气体读法，全部对手同样可用。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from harness.learning.models.aircubes_innovation import IntervalMaintenance

AUX=("NO_00","NO_01","V_00")
PRIMARY="用本机实际回标检验跨气体校准能否接班"
INCUMBENT="本机混合上次误差6小时乘0.5"
VIEWS=("仅原两电极","加入同机NO读数","加入同机NO和供电","合用当前及上一窗变化")
DECAYS=(.82,.94,1.)


class JointProbes:
    def __init__(self):
        self.previous=None
        self.stamp=None

    def design(self,t,z):
        prev=z[0] if self.previous is None else self.previous
        diff=np.diff(z,axis=0,prepend=prev[None,:])
        times=np.r_[t[0]-600 if self.stamp is None else self.stamp,t]
        diff[np.diff(times)!=600]=0
        self.previous=z[-1].copy();self.stamp=t[-1]
        a,b,temp,rh,drh,n1,n2,supply=z.T
        core=[np.ones(len(z)),a,b,temp,rh,drh,a*temp,b*temp,a*rh,b*rh,temp*drh,temp**2,rh**2,a**2,b**2]
        extras=[n1,n2,n1*temp,n2*temp,n1*a,n2*b]
        volt=[supply,supply*temp,supply*(a+b)/2]
        return [np.column_stack(terms) for terms in (core,core+extras,core+extras+volt,
             core+extras+volt+[diff[:,0],diff[:,1],diff[:,2],diff[:,3],diff[:,5],diff[:,6]])]


class JointRelease:
    def fit(self,t,raw,y):
        self.scale=StandardScaler().fit(raw)
        x=JointProbes().design(t,np.clip(self.scale.transform(raw),-7,7))
        self.models=[]
        for f in x:
            g=f.T@f/len(f)+.004*np.eye(f.shape[1])
            coef=np.linalg.solve(g,f.T@y/len(f))
            self.models.append((g,coef))
        return self


class JointMaintenance:
    def __init__(self,tools):
        self.old=IntervalMaintenance(tools)
        self.release=tools["joint"]
        self.grams=[np.zeros((3,*g.shape)) for g,_ in self.release.models]
        self.rhs=[np.zeros((3,len(c))) for _,c in self.release.models]
        self.sensor=JointProbes()
        self.recent=[];self.running=np.ones(13)*7;self.last=None

    def predict(self,t,raw,base):
        previous=self.old.predict(t,raw[:,:5],base)
        p=previous[INCUMBENT]
        z=np.clip(self.release.scale.transform(raw),-7,7)
        views=self.sensor.design(t,z)
        bank=[p];out={}; self.pending=[]
        if self.last is None:
            carry=np.zeros((len(t),13))
        else:
            time,err=self.last
            carry=.5*np.exp(-(t-time)[:,None]/21600)*np.clip(err,-12,12)
        for j,(f,(g,start)) in enumerate(zip(views,self.release.models,strict=True)):
            coef=np.linalg.solve(self.grams[j]+12*g,self.rhs[j][...,None])[...,0]
            pred=np.maximum(0,f@(start+coef).T)
            self.pending.append((f[-1],f[-1]@start))
            for i,decay in enumerate(DECAYS):
                name=VIEWS[j]+f"重训{decay:g}"
                index=1+3*j+i
                val=np.maximum(0,pred[:,i]+carry[:,index])
                bank.append(val);out[name]=val
                out[name+"保持本次校准"]=pred[:,i]
                out["论文强校正与"+name+"各半"]=(val+p)/2
        bank=np.column_stack(bank)
        self.uncorrected=bank[-1].copy()
        for j,f in enumerate(views):
            for i,d in enumerate(DECAYS):self.uncorrected[1+3*j+i]=out[VIEWS[j]+f"重训{d:g}保持本次校准"][-1]
        context=np.column_stack((z[:,0]-z[:,1],z[:,2],z[:,5:7].mean(1),z[:,7],p/20)).clip(-6,6)
        loss=self.running
        if self.recent:
            loc,mse=zip(*self.recent,strict=True)
            distance=np.mean((context[:,None,:]-np.array(loc)[None,:,:])**2,axis=2)
            w=np.exp(-2*distance)*(.97**np.arange(len(mse)-1,-1,-1))
            loss=(w@np.array(mse)+5*loss)/(w.sum(1)[:,None]+5)
        weights=(1+loss)**-2
        out[PRIMARY]=(bank*weights).sum(1)/weights.sum(-1)
        out["跨气体工具等权"]=bank.mean(1)
        self.ctx=context[-1].copy();self.bank=bank[-1].copy();self.t=t[-1]
        return {**previous,**out}

    def update(self,y):
        self.old.update(y)
        self.last=self.t,y-self.uncorrected
        loss=np.minimum((y-self.bank)**2,625)
        self.recent.append((self.ctx,loss));self.recent=self.recent[-100:]
        self.running=.96*self.running+.04*loss
        d=np.array(DECAYS)
        for j,(x,old) in enumerate(self.pending):
            self.grams[j]=d[:,None,None]*self.grams[j]+np.outer(x,x)
            self.rhs[j]=d[:,None]*self.rhs[j]+x*(y-old)
