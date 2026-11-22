"""估计已知浓度、温湿与共址NO读数应产生的两电极响应，再以新观测反推浓度。"""
import numpy as np
from harness.learning.models.air_quality.aircubes_stack import StackMaintenance

INCUMBENT="现场统一重估六工具配比"
PRIMARY="按两电极实际响应可信度反推并维护浓度"
CHANNELS=("第一电极反推","第二电极反推","两电极联合反推","随近期变化联合反推")


def environmental(z):
    temp,rh,drh,n1,n2,power=z[:,2:].T
    return np.column_stack((np.ones(len(z)),temp,rh,drh,n1,n2,power,temp**2,rh**2,n1*temp,n2*temp,temp*rh))


def sensitivity(z):
    return np.column_stack((np.ones(len(z)),z[:,2:4],z[:,4:7]))


class ResponseRelease:
    def fit(self,t,raw,y,scaler):
        self.scale=scaler
        z=np.clip(scaler.transform(raw),-7,7)
        self.center=np.median(y);self.unit=max(1.,np.std(y))
        c=(y-self.center)/self.unit
        a=np.column_stack((environmental(z),sensitivity(z)*c[:,None],c**2,c**3/5))
        self.gram=a.T@a/len(a)+.001*np.eye(a.shape[1])
        self.coef=np.linalg.solve(self.gram,a.T@z[:,:2]/len(a))
        residual=z[:,:2]-a@self.coef
        self.cov=residual.T@residual/len(a)+.0001*np.eye(2)
        self.grid=np.arange(0,max(100.,np.ceil(y.max()*1.5)),.5)
        self.level=(self.grid-self.center)/self.unit
        valid=np.diff(t)==600
        self.change=max(1.,np.median(abs(np.diff(y)[valid]))*1.4826)
        return self


class ResponseMaintenance:
    def __init__(self,tools):
        self.old=StackMaintenance(tools);self.q=tools["response"]
        self.gram=np.zeros_like(self.q.gram);self.rhs=np.zeros_like(self.q.coef)
        self.cov=self.q.cov.copy()
        self.previous=None;self.last_ref=None;self.loss=np.ones(5)*6.

    def predict(self,t,x,base):
        old=self.old.predict(t,x,base);p=old[INCUMBENT];q=self.q
        z=q.scale.transform(x).clip(-7,7);a=environmental(z);b=sensitivity(z)
        coef=q.coef+np.linalg.solve(self.gram+25*q.gram,self.rhs)
        v0=a@coef[:a.shape[1]];v1=b@coef[a.shape[1]:-2]
        expected=v0[:,None,:]+v1[:,None,:]*q.level[None,:,None]+(q.level[:,None]**2)*coef[-2]+(q.level[:,None]**3/5)*coef[-1]
        residual=z[:,None,:2]-expected
        precision=np.linalg.inv(.5*q.cov+.5*self.cov+.00001*np.eye(2))
        joint=np.einsum("nti,ij,ntj->nt",residual,precision,residual)
        distances=[residual[:,:,i]**2/max(self.cov[i,i],.001) for i in (0,1)]+[joint]
        forecasts=[]
        def mean(dist):
            wt=np.exp(-.5*(dist-dist.min(axis=-1,keepdims=True)).clip(0,100))
            return wt@q.grid/wt.sum(axis=-1)
        for dist in distances:forecasts.append(mean(dist))
        dynamic=[]
        for i,stamp in enumerate(t):
            dist=joint[i].copy()
            if self.previous is not None:
                prior_time,prior_value,prior_p=self.previous
                gap=(stamp-prior_time)/600
                ahead=prior_value+.8*(p[i]-prior_p)
                if gap<=6:
                    dist+=((q.grid-ahead)/(q.change*2*np.sqrt(gap)))**2
            value=float(mean(dist));dynamic.append(value)
            self.previous=stamp,value,p[i]
        forecasts.append(np.array(dynamic))
        bank=np.column_stack(forecasts)
        cols={};carry=np.zeros_like(bank)
        if self.last_ref is not None:
            stamp,error=self.last_ref
            carry=.5*np.exp(-(t-stamp)[:,None]/21600)*np.clip(error,-15,15)
        bank=np.maximum(0,bank+carry)
        for j,name in enumerate(CHANNELS):
            cols[name]=bank[:,j]
            for part in (.25,.5,.75):
                cols[f"论文联合与{name}占{part:g}"]=p+part*(bank[:,j]-p)
        opts=np.column_stack((p,bank))
        weights=1/(1+self.loss)**2
        cols[PRIMARY]=.5*p+.5*(opts@weights/weights.sum())
        cols["冻结响应初值与原强工具"]=.5*(p+opts.mean(1))
        self.pending=(t[-1],z[-1,:2],a[-1],b[-1],np.array([v[-1] for v in forecasts]),opts[-1])
        return {**old,**cols}

    def update(self,y):
        self.old.update(y);t,measured,a,b,p,opts=self.pending;q=self.q
        c=(y-q.center)/q.unit
        design=np.r_[a,b*c,c*c,c**3/5]
        residual=measured-design@q.coef
        before=residual-design@np.linalg.solve(self.gram+25*q.gram,self.rhs)
        self.gram=.93*self.gram+np.outer(design,design)
        self.rhs=.93*self.rhs+np.outer(design,residual)
        self.cov=.95*self.cov+.05*np.outer(before,before)
        self.last_ref=t,y-p
        self.loss=.95*self.loss+.05*np.minimum((y-opts)**2,400)
