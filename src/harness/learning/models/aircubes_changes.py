"""分别训练参考浓度的绝对值与变化量，预测从最后一次实际校准时刻起累积的变化。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from harness.learning.models.aircubes_stack import StackMaintenance

PRIMARY="按已回参考对各段浓度变化的误差更新斜率"
INCUMBENT="现场统一重估六工具配比"
MODES=("单步浓度变化","相隔一小时","相隔六小时","三种间隔共同训练")


def design(z):
    a,b,temp,rh,drh,n1,n2,power=z.T
    return np.column_stack((a,b,temp,rh,drh,n1,n2,power,a*temp,b*temp,a*rh,b*rh,
        a*n1,b*n2,n1*temp,n2*temp,a**2,b**2,temp**2,rh**2,a**3/3,b**3/3))


class ChangeRelease:
    def fit(self,t,x,y):
        self.scale=StandardScaler().fit(x)
        f=design(np.clip(self.scale.transform(x),-6,6))
        self.coefficients=[];self.matrices=[]
        grams=[];rhs=[]
        for lag in (600,3600,21600):
            ix=np.searchsorted(t,t-lag)
            use=np.flatnonzero(t-t[ix]==lag)
            delta=f[use]-f[ix[use]]
            dy=y[use]-y[ix[use]]
            g=delta.T@delta/len(dy)+.005*np.eye(f.shape[1])
            r=delta.T@dy/len(dy)
            grams.append(g);rhs.append(r)
        self.matrices=np.stack([*grams,np.mean(grams,axis=0)])
        self.coefficients=np.linalg.solve(self.matrices,np.stack([*rhs,np.mean(rhs,axis=0)])[...,None])[...,0]
        return self


class ChangeMaintenance:
    def __init__(self,tools,baseline=None):
        self.old=StackMaintenance(tools) if baseline is None else baseline
        self.release=tools["changes"]
        dimension=self.release.matrices.shape[-1]
        self.gram=np.zeros((dimension,dimension))
        self.rhs=np.zeros((4,dimension))
        self.reference=None;self.loss=np.full(5,6.)
        self.calibration=np.zeros((2,2));self.target=np.zeros(2)

    def predict(self,t,x,base):
        out=self.old.predict(t,x,base);anchor=out[INCUMBENT]
        q=self.release
        f=design(q.scale.transform(x).clip(-6,6))
        corrected=q.coefficients+np.linalg.solve(24*q.matrices+self.gram,self.rhs[...,None])[...,0]
        if self.reference is None:
            bank=np.broadcast_to(anchor[:,None],(len(t),4)).copy()
            source=bank.copy();h=np.zeros(len(t))
        else:
            last,ref_time,truth=self.reference
            dx=f-last;h=(t-ref_time)/3600
            bank=np.maximum(0,truth+dx@corrected.T);source=np.maximum(0,truth+dx@q.coefficients.T)
        for j,name in enumerate(MODES):
            out[name+"固定源斜率"]=source[:,j]
            out[name+"现场重训"]=bank[:,j]
            for w in (.25,.5,.75):
                out[f"论文组合与{name}占{w:g}"]=anchor+w*(bank[:,j]-anchor)
        d=bank[:,-1]-anchor;fade=np.exp(-h/4)
        out["从参考恢复的变化量随时间减弱"]=anchor+.7*fade*d
        context=np.column_stack((d,d*fade))
        coef=np.linalg.solve(self.calibration+25*np.eye(2),self.target+np.array([5.,10.])).clip(-.4,1.)
        learned=anchor+(context@coef).clip(-12,12)
        options=np.column_stack((anchor,bank))
        w=1/(.5+self.loss)**2
        out[PRIMARY]=np.maximum(0,(learned+options@w/w.sum())/2)
        out["冻结浓度差选择初值"]=np.maximum(0,(anchor+context@np.array([.2,.4])+options.mean(1))/2)
        self.pending=(t[-1],f[-1].copy(),context[-1],options[-1])
        return out

    def update(self,y):
        self.old.update(y);t,f,c,options=self.pending
        if self.reference is not None:
            last,rt,truth=self.reference
            # 缺少参考时保留已发预测，但不把跨多日的差误当作单槽训练对。
            if t-rt<=12*3600:
                dx=f-last;dy=y-truth
                self.gram=.94*self.gram+np.outer(dx,dx)
                self.rhs=.94*self.rhs+(dy-self.release.coefficients@dx)[:,None]*dx
        self.reference=f,t,y
        self.loss=.94*self.loss+.06*np.minimum((y-options)**2,625)
        self.calibration=.94*self.calibration+np.outer(c,c)
        self.target=.94*self.target+c*(y-options[0])
