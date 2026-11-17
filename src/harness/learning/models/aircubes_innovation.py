"""从源期各区间校准余差学习跨时刻的持续性，回传后只修改之后的实际预测。"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from harness.learning.models.aircubes_retrain import ElectrodeMaintenance,PRIMARY as MIXED

BASELINE="双电极原始重训记忆0.88"
PRIMARY="从本机校准试验学习参考误差在下一段的延续"
TIMES=(1.,3.,6.,24.)


def describe_interval(current,previous,hours):
    diff=abs(current-previous)
    return np.column_stack((hours/6,np.exp(-hours/3),diff[:,:5],np.clip(np.abs(current[:,0]-current[:,1]),0,4),
        np.abs(current[:,2]),np.abs(current[:,4])))


class TransferRelease:
    def fit(self,time,rows,truth,raw_release):
        self.scale=raw_release.scaler
        count=len(time);tr=len(time)//2
        if count<500:raise ValueError("可用于源端跨时试验的时刻太少")
        # 先用前半源记录训练，再在后半源期产生未用本行真值拟合的误差样本。
        from harness.learning.models.aircubes_retrain import ElectrodeRelease,features
        small=ElectrodeRelease().fit(rows[:tr],truth[:tr])
        z=np.clip(small.scaler.transform(rows[tr:]),-7,7)
        f=features(z,2);initial=small.source[2][1];g=small.source[2][0]*12
        gram=np.zeros_like(g);rhs=np.zeros(len(g));previous=None
        data=[];targets=[]
        clock=time[tr:];target=truth[tr:];current=np.clip(self.scale.transform(rows[tr:]),-7,7)
        # 让源试验的返回时刻、频率和部署完全相同，缺测不补。
        edges=np.r_[0,np.flatnonzero(np.diff(clock//21600))+1,len(clock)]
        for a,b in zip(edges[:-1],edges[1:],strict=True):
            pred=np.maximum(0,f[a:b]@(initial+np.linalg.solve(g+gram,rhs)))
            error=target[a:b]-pred
            if previous is not None:
                stamp,ctx,prior_err=previous
                hours=(clock[a:b]-stamp)/3600
                keep=(hours>0)&(hours<=24)
                if keep.any():
                    h=describe_interval(current[a:b][keep],ctx,hours[keep])
                    data.extend(np.column_stack((np.repeat(prior_err,keep.sum()),h)))
                    targets.extend(error[keep])
            if (clock[b-1]+600)%21600==0:
                last=f[b-1];gram=.88*gram+np.outer(last,last)
                rhs=.88*rhs+last*(target[b-1]-last@initial)
                previous=clock[b-1],current[b-1],np.clip(error[-1],-12,12)
        design=np.stack(data);e=design[:,0];ex=design[:,1:]
        self.norm=StandardScaler().fit(ex)
        ex=np.column_stack((np.ones(len(ex)),self.norm.transform(ex).clip(-3,3)))
        self.coef=Ridge(alpha=len(ex)*2.,fit_intercept=False).fit(e[:,None]*ex,np.clip(targets,-16,16)).coef_
        return self

    def strength(self,features):
        f=np.column_stack((np.ones(len(features)),self.norm.transform(features).clip(-3,3)))
        return (f@self.coef).clip(-.25,1.)


class IntervalMaintenance:
    def __init__(self,tools):
        self.base=ElectrodeMaintenance(tools["electrode"])
        self.transfer=tools["transfer"]
        self.prior=None
        self.squared=np.ones(5)*16.; self.rhs=np.zeros(2);self.mass=np.zeros((2,2))

    def predict(self,clock,x,bank):
        out=self.base.predict(x,bank)
        z=np.clip(self.transfer.scale.transform(x),-7,7)
        anchors=np.column_stack((out[BASELINE],out[MIXED]))
        last_error=np.zeros(2);scales=np.zeros((len(x),5))
        if self.prior is not None:
            t,old,err=self.prior
            h=(clock-t)/3600;last_error=np.clip(err,-12,12)
            if (h<=0).any():raise ValueError("参考必须来自已发的过去")
            scales[:,:4]=np.exp(-h[:,None]/np.array(TIMES))
            scales[:,4]=self.transfer.strength(describe_interval(z,old,h))*np.exp(-np.maximum(h-6,0)/12)
        # 在两种基准上都给确定性校正，幅度/持续时长对强固定完全开放。
        for j,label in enumerate(("固定电极","本机混合")):
            for i,key in enumerate((*[str(int(h))+"小时" for h in TIMES],"源期拟合")):
                for gain in (.25,.5,1.):
                    out[f"{label}上次误差{key}乘{gain:g}"]=np.maximum(0,anchors[:,j]+last_error[j]*scales[:,i]*gain)
        w=1/(1+self.squared)**2;w/=w.sum()
        # 用已返参考学习对本机曲线施加多少余差校正；在刚反馈和间隔甚长时区别处理。
        contexts=scales[:,[1,4]]*last_error[1]
        coef=np.linalg.solve(self.mass+np.eye(2)*24,self.rhs+np.array([3.,15.])).clip(-.5,1.5)
        extra=.5*(contexts@coef)+.5*last_error[1]*(scales@w)
        out[PRIMARY]=np.maximum(0,anchors[:,1]+extra)
        out["只用源期误差持续程度"]=np.maximum(0,anchors[:,1]+scales[:,4]*last_error[1])
        out["冻结初始持续权重"]=np.maximum(0,anchors[:,1]+.5*(contexts@np.array([.125,.625]))+.5*last_error[1]*scales.mean(1))
        self.pending=clock[-1],z[-1],anchors[-1],contexts[-1],scales[-1]*last_error[1]
        return out

    def update(self,y):
        t,z,anchors,context,offsets=self.pending
        error=y-anchors
        self.squared=.95*self.squared+.05*np.minimum((error[1]-offsets)**2,625)
        self.rhs=.95*self.rhs+context*error[1]
        self.mass=.95*self.mass+np.outer(context,context)
        self.prior=t,z,error
        self.base.update(y)
