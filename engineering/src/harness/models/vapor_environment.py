"""用同层温湿度修正相对电阻；每完成一批才可按它的末窗参考更新。"""
import numpy as np
from scipy.special import expit

from harness.models.vapor_field import ProbeEncoder, voltage_ratio
from harness.models.vapor_boundary import LevelFit

CHANNELS=("bme680-temp","bme680-humidity")
AUTHOR="作者温湿补偿及压缩论文读法"
JOINT="温湿变化与储备池联合训练"
PRIMARY="依已返参考分别训练温湿补偿幅度"
FIXED="固定源端温湿组合"


def fixed_grid(bank):
    """列顺序为原强校正、作者补偿、环境联合训练；固定权重与现场使用同一压缩域。"""
    base,author,joint=np.arcsinh(np.asarray(bank)).T
    return {f"固定温湿比例{a}-{b}":np.sinh(np.clip(base+a/10*(author-base)+b/10*(joint-base),-5,5))
            for a in range(11) for b in range(11-a)}


def shift(env):
    env=np.asarray(env,dtype=float)
    if env.ndim!=2 or env.shape[1]!=2 or not np.isfinite(env).all() or np.any((env[:,1]<0)|(env[:,1]>100)) or np.any(env[:,0]<=-273.15):
        raise ValueError("温湿度通道不完整")
    return np.column_stack((3073*(1/(env[:,0]+273.15)-1/298.15),np.log1p(-.005*(env[:,1]-50))))


class EnvironmentalProbe:
    def __init__(self,warm,environment,scale):
        self.env0=np.median(shift(np.vstack(environment)),axis=0)
        corrected=[self.compensate(v,e) for v,e in zip(warm,environment,strict=True)]
        self.sensor=ProbeEncoder(np.median(voltage_ratio(np.vstack(corrected)),axis=0),scale)
        self.previous=np.zeros(2)

    def compensate(self,values,environment):
        env=shift(environment)
        if len(env)!=len(values):
            raise ValueError("每帧气体及温湿度必须同步")
        response=voltage_ratio(values)-(env-self.env0).sum(1)[:,None]
        # 恢复到原电桥方向的等效电压，沿用既有因果储备池和同一标度。
        return 5*expit(response*np.array([1.,1.,-1.]))

    def next(self,v,env):
        corrected=self.compensate(v,env)
        views,_=self.sensor.next(corrected)
        delta=(shift(env).mean(0)-self.env0)*100
        change=delta-self.previous
        self.previous=delta
        raw=views[5]
        interaction=np.outer(delta,raw[:3]).ravel()
        return raw,np.r_[raw,delta,change,interaction]


class EnvironmentReadout:
    def fit(self,records):
        self.models=[]
        y=np.concatenate([r[1][20:] for r in records])
        for j in (0,1):
            rows=np.vstack([xs[j][20:] for xs,_ in records])
            self.models.append(LevelFit().fit(rows,y))
        return self

    def begin(self,warm,environment,scale):
        self.encoder=EnvironmentalProbe(warm,environment,scale)
        for v,e in zip(warm,environment,strict=True):
            self.encoder.next(v,e)
        self.gram=np.eye(2)*6
        self.rhs=np.array([1.,1.])

    def measure(self,raw,env,base):
        self.x=[v[None,:] for v in self.encoder.next(raw,env)]
        pred=np.array([m.predict(x)[0,1] for m,x in zip(self.models,self.x,strict=True)])
        self.base=np.arcsinh(base)
        self.deltas=np.arcsinh(pred)-self.base
        coef=np.linalg.solve(self.gram,self.rhs).clip(0.,1.)
        coef/=max(1.,coef.sum())
        def mix(w):
            return float(np.sinh(np.clip(self.base+self.deltas@w,-5,5)))
        out={AUTHOR:float(pred[0]),JOINT:float(pred[1]),PRIMARY:mix(coef),FIXED:mix(np.ones(2)/6)}
        out["固定温湿比例0-1"]=mix(np.array([0.,.1]))
        for j,name in enumerate((AUTHOR,JOINT)):
            for k in (.25,.5,.75):
                w=np.zeros(2);w[j]=k
                out[f"原论文与{name}按{k:g}补偿"]=mix(w)
        return out

    def feedback(self,truth):
        target=np.arcsinh(truth)-self.base
        near=1/(.3+abs(np.arcsinh(truth)-np.arcsinh(1)))
        # 只对上一批实际交付过的末窗学习额外的环境修正。
        self.gram=.95*self.gram+.05*6*np.eye(2)+near*np.outer(self.deltas,self.deltas)
        self.rhs=.95*self.rhs+.05*np.ones(2)+near*self.deltas*target
        for m,x in zip(self.models,self.x,strict=True):
            m.update(x,np.array([truth]))
