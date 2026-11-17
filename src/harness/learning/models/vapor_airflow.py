"""用同步风速调节慢元件的读数记忆，每个模型均可从已确认的末窗参考重新训练。"""
import numpy as np

from harness.learning.models.vapor_boundary import LevelFit
from harness.learning.models.vapor_field import voltage_ratio

WIND=("wind-u","wind-v","wind-w")
READERS=("固定时间记忆","随局部风速更新记忆","气流和瞬态响应联训")
PRIMARY="由已回参考选择气流时间模型"
PREVIOUS="固定温湿比例0-1"


class AirHistory:
    def __init__(self,warm,scale):
        self.base=np.median(voltage_ratio(np.vstack(warm)),axis=0)
        self.scale=scale
        self.last=None
        self.memory=None
        self.taus=np.array([.25,1.,4.,12.])[:,None]

    def next(self,raw,wind,clock):
        gas=(voltage_ratio(raw)-self.base)/self.scale
        clock=np.asarray(clock,dtype=float)
        wind=np.asarray(wind,dtype=float)
        if wind.shape!=gas.shape or len(clock)!=len(gas) or not np.isfinite(wind).all() or not np.isfinite(clock).all():
            raise ValueError("风速和气体需逐帧同步")
        speed=np.linalg.norm(wind,axis=1)
        for v,u,t in zip(gas,speed,clock,strict=True):
            if self.last is None:
                self.memory=np.broadcast_to(v,(2,4,3)).copy()
            else:
                if t<=self.last:raise ValueError("风速读数时间必须递增")
                speed_factor=np.array([1.,np.clip(u/1.4,.15,3.)])[:,None,None]
                self.memory+=-np.expm1(-(t-self.last)*speed_factor/self.taus)*(v-self.memory)
            self.last=t
        mean=gas.mean(0)
        fixed,flow=[np.r_[mean,m.ravel(),(mean-m).ravel(),(gas[-1]-gas[0])] for m in self.memory]
        w=wind.mean(0)
        return (fixed,flow,np.r_[flow,w,np.std(wind,axis=0),np.linalg.norm(w),
                                 np.outer(w,self.memory[1,0]-self.memory[1,-1]).ravel()])


class AirReadout:
    def fit(self,records):
        y=np.concatenate([t[20:] for _,t in records])
        self.models=[LevelFit().fit(np.vstack([v[i][20:] for v,_ in records]),y) for i in range(3)]
        return self

    def begin(self,warm,wind,clocks,scale):
        self.memory=AirHistory(warm,scale)
        for v,w,t in zip(warm,wind,clocks,strict=True):
            self.memory.next(v,w,t)
        self.gram=8*np.eye(3)
        self.rhs=np.ones(3)*.5

    def measure(self,raw,wind,clock,previous):
        self.x=[a[None,:] for a in self.memory.next(raw,wind,clock)]
        pred=np.array([model.predict(x)[0,1] for model,x in zip(self.models,self.x,strict=True)])
        self.base=np.arcsinh(previous)
        self.delta=np.arcsinh(pred)-self.base
        def mix(weights):
            return float(np.sinh(np.clip(self.base+self.delta@weights,-5,5)))
        out=dict(zip(READERS,pred.tolist(),strict=True))
        for j,name in enumerate(READERS):
            for fraction in (.1,.25,.5,.75):
                w=np.zeros(3);w[j]=fraction
                out[f"论文和{name}占{fraction:g}"]=mix(w)
        for fraction in (.1,.25,.5):
            out[f"论文和三种气流读法平均占{fraction:g}"]=mix(np.full(3,fraction/3))
        out["固定气流初始权重"]=mix(np.full(3,.5/8))
        w=np.linalg.solve(self.gram,self.rhs).clip(0,1)
        out[PRIMARY]=mix(w/max(w.sum(),1))
        return out

    def feedback(self,y):
        # 上一窗更靠近分界时，给它在已发误差拟合中更多权重。
        w=1/(.3+abs(np.arcsinh(y)-np.arcsinh(1.)))
        self.gram=.92*self.gram+.08*8*np.eye(3)+w*np.outer(self.delta,self.delta)
        self.rhs=.92*self.rhs+.08*np.full(3,.5)+w*self.delta*(np.arcsinh(y)-self.base)
        for model,x in zip(self.models,self.x,strict=True):
            model.update(x,np.array([y]))
