"""按转速是否接近调速上限分别拟合功率曲线，逐次回标重估本机的上限与风速偏差。"""
import numpy as np
from harness.learning.models.wind_rotor import hats
from harness.learning.models.wind_persistence import PersistenceRelease,PRIMARY as PREVIOUS, PAPER as OLD_PAPER

ANCHOR = "固定回标校正3小时乘0.25"
PRIMARY = "由已返本机功率试验决定转速上限曲线的修正"
PAPER = "论文短时校准与转速上限校正合用"


class ControlCurve:
    def fit(self,rows,y):
        v,r,temp = rows.T
        # 用源期高风下的转速判定调速边界，不用开发机的功率或最大值估计边界。
        self.limit = np.median(r[v>np.quantile(v,.95)])
        self.temp = np.mean(temp)
        self.nodes = np.unique(np.r_[np.linspace(0,self.limit*.90,18),
                                   self.limit+np.array([-1.3,-1,-.8,-.6,-.45,-.30,-.20,-.12,-.06,0,.06,.12,.25,.45])])
        self.wind_nodes = np.array([2.,4.,6.,7.5,8.5,9.5,10.5,11.5,13.,15.,18.])
        bins = np.argmin(abs(r[:,None]-self.nodes),axis=1)
        good = [i for i in range(len(self.nodes)) if (bins==i).sum()>=12]
        self.typical = np.interp(self.nodes,self.nodes[good],[np.median(v[bins==i]) for i in good])
        f = self.features(rows)
        # 对低速的功率读数与饱和区的风速响应分别施加平滑约束。
        n,m = len(self.nodes),len(self.wind_nodes)
        penalty = .08*np.eye(f.shape[1])
        for a,h in ((0,n),(n,n),(2*n,m),(2*n+m,m),(2*n+2*m,m)):
            d = np.diff(np.eye(h),n=2,axis=0)
            penalty[a:a+h,a:a+h] += 2*d.T@d
        self.weights = np.linalg.solve(f.T@f+penalty,f.T@y)
        return self

    def gate(self,rpm):
        return np.clip((rpm-self.limit+1.25)/1.15,0.,1.)

    def features(self,x):
        v,r,t = x.T
        rv = hats(r,self.nodes)
        dv = (v-np.interp(r,self.nodes,self.typical)).clip(-5,5)
        gate = self.gate(r)
        vw = hats(v,self.wind_nodes)*gate[:,None]
        return np.column_stack((rv,rv*dv[:,None],vw,vw*((r-self.limit).clip(-1.5,.5))[:,None],
                                vw*((t-self.temp)/10)[:,None]))

    def predict(self,x):
        p = (self.features(x)@self.weights).clip(-.1,1.15)
        # 用微小物理单位偏移度量曲线对风速、转速校准误差的灵敏度。
        slopes = []
        for j,step in ((0,.12),(1,.06)):
            hi = x.copy()
            lo = x.copy()
            hi[:,j] += step
            lo[:,j] -= step
            slopes.append(((self.features(hi)-self.features(lo))@self.weights/(2*step)).clip(-.6,.6))
        return np.column_stack((p,*slopes,self.gate(x[:,1])))


class LimitRelease(PersistenceRelease):
    def __init__(self,previous,curve):
        self.previous = previous
        self.curve = curve

    def predict(self,x):
        return np.column_stack((self.previous.predict(x),self.curve.predict(x)))

    def controller(self):
        return LimitMaintenance(self.previous.controller())


class LimitMaintenance:
    def __init__(self,old):
        self.old = old
        self.decays = np.array([.90,.98,1.])
        self.grams = np.zeros((3,9,9))
        self.rhs = np.zeros((3,9))
        self.last = None
        self.cov = np.zeros((3,2))
        self.cross = np.zeros((3,2))

    def predict_timed(self,t,x,predictions):
        prev = self.old.predict_timed(t,x,predictions[:,:9])
        out = {n:prev[n] for n in (PREVIOUS,ANCHOR,OLD_PAPER,"同样工具按全机累计误差分配","三变量提升树源训练")}
        anchor = out[ANCHOR]
        p,dv,dr,gate = predictions[:,9:].T
        v,r,temp = x.T
        scaled_temp = ((temp-10)/10).clip(-3,3)
        rated = np.clip((v-8)/4,0,1)*gate
        f = np.column_stack((np.ones(len(v)),p,dv,dr,gate,rated,rated*scaled_temp,
                            gate*dv,(1-gate)*dr))
        penalty = np.diag([5.,3.,.3,.3,1.5,1.,1.5,.5,.5])
        updates = np.linalg.solve(self.grams+penalty,self.rhs[:,:,None])[:,:,0]
        curves = np.clip(p[:,None]+f@updates.T,-.1,1.15)
        if self.last is not None:
            stamp,errs = self.last
            curves += .25*np.exp(-(t-stamp)[:,None]/10800)*errs
        delta = curves-anchor[:,None]
        fixed_zone = np.where(gate<.5,.8,.3)
        out["源端转速上限曲线"] = p
        for i,decay in enumerate(self.decays):
            out[f"上限与灵敏度校正{decay:g}"] = curves[:,i]
            for w in (.25,.5,.75):
                out[f"原固定加上限校正{decay:g}占{w:g}"] = anchor+w*delta[:,i]
        out["固定按转速切换灵敏度曲线"] = anchor+fixed_zone*delta[:,1]
        out[PAPER] = (out[OLD_PAPER]+curves[:,1])/2
        zones = np.digitize(gate,[.15,.85])
        w = (self.cross+.002*np.array([.5,.5]))/(self.cov+.002)
        out[PRIMARY] = anchor+.5*np.sum(w[zones].clip(0,1)*delta[:,[0,1]],axis=1)
        self.pending = t[-1],f[-1],p[-1],anchor[-1],zones[-1],delta[-1,[0,1]]
        # 下一次回标用于估计新曲线的误差，避免把刚加的短期修正再累计一遍。
        self.raw_last = (p[-1]+updates@f[-1]).clip(-.1,1.15)
        return out

    def update(self,y):
        self.old.update(y)
        stamp,f,p,anchor,zone,d = self.pending
        self.grams = self.decays[:,None,None]*self.grams+np.outer(f,f)
        self.rhs = self.decays[:,None]*self.rhs+f*(y-p)
        self.cov[zone] = .96*self.cov[zone]+d*d
        self.cross[zone] = .96*self.cross[zone]+d*(y-anchor)
        self.last = stamp,np.clip(y-self.raw_last,-.15,.15)
