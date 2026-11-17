"""学习相邻时刻功率之差，在两次参考之间从旧预测递推，并用当前量锚定漂移。"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from harness.learning.models.wind_history import WindHistory,HISTORY_EFFECTS,PRIMARY as PREVIOUS,PAPER as PREVIOUS_PAPER
from harness.learning.models.wind_source_control import CONTROL

ANCHOR = "强固定加同结构增减历史占0.5"
PRIMARY = "以本机回标试验重训功率变化并逐窗积分"
PAPER = "论文当前与递推功率各半"
MEMORIES = (0.,.5,.83,.96,1.)


def features(x,prior,h):
    dv = x[:,0]-prior[:,0]
    spin = (x[:,1]**2-prior[:,1]**2)/100
    wind = (x[:,0]**3-prior[:,0]**3)/500
    return np.column_stack((x,prior,spin,wind,np.abs(spin),np.abs(wind),dv*x[:,1],h))


class PowerChange:
    def fit(self,rows,fields):
        xs,ys = [],[]
        for _,d in rows.groupby("turbine_id",sort=True):
            d = d.sort_values("clock")
            x = d[fields].to_numpy()
            clock = d.clock.to_numpy()
            truth = d.power_pu.to_numpy()
            h = WindHistory().transform(clock,x)
            ok = np.r_[False,np.diff(clock)==600]
            prior = np.vstack((x[0],x[:-1]))
            xs.append(features(x,prior,h)[ok])
            ys.append(np.r_[0,np.diff(truth)][ok])
        self.regressor = HistGradientBoostingRegressor(max_iter=240,max_leaf_nodes=23,min_samples_leaf=24,
                        early_stopping=False,random_state=43).fit(np.vstack(xs),np.concatenate(ys))
        return self


class TransitionRelease:
    def __init__(self,base,change):
        self.base = base
        self.change = change

    def predict(self,x):
        return self.base.predict(x)

    def controller(self):
        return TransitionMaintenance(self.base.controller(),self.change)


class TransitionMaintenance:
    def __init__(self,previous,change):
        self.previous = previous
        self.change = change
        self.history = WindHistory()
        self.clock = None
        self.x = None
        self.power = None
        self.risk = np.full((3,len(MEMORIES)),.00045)
        self.g = np.eye(3)*.012
        self.rhs = np.zeros(3)
        self.reference = None
        self.accumulated = np.zeros(3)

    def predict_timed(self,t,x,base):
        old = self.previous.predict_timed(t,x,base)
        keep = (PREVIOUS,PREVIOUS_PAPER,ANCHOR,CONTROL,*HISTORY_EFFECTS,"同样工具按全机累计误差分配","三变量提升树源训练")
        out = {n:old[n] for n in keep}
        p = out[ANCHOR]
        prior = np.vstack((x[0] if self.x is None else self.x,x[:-1]))
        h = self.history.transform(t,x)
        f = features(x,prior,h)
        change = self.change.regressor.predict(f)
        momentum = np.column_stack((change,f[:,6],f[:,7]))
        coef = np.linalg.solve(self.g,self.rhs).clip(-.4,.4)
        delta = change+momentum@coef
        bank = np.empty((len(x),len(MEMORIES)))
        fixed = np.empty_like(bank)
        memory = np.array(MEMORIES)
        zones = np.digitize(x[:,0],[5,10])
        for i,stamp in enumerate(t):
            if self.clock is None or stamp-self.clock!=600:
                self.power = np.full((2,len(MEMORIES)),p[i])
                self.reference = None
                self.accumulated.fill(0.)
            else:
                self.power = np.clip((1-memory)*p[i]+memory*(self.power+np.array([change[i],delta[i]])[:,None]),-.1,1.1)
                self.accumulated += momentum[i]
            fixed[i],bank[i] = self.power
            self.clock = stamp
        self.x = x[-1].copy()
        for j,r in enumerate(MEMORIES):
            out[f"固定递推保留前值{r:g}"] = fixed[:,j]
            out[f"重训变化递推保留前值{r:g}"] = bank[:,j]
        out["按风况固定递推时间"] = np.choose(zones,[bank[:,1],bank[:,2],bank[:,1]])
        out[PAPER] = (old[PREVIOUS_PAPER]+bank[:,2])/2
        weights = 1/(self.risk[zones]+.000035)**2
        out[PRIMARY] = (bank*weights).sum(1)/weights.sum(1)
        out["保持递推初值等权"] = bank.mean(1)
        self.pending = zones[-1],bank[-1].copy()
        return out

    def update(self,y):
        self.previous.update(y)
        z,bank = self.pending
        if self.reference is not None:
            # 两个已返标签间没有缺测才有完整积分；只用真实跨度的变化校正累计误差。
            error = (y-self.reference)-self.accumulated[0]
            u = self.accumulated
            self.g = .97*self.g+np.outer(u,u)+.03*.012*np.eye(3)
            self.rhs = .97*self.rhs+u*error
        self.risk[z] = .95*self.risk[z]+.05*np.minimum((y-bank)**2,.15)
        self.reference = y
        self.accumulated.fill(0.)
        self.power.fill(y)
