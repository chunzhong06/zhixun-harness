"""持续测量同转速下的风速偏差，用已购功率训练偏差与本机功率之间的关系。"""
import numpy as np
from harness.models.wind_history import tree,HISTORY_EFFECTS,CONTROL,PRIMARY as HISTORY_SITE
from harness.models.wind_rotor import hats
from harness.models.wind_dynamic_head import PRIMARY as PREVIOUS,ANCHOR

PRIMARY = "从无标签风速偏移与已返功率试验决定是否校准"
OLD_PAPER = "原固定与论文A1连续风况全层微调占0.1"
PAPER = "论文历史功率网络与本机风速校准各半"


class InputMap:
    def fit(self,rows,fields):
        x = rows[fields].to_numpy()
        self.reference = tree().fit(x[:,[1,2]],x[:,0])
        self.power = tree().fit(x,rows.power_pu)
        return self

    def predict(self,x):
        higher = x.copy()
        lower = x.copy()
        higher[:,0] += .25
        lower[:,0] -= .25
        slope = (self.power.predict(higher)-self.power.predict(lower))/.5
        return np.column_stack((self.reference.predict(x[:,[1,2]]),slope.clip(-.03,.35)))


class MappingRelease:
    def __init__(self,previous,mapping):
        self.old = previous
        self.mapping = mapping

    def predict(self,x):
        return np.column_stack((self.old.predict(x),self.mapping.predict(x)))

    def controller(self):
        return MappingMaintenance(self.old.controller())


class MappingMaintenance:
    def __init__(self,old):
        self.old = old
        self.nodes = np.array([1,6,8,9.5,11,12.5,13.5,14.3,14.8,15.05,15.18,15.5])
        self.n = np.zeros((2,len(self.nodes)))
        self.s = np.zeros_like(self.n)
        self.last = None
        self.g = np.zeros((3,4,4))
        self.rhs = np.zeros((3,4))

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base[:,:-2])
        keep = (PREVIOUS,OLD_PAPER,ANCHOR,HISTORY_SITE,CONTROL,*HISTORY_EFFECTS,"三变量提升树源训练","同样工具按全机累计误差分配")
        out = {n:old[n] for n in keep}
        p = out[ANCHOR]
        expected,slope = base[:,-2:].T
        distance = x[:,0]-expected
        basis = hats(x[:,1],self.nodes)
        bias = np.zeros((len(t),2))
        for i,(time,f) in enumerate(zip(t,basis,strict=True)):
            if self.last is not None:
                decay = np.exp(-(time-self.last)/(3600*np.array([12.,168.])))
                self.n *= decay[:,None]
                self.s *= decay[:,None]
            # 估计器只用该时刻之前的测量，无须目标机功率，也无需在缺测间插值。
            bias[i] = (self.s/(self.n+5.))@f
            self.n += f
            self.s += f*np.clip(distance[i],-2,2)
            self.last = time
        raw = np.column_stack((slope*bias[:,0],slope*bias[:,1],
                   slope*(bias[:,0]-bias[:,1]),bias[:,1]*.025))
        zone = np.digitize(x[:,1],[10.,14.7])
        for j,label in enumerate(("半日","一周")):
            for weight in (-.5,-.25,.25,.5):
                out[f"固定{label}风速偏移系数{weight:g}"] = p+weight*raw[:,j]
        for ridge in (.002,.012,.06):
            coeff = np.linalg.solve(self.g+ridge*np.eye(4),self.rhs[:,:,None])[:,:,0].clip(-1.5,1.5)
            out[f"回标训练风速偏移先验{ridge:g}"] = p+(raw*coeff[zone]).sum(1).clip(-.08,.08)
        out[PRIMARY] = (out["回标训练风速偏移先验0.012"]+p)/2
        out["固定按转速映射一周偏移"] = p+np.choose(zone,[-.25,0.,.25])*raw[:,1]
        out[PAPER] = old[OLD_PAPER]+.5*(out["回标训练风速偏移先验0.012"]-p)
        self.pending = zone[-1],raw[-1],p[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,f,p = self.pending
        self.g[z] = .98*self.g[z]+np.outer(f,f)
        self.rhs[z] = .98*self.rhs[z]+f*(y-p)
