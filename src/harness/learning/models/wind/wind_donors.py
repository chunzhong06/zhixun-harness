"""每台源机保留独立校准曲线。用目标机已经测得的风与转速检验哪一台可迁移。"""
import numpy as np
from harness.learning.models.wind.wind_limit import ControlCurve
from harness.learning.models.wind.wind_rotor import hats
from harness.learning.models.wind.wind_reference import PRIMARY as PREVIOUS

ANCHOR = "原固定加同温同风长期误差占0.25"
PRIMARY = "用无标签转速与真实回标挑选源机校准"
PAPER = "论文上限修正与三台源机校准各半"
PREVIOUS_PAPER = "论文短时校准与转速上限校正合用"


class SourceMachines:
    def fit(self,rows,fields):
        self.curves = []
        self.profiles = []
        for _,unit in rows.groupby("turbine_id",sort=True):
            x = unit[fields].to_numpy()
            self.curves.append(ControlCurve().fit(x,unit.power_pu.to_numpy()))
            # 不同风速下叶轮的实际典型转速可在无功率参考时照样观察。
            v,r,_ = x.T
            bins = np.floor(v*2).astype(int)
            groups = [g for g in np.unique(bins) if (bins==g).sum()>=12]
            m = np.array([np.median(r[bins==g]) for g in groups])
            sd = np.array([max(.25,np.std(r[bins==g])) for g in groups])
            self.profiles.append(((np.asarray(groups)+.5)/2,m,sd))
        return self

    def predict(self,x):
        power = [model.predict(x)[:,0] for model in self.curves]
        mismatch = [(x[:,1]-np.interp(x[:,0],v,r))/np.interp(x[:,0],v,sd) for v,r,sd in self.profiles]
        return np.column_stack((*power,*mismatch))


class DonorRelease:
    def __init__(self,old,sources):
        self.previous = old
        self.sources = sources

    def predict(self,x):
        return np.column_stack((self.previous.predict(x),self.sources.predict(x)))

    def controller(self):
        return DonorMaintenance(self.previous.controller())


class DonorMaintenance:
    def __init__(self,previous):
        self.old = previous
        self.g = np.zeros((6,6))
        self.rhs = np.zeros((6,3))
        self.loss = np.full((3,4),.0005)
        self.distance = np.zeros((3,3))
        self.count = np.zeros(3)
        self.calibration_g = np.zeros(3)
        self.calibration_r = np.zeros(3)

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base[:,:-6])
        keep = (ANCHOR,PREVIOUS,PREVIOUS_PAPER,"同样工具按全机累计误差分配","三变量提升树源训练",
                "由已返本机功率试验决定转速上限曲线的修正")
        out = {n:old[n] for n in keep}
        p = old[ANCHOR]
        raw,mis = base[:,-6:-3],base[:,-3:]
        v = x[:,0]
        f = hats(v,np.array([2.,5.,7.5,9.,11.,14.]))
        g = np.linalg.solve(self.g+2*np.eye(6),self.rhs)
        bank = (raw+f@g).clip(-.1,1.1)
        zone = np.digitize(v,[5,10])
        weights = 1/(self.loss[zone,1:]+.00005)**2
        paid = (bank*weights).sum(1)/weights.sum(1)
        blind = np.empty(len(x))
        # 每个时刻仅递推自身及过去的读数，不能一次用完整服务批次的输入选源机。
        for i,z in enumerate(zone):
            self.distance[z] = .99*self.distance[z]+np.minimum(mis[i]**2,9.)
            self.count[z] = .99*self.count[z]+1.
            similarity = np.exp(-3.*self.distance[z]/(10.+self.count[z]))
            blind[i] = bank[i]@similarity/similarity.sum()
        candidates = {**{f"源机{k+1}":bank[:,k] for k in range(3)},
                      "三机等权":bank.mean(1),"无标签匹配":blind,"回标匹配":paid}
        for name,forecast in candidates.items():
            out[name+"功率曲线"] = forecast
            for w in (.25,.5,.75):
                out[f"原曲线加{name}占{w:g}"] = p+w*(forecast-p)
        out["开发常规风速分段选源"] = p+.25*(np.choose(zone,[bank[:,2],bank[:,1],bank[:,0]])-p)
        mixture = (blind+paid)/2
        alpha = (.25*.002+self.calibration_r)/(.002+self.calibration_g)
        out[PRIMARY] = p+alpha[zone].clip(0,.85)*(mixture-p)
        out["冻结匹配源机初始占比"] = p+.25*(mixture-p)
        out[PAPER] = (old[PREVIOUS_PAPER]+bank.mean(1))/2
        self.pending = f[-1],raw[-1],np.r_[p[-1],bank[-1]],zone[-1],mixture[-1]-p[-1]
        return out

    def update(self,y):
        self.old.update(y)
        f,raw,issued,k,d = self.pending
        self.g = .98*self.g+np.outer(f,f)
        self.rhs = .98*self.rhs+f[:,None]*(y-raw)
        self.loss[k] = .93*self.loss[k]+.07*np.minimum((y-issued)**2,.09)
        self.calibration_g[k] = .96*self.calibration_g[k]+d*d
        self.calibration_r[k] = .96*self.calibration_r[k]+d*(y-issued[0])
