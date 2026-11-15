"""按连续十分钟还原先前风速和转速，用同一瞬时工况的对照训练短时变化对功率的贡献。"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from harness.models.wind_pmlp import PRIMARY as PREVIOUS
from harness.models.wind_source_control import CONTROL,PRIMARY as PREVIOUS_BEST

OLD_PAPER = "原固定与论文A1源网络占0.1"
PRIMARY = "根据本机已返功率决定短时风况修正"
PAPER = "论文源网络组合加连续风况修正"
HISTORY_EFFECTS = ("强固定加同结构增减历史占1","强固定加相同当前工况替换历史占1")


class WindHistory:
    def __init__(self):
        self.clock = None
        self.previous = None
        self.smooth = None
        self.spread = None

    def transform(self,t,x):
        data = np.empty((len(x),13))
        decay = np.exp(-1/np.array([3.,12.]))[:,None]
        for i,(stamp,values) in enumerate(zip(t,x,strict=True)):
            valid = self.clock is not None and stamp-self.clock==600
            if not valid:
                self.previous = values
                self.smooth = np.tile(values[:2],(2,1))
                self.spread = np.zeros((2,2))
            step = values-self.previous
            shift = values[:2]-self.smooth
            self.smooth = self.smooth+(1-decay)*shift
            self.spread = decay*(self.spread+(1-decay)*shift**2)
            data[i] = np.r_[step,shift.ravel(),np.sqrt(self.spread).ravel(),abs(step[0]),float(valid)]
            self.clock = stamp
            self.previous = values.copy()
        return data


def tree():
    return HistGradientBoostingRegressor(max_iter=165,max_leaf_nodes=17,
                learning_rate=.075,min_samples_leaf=40,early_stopping=False,random_state=43)


class Dynamics:
    def fit(self,rows,fields):
        inputs,changes,targets,sites,times = [],[],[],[],[]
        for site,device in rows.groupby("turbine_id",sort=True):
            device = device.sort_values("clock")
            x = device[fields].to_numpy()
            t = device.clock.to_numpy()
            inputs.append(x)
            changes.append(WindHistory().transform(t,x))
            targets.append(device.power_pu.to_numpy())
            sites.extend([site]*len(x))
            times.append(t)
        x,h,y,t = np.vstack(inputs),np.vstack(changes),np.concatenate(targets),np.concatenate(times)
        sites = np.array(sites)
        f = np.column_stack((x,h))
        self.current = tree().fit(x,y)
        self.history = tree().fit(f,y)
        # 源端各机后半段相对另外两机更早资料的实际误差，可作为拟合动态偏差的训练标签。
        past = t<np.median(t)
        residual = np.empty(len(x))
        for site in np.unique(sites):
            tr = (sites!=site)&past
            te = (sites==site)&~past
            baseline = tree().fit(x[tr],y[tr])
            residual[te] = y[te]-baseline.predict(x[te])
        self.resid = tree().fit(f[~past],residual[~past])
        self.center = x.mean(0)
        self.scale = np.maximum(x.std(0),.2)
        self.x = x
        self.h = h
        self.lookup = NearestNeighbors(n_neighbors=9,algorithm="kd_tree").fit((x-self.center)/self.scale)
        return self

    def predict(self,x,h):
        f = np.column_stack((x,h))
        direct = self.history.predict(f)
        absolute = self.current.predict(x)
        ix = self.lookup.kneighbors((x-self.center)/self.scale,return_distance=False)
        # 保持当前三个读数完全相同，只替换为源机相近时刻曾出现过的历史，隔离变化方向的增量。
        contrast = np.column_stack((np.repeat(x,ix.shape[1],axis=0),self.h[ix.ravel()]))
        ref = self.resid.predict(contrast).reshape(ix.shape).mean(1)
        centered = self.resid.predict(f)-ref
        return np.column_stack((direct-absolute,centered,direct))


class HistoryRelease:
    def __init__(self,base,dynamics):
        self.base = base
        self.dynamics = dynamics

    def predict(self,x):
        return self.base.predict(x)

    def controller(self):
        return HistoryMaintenance(self.base.controller(),self.dynamics)


class HistoryMaintenance:
    def __init__(self,old,tools):
        self.old = old
        self.tools = tools
        self.history = WindHistory()
        self.g = np.zeros((3,2,2))
        self.rhs = np.zeros((3,2))
        self.pending = None

    def predict_timed(self,t,x,base):
        previous = self.old.predict_timed(t,x,base)
        keep = (PREVIOUS,CONTROL,OLD_PAPER,PREVIOUS_BEST,"三变量提升树源训练","同样工具按全机累计误差分配")
        out = {n:previous[n] for n in keep}
        p = out[CONTROL]
        h = self.history.transform(t,x)
        effects = self.tools.predict(x,h)
        raw = effects[:,:2]
        out["直接由连续读数预测"] = effects[:,2]
        for i,name in enumerate(("同结构增减历史","相同当前工况替换历史")):
            for w in (.25,.5,.75,1.):
                out[f"强固定加{name}占{w:g}"] = p+w*raw[:,i]
        out[PAPER] = out[OLD_PAPER]+.5*raw[:,1]
        zone = np.digitize(x[:,0],[5,10])
        initial = np.array([.10,.45])
        for ridge in (.0005,.003,.02):
            coeff = np.linalg.solve(self.g+ridge*np.eye(2),(self.rhs+ridge*initial)[:,:,None])[:,:,0].clip(-.5,1.5)
            out[f"风况回标系数先验{ridge:g}"] = p+(raw*coeff[zone]).sum(1)
        out["冻结源期风况修正"] = p+raw@initial
        out["按风速冻结历史差分幅度"] = p+np.choose(zone,[.25,.5,1.])*raw[:,1]
        out[PRIMARY] = (out["风况回标系数先验0.003"]+out["冻结源期风况修正"])/2
        self.pending = zone[-1],raw[-1],p[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,d,p = self.pending
        self.g[z] = .97*self.g[z]+np.outer(d,d)
        self.rhs[z] = .97*self.rhs[z]+d*(y-p)
