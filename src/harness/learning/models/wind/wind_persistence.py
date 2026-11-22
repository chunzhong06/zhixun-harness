"""用源机迁移试验拟合误差的时间和工况相关性；已返功率只参与它后面的预测。"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from harness.learning.models.wind.wind_rotor import RotorMaintenance

INCUMBENT = "原分段与原强分段按转速回标记忆0.94占0.5"
PRIMARY = "按源机试验与本机近期功率误差修正下一槽"
PAPER = "分段论文与回标短时校正合用"
PREVIOUS_PAPER = "论文分段标定与转矩按风况固定合用"


def covariates(t,x,ref_time,ref_x):
    h = (t-ref_time)/3600
    delta = x-ref_x
    return np.column_stack((np.minimum(h,48),np.log1p(h),
                            np.abs(delta[:,0]),np.abs(delta[:,1]),np.abs(delta[:,2]),
                            x[:,0],ref_x[0]*np.ones(len(t)),x[:,1],delta[:,0]))


class ErrorPersistence:
    def fit(self,source,fields):
        features, weights, targets = [],[],[]
        sites = sorted(source.turbine_id.unique())
        # 每台源设备留作一次真实迁入；拟合模型时其余设备只使用该测试起点前的测量。
        for site in sites:
            first = source.clock.min()
            split = int(first + .50*(source.clock.max()-first))
            train = source[(source.turbine_id!=site)&(source.clock<split)]
            test = source[(source.turbine_id==site)&(source.clock>=split)].sort_values("clock")
            fit = HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=23,min_samples_leaf=26,
                      random_state=7,early_stopping=False).fit(train[fields].to_numpy(),train.power_pu.to_numpy())
            x = test[fields].to_numpy()
            t = test.clock.to_numpy()
            raw_err = test.power_pu.to_numpy()-fit.predict(x)
            bounds = np.r_[0,np.flatnonzero(np.diff(t//21600))+1,len(t)]
            previous = None
            offset = 0.
            for a,b in zip(bounds[:-1],bounds[1:],strict=True):
                err = raw_err[a:b]-offset
                if previous is not None:
                    rt,rx,residual = previous
                    if abs(residual)>.001:
                        features.append(covariates(t[a:b],x[a:b],rt,rx))
                        # 加权回归等价于以实际功率残差为单位拟合衰减乘数。
                        targets.extend(err/residual)
                        weights.extend(np.full(len(err),residual**2))
                if b<len(t) and (t[b-1]+600)%21600==0:
                    previous = t[b-1],x[b-1],np.clip(err[-1],-.15,.15)
                    offset += .12*previous[2]
        w = np.array(weights)
        self.model = HistGradientBoostingRegressor(max_iter=75,max_leaf_nodes=9,
            min_samples_leaf=120,l2_regularization=20,early_stopping=False,random_state=7)
        self.model.fit(np.vstack(features),targets,sample_weight=w/w.mean())
        return self


class PersistenceRelease:
    def __init__(self,rotor,decay):
        self.rotor = rotor
        self.decay = decay

    def predict(self,x):
        return self.rotor.predict(x)

    def controller(self):
        return PersistenceMaintenance(self.decay)


class PersistenceMaintenance:
    def __init__(self,decay):
        self.old = RotorMaintenance()
        self.decay = decay
        self.reference = None
        self.gram = np.eye(4)*.01
        self.rhs = np.zeros(4)

    def predict_timed(self,t,x,base):
        old = self.old.predict(x,base)
        # 保留前次主现场、完整最强固定与论文以及审计字段。旧预测的评分原件继续独立保存。
        keep = (INCUMBENT, "据实际回标选择转速偏离下的功率校正", PREVIOUS_PAPER,
                "开发冻结三段配比","同样工具按全机累计误差分配","三变量提升树源训练")
        out = {n:old[n] for n in keep}
        p = out[INCUMBENT]
        if self.reference is None:
            factors = np.zeros((len(t),4))
            h = np.zeros(len(t))
            residual = 0.
        else:
            rt,rx,residual = self.reference
            f = covariates(t,x,rt,rx)
            h = (t-rt)/3600
            same = np.exp(-abs(x[:,0]-rx[0])/2)
            learned = self.decay.model.predict(f).clip(-.2,1)
            factors = np.column_stack((learned,np.exp(-h/3),np.exp(-h/3)*same,
                                       np.exp(-h/12)*same))
        for half in (1,3,6,12):
            for amplitude in (.25,.5,.75):
                out[f"固定回标校正{half}小时乘{amplitude:g}"] = p+amplitude*np.exp(-h/half)*residual
        for i,label in enumerate(("源试验学定持续性","时间衰减","同时看风速变化","慢衰减及风速变化")):
            out[label] = p+residual*factors[:,i]
        feats = residual*factors
        fixed = .5*factors[:,0]+.25*factors[:,1]+.25*factors[:,2]
        correction = np.linalg.solve(self.gram,self.rhs)
        refined = (fixed+factors@correction).clip(-.25,1)
        out["冻结源试验及风速校正配比"] = p+residual*fixed
        out[PRIMARY] = p+residual*refined
        out[PAPER] = out[PREVIOUS_PAPER]+.5*residual*np.exp(-h/3)
        self.pending = t[-1],x[-1].copy(),p[-1],feats[-1],residual*fixed[-1]
        return out

    def update(self,y):
        self.old.update(y)
        t,x,p,f,previous = self.pending
        self.gram = .97*self.gram+np.outer(f,f)+.03*.01*np.eye(4)
        self.rhs = .97*self.rhs+f*(y-p-previous)
        self.reference = t,x,np.clip(y-p,-.15,.15)
