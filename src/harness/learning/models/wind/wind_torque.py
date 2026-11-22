"""分别学习功率和单位转速的功率，再用已返功率修正本机在各风速区间的偏差。"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from harness.learning.models.wind.wind_power import WindMaintenance

INCUMBENT = "固定低风树中风提升高风曲线"
PRIMARY = "在本机实测中选择功率或转矩标定"
PAPER = "原分段论文与转矩标定各半"


class TorqueRelease:
    def fit(self, x, y):
        self.models = []
        cols = self.design(x)
        rpm = np.maximum(x[:,1],2.)
        # 低转速设下界。此处估计的是与转矩成比例的量，最终统一还原为功率评分。
        for truth in (y,15*y/rpm):
            tree = HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=23,min_samples_leaf=30,
                                  early_stopping=False,random_state=7).fit(cols,truth)
            self.models.append(tree)
        return self

    def design(self,x):
        v,r,temp = x.T
        return np.column_stack((v,r,temp,r/np.maximum(v,2),v**3/(273.15+temp),r**3))

    def predict(self,x):
        f = self.design(x)
        p,q = [m.predict(f) for m in self.models]
        return np.column_stack((p,q*np.maximum(x[:,1],2.)/15)).clip(-.1,1.1)


class TorqueMaintenance:
    def __init__(self):
        self.old = WindMaintenance()
        self.gram = np.zeros((7,7))
        self.rhs = np.zeros((7,3))
        self.last = None
        self.risk = np.full((3,3),.0006)

    def predict(self,x,base):
        out = self.old.predict(x,base[:,:5])
        initial = np.column_stack((out[INCUMBENT],base[:,5:]))
        v,r,_ = x.T
        zones = np.digitize(v,[5,10])
        # 在高风见到足够的参考前，以全量程和相邻风段共同锚定校准。
        centers = np.array([2.,5.,8.,11.,14.])
        near = np.exp(-.5*((v[:,None]-centers)/2.)**2)
        features = np.column_stack((np.ones(len(x)),near,(r-12)/8))
        coef = np.linalg.solve(self.gram+np.diag([5.,4.,4.,4.,4.,4.,7.]),self.rhs)
        bank = (initial+features@coef).clip(-.1,1.1)
        for j,name in enumerate(("原分段","直接学功率","先学转矩再还原功率")):
            out[name+"保持"] = initial[:,j]
            out[name+"按本机风速重新标定"] = bank[:,j]
            for w in (.25,.5,.75):
                out[f"原分段与{name}标定占{w:g}"] = initial[:,0]+w*(bank[:,j]-initial[:,0])
        out[PAPER] = .5*(initial[:,0]+bank[:,2])
        out["三个分风速标定等权"] = bank.mean(1)
        out["三个分风速标定各半原固定"] = (bank.mean(1)+initial[:,0])/2
        weight = 1/((self.risk[zones]+.00005)**2)
        out[PRIMARY] = (bank*weight).sum(1)/weight.sum(1)
        self.last = features[-1],initial[-1],bank[-1],zones[-1]
        return out

    def update(self,y):
        self.old.update(y)
        f,p,q,zone = self.last
        self.gram = .97*self.gram+np.outer(f,f)
        self.rhs = .97*self.rhs+f[:,None]*(y-p)
        self.risk[zone] = .9*self.risk[zone]+.1*np.minimum((y-q)**2,.2)


class CombinedWindRelease:
    def __init__(self, original, torque):
        self.original = original
        self.torque = torque

    def predict(self,x):
        return np.column_stack((self.original.predict(x),self.torque.predict(x)))
