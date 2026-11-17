"""由源期风速和转速的对应关系拟合连续功率面，现场用已确认的功率重估局部响应。"""
import numpy as np
from harness.learning.models.wind_regime import RegimeMaintenance, RegimeRelease, CONTROL, PAPER as OLD_PAPER

PRIMARY = "据实际回标选择转速偏离下的功率校正"
PAPER = "分段论文与连续转速功率面各半"


def hats(values, nodes):
    """在相邻两个结点间分配权重，超出范围时保持边缘结点。"""
    x = np.clip(values,nodes[0],nodes[-1])
    j = np.searchsorted(nodes,x,side="right").clip(1,len(nodes)-1)
    fraction = (x-nodes[j-1])/(nodes[j]-nodes[j-1])
    out = np.zeros((len(x),len(nodes)))
    rows = np.arange(len(x))
    out[rows,j-1] = 1-fraction
    out[rows,j] = fraction
    return out


class RotorSurface:
    def fit(self,x,y):
        wind,rpm,temp = x.T
        intervals = np.floor(wind*4).astype(int)
        bins = [b for b in np.unique(intervals) if (intervals==b).sum()>=20]
        self.wind = (np.asarray(bins)+.5)/4
        self.median = np.array([np.median(rpm[intervals==b]) for b in bins])
        self.sd = np.array([max(.2,rpm[intervals==b].std(ddof=1)) for b in bins])
        self.temp = np.mean(temp)
        z = self.departure(x)
        self.knots = np.unique(np.r_[np.quantile(wind,np.linspace(0,1,31)),8,10,12,14,17])
        self.offsets = np.array([-4.,-2.,-1.,-.4,0.,.4,1.,2.,4.])
        f = self.design(x,z)
        # 二阶差分正则使邻近风速和相邻转速区间借用源数据；稀有极值保留源端约束。
        n,m = len(self.knots),len(self.offsets)
        dx = np.kron(np.diff(np.eye(n),n=2,axis=0),np.eye(m))
        dz = np.kron(np.eye(n),np.diff(np.eye(m),n=2,axis=0))
        penalty = .12*np.eye(f.shape[1])
        penalty[:n*m,:n*m] += 2*(dx.T@dx)+3*(dz.T@dz)
        self.coef = np.linalg.solve(f.T@f+penalty,f.T@y)
        return self

    def departure(self,x):
        wind,rpm,_ = x.T
        return ((rpm-np.interp(wind,self.wind,self.median))/np.interp(wind,self.wind,self.sd)).clip(-5,5)

    def design(self,x,z):
        v = hats(x[:,0],self.knots)
        r = hats(z,self.offsets)
        product = (v[:,:,None]*r[:,None,:]).reshape(len(v),-1)
        return np.column_stack((product,v*((x[:,2]-self.temp)/10)[:,None]))

    def predict(self,x):
        z = self.departure(x)
        return np.column_stack(((self.design(x,z)@self.coef).clip(-.1,1.1),z))


class RotorRelease(RegimeRelease):
    def __init__(self,original,torque,surface):
        super().__init__(original,torque)
        self.surface = surface

    def predict(self,x):
        return np.column_stack((super().predict(x),self.surface.predict(x)))


class RotorMaintenance:
    def __init__(self):
        self.old = RegimeMaintenance()
        self.decays = np.array([.94,.985,1.])
        self.gram = np.zeros((3,35,35))
        self.rhs = np.zeros((3,35,2))
        self.cost = np.full((6,7),.0004)

    def predict(self,x,base):
        out = self.old.predict(x,base[:,:7])
        p = out[CONTROL]
        curve,z = base[:,7:].T
        v = hats(x[:,0],np.array([1.,3.5,5.5,7.5,9.5,12.,15.]))
        r = hats(z,np.array([-3.,-1.,0.,1.,3.]))
        f = (v[:,:,None]*r[:,None,:]).reshape(len(x),35)
        coef = np.linalg.solve(self.gram+2*np.eye(35),self.rhs)
        absolute = np.column_stack((p,curve))
        bank = [p]
        for j,decay in enumerate(self.decays):
            adjusted = (absolute+f@coef[j]).clip(-.1,1.1)
            for i,title in enumerate(("原强分段","连续转速面")):
                name = f"{title}按转速回标记忆{decay:g}"
                out[name] = adjusted[:,i]
                bank.append(adjusted[:,i])
                for w in (.25,.5,.75):
                    out[f"原分段与{name}占{w:g}"] = p+w*(adjusted[:,i]-p)
        out["保持源期连续转速面"] = curve
        out[PAPER] = .5*(out[OLD_PAPER]+out["连续转速面按转速回标记忆0.985"])
        bank = np.column_stack(bank)
        zones = 2*np.digitize(x[:,0],[5,10])+(z>0)
        weight = 1/(self.cost[zones]+.00005)**2
        out[PRIMARY] = (bank*weight).sum(1)/weight.sum(1)
        out["同工具冻结转速校正等权"] = bank.mean(1)
        self.pending = f[-1],absolute[-1],bank[-1],zones[-1]
        return out

    def update(self,y):
        self.old.update(y)
        f,p,q,zone = self.pending
        self.gram = self.decays[:,None,None]*self.gram+np.outer(f,f)
        self.rhs = self.decays[:,None,None]*self.rhs+f[:,None]*(y-p)
        self.cost[zone] = .90*self.cost[zone]+.10*np.minimum((y-q)**2,.25)
