"""风场功率估计。源模型与回标维护共用当时可知的风、转速与温度。"""
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

NAMES = ("半米风速分箱","论文五参数功率曲线","论文多变量回归树集成","三变量提升树","风与转速物理回归")
PRIMARY = "按本机回标与运行区间发布功率"
GLOBAL = "同样工具按全机累计误差分配"


def logistic(speed, coefficients):
    low, high, midpoint, slope, shape = coefficients
    return low+(high-low)/(1+(midpoint/np.maximum(speed,.05))**slope)**shape


def physics(x):
    v, r, temp = x.T
    a = v/12
    b = r/15
    return np.column_stack((np.ones(len(x)), a,b,a*a,b*b,a**3,b**3,a*b,
                            a**3*(temp-10)/100,np.maximum(0.,a-.8),np.maximum(0.,b-.85)))


class WindRelease:
    def fit(self,x,y):
        vel = x[:,0]
        bins = np.floor(vel*2).astype(int)
        self.centers = np.unique(bins)
        self.values = np.array([np.mean(y[bins==b]) for b in self.centers])
        self.coef = least_squares(lambda c:logistic(vel,c)-y, [0,1,9,6,1],
            bounds=([-.1,.65,2,1,.1],[.1,1.2,20,18,5]),max_nfev=100).x
        self.tree = RandomForestRegressor(n_estimators=50,min_samples_leaf=12,max_features=1.,
                                         n_jobs=1,random_state=7).fit(x,y)
        self.boost = HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=23,min_samples_leaf=26,
                           early_stopping=False,random_state=7).fit(x,y)
        # 不用目标场功率或跨场额定风速估值做归一化。物理回归统一在源端训练。
        z = physics(x)
        self.ridge = np.linalg.solve(z.T@z+.03*np.eye(z.shape[1]),z.T@y)
        return self

    def predict(self,x):
        v = x[:,0]
        return np.column_stack((np.interp(2*v-.5,self.centers,self.values),logistic(v,self.coef),
                        self.tree.predict(x),self.boost.predict(x),physics(x)@self.ridge)).clip(-.1,1.1)


class WindMaintenance:
    def __init__(self):
        self.decays = np.array([.85,.97,1.])
        self.g = np.zeros((3,len(NAMES),2,2))
        self.rhs = np.zeros((3,len(NAMES),2))
        count = 4*len(NAMES)
        self.cost = np.full(count,.0025)
        self.regime_cost = np.full((3,count),.0025)
        self.regime_mass = np.zeros(3)
        self.last = None

    def predict(self,x,base):
        # 严格以末窗之前已经归还的功率更新状态；当前槽内多个输入仅做逐行独立推理。
        f = np.stack((np.ones_like(base), base-.4),axis=-1)
        penalty = np.diag([3.,1.])
        coefficients = np.linalg.solve(self.g+penalty, self.rhs[:,:,:,None])[:,:,:,0]
        bank = [base]
        for coef in coefficients:
            bank.append((base+np.einsum("nmi,mi->nm",f,coef)).clip(-.1,1.1))
        matrix = np.column_stack(bank)
        out = {name+suffix:values[:,i] for suffix,values in zip(("源训练","回标短记忆","回标长记忆","回标累计"),bank,strict=True)
               for i,name in enumerate(NAMES)}
        zones = np.digitize(x[:,0],[5.,10.])
        local = (self.regime_mass[zones,None]*self.regime_cost[zones]+12*self.cost)/(12+self.regime_mass[zones,None])
        global_weights = 1/(self.cost+.00015)**2
        local_weights = 1/(local+.00015)**2
        out[GLOBAL] = matrix@(global_weights/global_weights.sum())
        out[PRIMARY] = (matrix*local_weights).sum(1)/local_weights.sum(1)
        out["叶轮与风速工具固定等权"] = matrix.mean(1)
        out["两种论文与提升树逐版均分"] = bank[2][:,1:4].mean(1)
        # 同样的区间信息给事先写定的分段读法。
        out["固定低风树中风提升高风曲线"] = np.choose(zones,[bank[2][:,2],bank[2][:,3],bank[2][:,1]])
        self.last = f[-1],base[-1],matrix[-1],zones[-1]
        return out

    def update(self,y):
        f,p,q,zone = self.last
        self.g = self.decays[:,None,None,None]*self.g+np.einsum("mi,mj->mij",f,f)
        self.rhs = self.decays[:,None,None]*self.rhs+(y-p)[:,None]*f
        loss = np.minimum((q-y)**2,.25)
        self.cost = .95*self.cost+.05*loss
        self.regime_cost[zone] = .9*self.regime_cost[zone]+.1*loss
        self.regime_mass[zone] = .96*self.regime_mass[zone]+1.
