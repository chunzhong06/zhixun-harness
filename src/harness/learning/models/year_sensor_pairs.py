"""以共同配方配对设备，分别校正同一秒的偏移、尺度及同一敏感层的形状。"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TOOLS = ("逐秒同配方平均偏移","逐秒同配方尺度校正","四敏感层各自映射回源","全曲线共同映射回源","源端跨设备残差纠偏")
PRIMARY = "按留出标气跨设备重建误差与分类得失合用读法"


class PairedYears:
    def fit(self,base,source,target,y):
        self.base = base
        self.shift = (source-target).mean(0)
        self.src = StandardScaler().fit(source)
        self.tgt = StandardScaler().fit(target)
        a = self.src.transform(source)
        b = self.tgt.transform(target)
        self.ratio = np.clip(np.mean(a*b,axis=0),0,1)
        self.maps = [Ridge(alpha=10).fit(t,s) for t,s in zip(np.split(b,4,axis=1),np.split(a,4,axis=1),strict=True)]
        self.full = Ridge(alpha=10).fit(b,a)
        self.error = Ridge(alpha=10).fit(b,y-base.predict(source))
        self.ref = b
        return self

    def predict(self,x):
        b = self.tgt.transform(x)
        per_layer = np.column_stack([m.predict(v) for m,v in zip(self.maps,np.split(b,4,axis=1),strict=True)])
        shapes = (x+self.shift, self.src.inverse_transform(b*self.ratio), self.src.inverse_transform(per_layer),
                  self.src.inverse_transform(self.full.predict(b)))
        forecasts = [self.base.predict(v) for v in shapes]
        forecasts.append(forecasts[2]+self.error.predict(b))
        # 本机合法读数到装机已测配方的最近距离和读法之间的不一致用于判断选用。
        norms = np.sum(b*b,axis=1)[:,None]+np.sum(self.ref*self.ref,axis=1)[None]-2*(b@self.ref.T)
        nearest = np.sqrt(np.maximum(0,norms.min(1))/b.shape[1])
        return np.column_stack(forecasts), nearest


class PairedChoice:
    def fit(self,table,dist,label,cut):
        self.cut = cut
        self.sd = max(label.std(),1.)
        self.context = self.info(table,dist)
        residual = table-label[:,None]
        self.cost = ((table>cut)!=(label[:,None]>cut)) + .25*np.minimum((residual/self.sd)**2,4)
        self.general = self.cost.mean(0)
        return self

    def info(self,table,dist):
        return np.column_stack((dist,np.median(table,axis=1)/self.sd,table.std(1)/self.sd,
                                (table[:,0]-table[:,-1])/self.sd))

    def predict(self,table,dist):
        scale = np.maximum(self.context.std(0),(.15,.15,.12,.12))
        distance = np.mean(((self.info(table,dist)[:,None]-self.context[None])/scale)**2,axis=2)
        near = np.exp(-.5*distance)
        costs = (near@self.cost+4*self.general)/(near.sum(1)[:,None]+4)
        w = np.exp(-10*(costs-costs.min(1)[:,None]))
        g = np.exp(-10*(self.general-self.general.min()))
        return {PRIMARY:(w*table).sum(1)/w.sum(1),"统一装机验证配对权重":table@(g/g.sum()),
                "配对候选固定平均":table.mean(1)}
