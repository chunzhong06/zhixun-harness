"""按同一瓶标气的有效周期先后次序建模。每瓶仍只发一个浓度和一个分类。"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from harness.learning.models.year_gas_calibration import YearTools


VIEWS = ("全部周期均值","后两周期均值","后半减前半并读末期","按本次变化外推半个周期")
PRIMARY = "按本次变化与装机纠错收益发布"


def summarize(groups):
    means,late,drift,half,stability = [],[],[],[],[]
    for g in groups:
        if g.ndim!=2 or g.shape[1]!=1152 or len(g)<4 or not np.isfinite(g).all():
            raise ValueError("需要同一次暴露至少四个有序有效周期")
        first = g[:2].mean(0)
        last = g[-2:].mean(0)
        mean = g.mean(0)
        difference = last-first
        # 两个前后均值的中心间隔为len(g)-2，统一成每个物理周期的变化。
        step = difference/(len(g)-2)
        means.append(mean)
        late.append(last)
        drift.append(np.r_[last,step])
        half.append(last+step)
        stability.append(np.sqrt(np.mean(difference.reshape(4,-1)**2,axis=1)))
    return dict(zip(VIEWS,(np.stack(means),np.stack(late),np.stack(drift),np.stack(half)),strict=True)),np.stack(stability)


class CycleTools:
    def fit(self,bases,target,ty):
        self.models = {v:YearTools().fit(bases[v],target[v],ty) for v in VIEWS}
        return self

    def predict(self,x):
        names,values = [],[]
        for view,model in self.models.items():
            old = model.outputs(x[view])
            names.extend([view+"/目标回归",view+"/论文迁移",view+"/论文与校正各半"])
            values.extend((old[:,4],old[:,1],old[:,5]))
        return names,np.column_stack(values)


class SettlingPolicy:
    def fit(self,held,context,y,cut):
        self.cut = cut
        self.scale = StandardScaler().fit(np.log1p(context))
        self.z = self.scale.transform(np.log1p(context))
        self.sd = max(y.std(),1.)
        self.resid = y[:,None]-held
        self.cost = ((held>cut)!=(y[:,None]>cut)).astype(float)+.2*np.minimum((self.resid/self.sd)**2,4)
        self.bias_weights = Ridge(alpha=20).fit(self.z,self.resid)
        return self

    def predict(self,values,context):
        z = self.scale.transform(np.log1p(context))
        distance = np.mean((z[:,None]-self.z)**2,axis=2)
        near = np.exp(-distance)
        cost = (near@self.cost+4*self.cost.mean(0))/(near.sum(1)[:,None]+4)
        weights = np.exp(-9*(cost-cost.min(1)[:,None]))
        general = np.exp(-9*(self.cost.mean(0)-self.cost.mean(0).min()))
        bias = self.bias_weights.predict(z)
        return {PRIMARY:((values+.5*bias)*weights).sum(1)/weights.sum(1),
                "各版等权":values.mean(1),"周期试验统一选权":values@(general/general.sum()),
                "依本次变化合用未校正版":(values*weights).sum(1)/weights.sum(1)}
