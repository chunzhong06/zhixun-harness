"""按两阶段预测分歧对寿命的实际代价，学习现场应保留多少正常阶段预测。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PRIMARY="按弄错退化阶段的寿命代价分配权重"
LEVEL="只用原寿命估计决定阶段分量"
CONSTANT="按历史损失选择一个固定阶段分量"


class StageRisk:
    def fit(self,bank,ctx,y,sites):
        anchor,h,d,p=bank.T
        delta=h-d
        weight=np.array([len(y)/(len(np.unique(sites))*np.sum(sites==s)) for s in sites])
        self.scale=StandardScaler().fit(ctx,sample_weight=weight)
        z=self.scale.transform(ctx).clip(-4,4)
        # 学习从合法的现场阶段概率偏移多少，直接回归最终预测对真实寿命的误差。
        base=(anchor+d+p*delta)/2
        self.linear=Ridge(alpha=len(y)*3,fit_intercept=False)
        self.linear.fit(.5*delta[:,None]*np.column_stack((np.ones(len(y)),z)),y-base,sample_weight=weight)
        denom=np.sum(weight*delta**2)
        self.fixed=np.clip(np.sum(weight*delta*(2*y-anchor-d))/max(denom,1e-8),0,1)
        target=(2*y-anchor-d)*delta/(delta**2+4.)
        self.level=ExtraTreesRegressor(n_estimators=96,max_depth=3,min_samples_leaf=35,random_state=17,n_jobs=1)
        self.level.fit(anchor[:,None],target,sample_weight=weight*(delta**2+4.))
        return self

    def predict(self,bank,ctx):
        anchor,h,d,p=bank.T
        z=self.scale.transform(ctx).clip(-4,4)
        corrected=np.clip(p+self.linear.predict(np.column_stack((np.ones(len(z)),z))),0,1)
        level=self.level.predict(anchor[:,None]).clip(0,1)
        project=lambda w:np.maximum(0,(anchor+d+w*(h-d))/2)
        return {PRIMARY:project(corrected),LEVEL:project(level),CONSTANT:project(self.fixed),
                "代价校准与原阶段概率各半":project((corrected+p)/2)}
