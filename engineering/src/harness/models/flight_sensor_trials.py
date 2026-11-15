"""用完整历史机队的配对误差直接学习是否修正当前航班；各控制模型共享这批证据。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PRIMARY="依实际改判收益与共同误差修正寿命"
SWITCH="只学习联合读法的净改进"
FIXED_BIAS="固定寿命分段校准"
MASK_BIAS="仅依还原偏差校准"
ANCHOR="邻域原读数分歧加权"
JOINT="原读数与还原偏差共同训练"
ROBUST="仅用跨传感器还原偏差"


def design(bank):
    anchor,joint,robust=bank.T
    return np.column_stack((anchor,joint-anchor,robust-anchor))


class TrialCorrection:
    def fit(self,bank,ctx,y,sites):
        sw=np.array([len(y)/(len(np.unique(sites))*np.sum(sites==s)) for s in sites])
        anchor=bank[:,0]
        gap=bank[:,1]-anchor
        self.center=StandardScaler().fit(np.column_stack((design(bank),ctx)),sample_weight=sw)
        x=self.center.transform(np.column_stack((design(bank),ctx))).clip(-3,3)
        # 直接拟合把两读法互换产生的寿命变化，差距很小时样本对权重影响自然很小。
        self.mix=Ridge(alpha=.18*len(y),fit_intercept=False)
        self.mix.fit(gap[:,None]*np.column_stack((np.ones(len(x)),x)),y-anchor,sample_weight=sw)
        # 对各工具共有的高估或低估另设同信息控制，按源端预测寿命分段学习。
        self.corrections={}
        for name,features in ((FIXED_BIAS,anchor[:,None]),(MASK_BIAS,ctx),(PRIMARY,np.column_stack((design(bank),ctx)))):
            est=ExtraTreesRegressor(n_estimators=80,max_depth=3,min_samples_leaf=36,random_state=17,n_jobs=1)
            self.corrections[name]=est.fit(features,y-anchor,sample_weight=sw)
        return self

    def predict(self,bank,ctx):
        anchor=bank[:,0]
        x=self.center.transform(np.column_stack((design(bank),ctx))).clip(-3,3)
        w=np.clip(self.mix.predict(np.column_stack((np.ones(len(x)),x))),0,1)
        switched=anchor+w*(bank[:,1]-anchor)
        anchor_bias=self.corrections[FIXED_BIAS].predict(anchor[:,None])
        mask_bias=self.corrections[MASK_BIAS].predict(ctx)
        full_bias=self.corrections[PRIMARY].predict(np.column_stack((design(bank),ctx)))
        return {SWITCH:switched, FIXED_BIAS:(anchor+anchor_bias).clip(0),MASK_BIAS:(anchor+mask_bias).clip(0),
                "固定校准与原版各半":(anchor+.5*anchor_bias).clip(0),
                "全部诊断固定校准":(anchor+full_bias).clip(0),
                PRIMARY:(switched+.5*full_bias).clip(0)}
