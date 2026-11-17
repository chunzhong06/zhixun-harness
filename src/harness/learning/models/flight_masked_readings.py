"""用其他传感器还原遮住的整路测量。安装时实际试验原读数和还原误差各能带来多少寿命信息。"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from harness.learning.models.flight_condition_response import ConditionReadout

ANCHOR = "邻域原读数分歧加权"
RESIDUAL = "仅用跨传感器还原偏差"
JOINT = "原读数与还原偏差共同训练"
PRIMARY = "按遮路试验估计各读法误差"
TOOLS = (ANCHOR,RESIDUAL,JOINT)


class SensorRecovery:
    """同一工况的14个监督任务，输入去掉该传感器的位置和波动两项。"""
    def fit(self,x):
        self.conditions=[]
        for point in x.transpose(1,0,2):
            scale=StandardScaler().fit(point[:,:28])
            z=scale.transform(point[:,:28])
            regressors=[]
            for i in range(14):
                out=np.array([i,i+14])
                other=np.delete(np.arange(28),out)
                reg=Ridge(alpha=.015*len(z)).fit(z[:,other],z[:,out])
                regressors.append((out,other,reg))
            self.conditions.append((scale,regressors))
        return self

    def errors(self,x):
        errors=[]
        for point,(scale,regs) in zip(x.transpose(1,0,2),self.conditions,strict=True):
            z=scale.transform(point[:,:28])
            res=np.empty_like(z)
            for out,other,reg in regs:
                res[:,out]=z[:,out]-reg.predict(z[:,other])
            errors.append(res)
        return np.stack(errors,axis=1)


def combine(values,spread):
    weights=(1+spread)**-2
    return np.sum(weights*values,axis=1)/weights.sum(1)


class MaskedReadouts:
    def fit(self,x,y):
        self.recovery=SensorRecovery().fit(x)
        residual=self.recovery.errors(x)
        self.readers=[ConditionReadout().fit(a,y) for a in self.views(x,residual)]
        return self

    @staticmethod
    def views(x,residual):
        return (x,np.concatenate((residual,x[:,:,28:]),axis=2),np.concatenate((x,residual),axis=2))

    def readings(self,x):
        residual=self.recovery.errors(x)
        predictions=[combine(*model.readings(v)) for v,model in zip(self.views(x,residual),self.readers,strict=True)]
        # 每路在整趟航班六个工况上的偏差都保留，寿命答案不参与现场判断。
        context=np.column_stack((np.abs(residual).mean(1),np.sqrt((residual**2).mean(axis=(1,2)))))
        return np.column_stack(predictions),context


class MaskedTrialPolicy:
    def fit(self,pred,context,y,sites):
        weight=np.array([len(y)/(len(set(sites))*sum(sites==s)) for s in sites])
        self.model=ExtraTreesRegressor(n_estimators=64,max_depth=4,min_samples_leaf=30,random_state=17,n_jobs=1)
        self.model.fit(context,np.abs(pred-y[:,None]),sample_weight=weight)
        self.mean=np.average(self.weights(context),axis=0,weights=weight)
        return self

    def weights(self,context):
        error=np.maximum(self.model.predict(context),1)
        w=error**-2
        return w/w.sum(1,keepdims=True)
