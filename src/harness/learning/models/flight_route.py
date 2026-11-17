"""从单航班持续观测长度与工况分布判断用哪组源机训练出的寿命工具。"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from harness.learning.models.flight_component import ComponentReadings
from harness.learning.models.flight_condition_response import ConditionReadout
from harness.learning.models.flight_masked_readings import combine

PRIMARY="按当前飞行工况试选源航线训练模型"
JOINT="工况描述和部件读数共同训练"


def route_features(rows):
    """只用当前实际输入的采样数量与工况分布。采样数量在这里不换算为小时。"""
    w=rows[:,14:]
    return np.r_[np.log1p(len(rows)),np.quantile(w,[.1,.5,.9],axis=0).ravel(),w.std(0)]


class RouteReadout:
    def fit(self,x,info,y,route):
        self.route=ExtraTreesClassifier(n_estimators=64,max_depth=5,min_samples_leaf=8,random_state=17,n_jobs=1).fit(info,route)
        self.model=ConditionReadout().fit(self.augmented(x,info),y)
        self.experts=[ConditionReadout().fit(x[route==r],y[route==r]) for r in self.route.classes_]
        self.prevalence=np.array([np.mean(route==r) for r in self.route.classes_])
        return self

    @staticmethod
    def augmented(x,info):
        return np.concatenate((x,np.broadcast_to(info[:,None,:],(len(info),x.shape[1],info.shape[1]))),axis=2)

    def predict(self,x,info):
        joint=combine(*self.model.readings(self.augmented(x,info)))
        probs=self.route.predict_proba(info)
        experts=np.column_stack([combine(*reader.readings(x)) for reader in self.experts])
        local=(experts*probs).sum(1)
        out={JOINT:joint,PRIMARY:(joint+local)/2,"仅按估计航线选工具":local,"三航线固定均值":experts.mean(1),
             "固定训练航线比例":experts@self.prevalence,"全局与平均航线各半":(joint+experts.mean(1))/2}
        for r,v in zip(self.route.classes_,experts.T,strict=True):
            out[f"固定航线{r}"]=v
            out[f"全局与航线{r}固定各半"]=(joint+v)/2
        return out
