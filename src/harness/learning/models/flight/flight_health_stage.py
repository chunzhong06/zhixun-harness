"""由源端的异常退化起点训练两个寿命读法；预测健康状态只接收单航班传感器。"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from harness.learning.models.flight.flight_condition_response import ConditionReadout
from harness.learning.models.flight.flight_masked_readings import combine

PRIMARY="按本次航班估计的退化阶段合用训练模型"
NORMAL="固定使用正常阶段训练模型"
DEGRADED="固定使用异常阶段训练模型"
POOLED="令所有航班共用一个模型"
CONSTANT="两阶段模型固定等权"


class HealthStage:
    def fit(self,x,y,healthy):
        self.readers=[ConditionReadout().fit(x[mask],y[mask]) for mask in (healthy,~healthy)]
        self.classifier=ExtraTreesClassifier(n_estimators=96,min_samples_leaf=6,random_state=17,n_jobs=1)
        self.classifier.fit(x.reshape(len(x),-1),healthy)
        self.prevalence=float(healthy.mean())
        return self

    def predict(self,x,anchor):
        healthy=self.classifier.predict_proba(x.reshape(len(x),-1))[:,1]
        first,last=[combine(*reader.readings(x)) for reader in self.readers]
        soft=healthy*first+(1-healthy)*last
        options={NORMAL:first,DEGRADED:last,POOLED:anchor,CONSTANT:(first+last)/2,
                 "按训练阶段比例固定相加":self.prevalence*first+(1-self.prevalence)*last,
                 "直接按现场阶段比例相加":soft,PRIMARY:(anchor+soft)/2}
        for p in (.25,.5,.75):
            options[f"全局模型与固定阶段配比{p:g}各半"]=(anchor+p*first+(1-p)*last)/2
        return options,healthy
