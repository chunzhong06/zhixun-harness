"""选回标的模型与发布模型分别持有尺度；发布模型只看以往已交付的普通读数。"""
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.core.contracts import digest
from harness.methods import feedback_sampling
from harness.models import site_training
from harness.learning.models.field_classification import model_scores, weights
from harness.learning.models.probability_fusion import observer_probabilities

MODE = "boundary-served-scale"


def implementation():
    return digest([sha256(Path(module.__file__).read_bytes()).hexdigest()
                   for module in (site_training,feedback_sampling)] + [sha256(Path(__file__).read_bytes()).hexdigest()])


class ServedScaleGas(site_training.SiteTraining):
    def __init__(self, x, y, bank_names, classes, mode, state=None):
        if mode != MODE:
            raise ValueError("未知服务尺度分类方式")
        super().__init__(x,y,bank_names,classes,"pooled-midway-cold",state)
        self.served = (state or {}).get("served_readings",[])
        self._published = None

    def current(self):
        return "served-scale" if self.rounds >= 5 else super().current()

    def snapshot(self):
        if self.rounds < 5:
            return super().snapshot()
        if self._published is None:
            if not self.served or not self.history:
                raise ValueError("新分类器缺少已经交付的输入或已回标训练行")
            scale = StandardScaler().fit(np.asfortranarray(self.served))
            y = np.asarray(self.answers)
            x = np.vstack((self.source_z,scale.transform(np.asfortranarray(self.history))))
            self._published = self._fit(x,np.r_[self.labels,y],
                                       np.r_[self.source_weights,weights(y,len(self.labels))],scale)
        return self._published

    def predict(self, x, votes):
        predictions = super().predict(x,votes)
        if self.rounds >= 5:
            predictions["served-scale"] = model_scores(self.snapshot(),x,self.classes).argmax(axis=1)+1
        return predictions

    def select(self, keys, x, votes, budget, seed):
        # 查询模型保持先前已实测的读法。发布模型的普通读数不改变它已承诺的查询顺序。
        old = super().current()
        p = (observer_probabilities(self.observer,votes) if old == "observer"
             else model_scores(self.models()[old],x,self.classes))
        ix,_ = feedback_sampling.select_feedback(keys,p,budget,seed,"边界",np.zeros((self.classes,2),dtype=int))
        return [keys[i] for i in ix]

    def observe_served(self, x):
        self.served.extend(np.asarray(x,dtype=float).tolist())
        self._published = None

    def state(self):
        return {**super().state(),"served_readings":self.served}
