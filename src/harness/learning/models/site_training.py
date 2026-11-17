"""用已揭晓的预测误差决定分类训练是否保留源样本，以及现场读数如何标准化。"""
from hashlib import sha256
from pathlib import Path
from time import process_time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness.learning.models.field_classification import fixed_snapshot, macro, model_scores, weights
from harness.learning.models.observer_errors import ObserverErrors


RECIPES=("pooled-source","pooled-separate","local-source","local-separate")
MODES=("recipe-ongoing","recipe-once","pooled-midway-cold",*RECIPES)


def implementation():
    return sha256(Path(__file__).read_bytes()).hexdigest()


class SiteTraining:
    def __init__(self,x,y,bank_names,classes,mode,state=None):
        if mode not in MODES:
            raise ValueError("未知现场训练配方")
        self.mode,self.classes=mode,classes
        self.source=np.asfortranarray(x,dtype=float)
        self.labels=np.asarray(y,dtype=int)
        if (len(self.source)!=len(self.labels) or not np.isfinite(self.source).all()
                or set(self.labels)!=set(range(1,classes+1))):
            raise ValueError("源样本需要完整类别及对应的有限输入")
        self.source_scale=StandardScaler().fit(self.source)
        self.source_z=self.source_scale.transform(self.source)
        self.source_weights=weights(self.labels,len(self.labels))
        saved=state or {}
        self.observer=ObserverErrors(bank_names,classes,saved.get("observer_counts"))
        self.history=saved.get("history",[])
        self.answers=saved.get("answers",[])
        self.active=saved.get("active",0)
        self.rounds=saved.get("rounds",0)
        self.loss=np.asarray(saved.get("loss",np.zeros((4,classes,classes))),dtype=int)
        if (not 0<=self.active<4 or len(self.history)!=len(self.answers) or self.loss.shape!=(4,classes,classes)
                or np.any(self.loss.sum(axis=(1,2))!=len(self.answers))
                or self.observer.counts[0].sum()!=len(self.answers)):
            raise ValueError("配方恢复状态与已记录反馈数不符")
        self.fits=0
        self.cpu=0.
        self._models=None

    def _fit(self,z,y,w,scale):
        start=process_time()
        model=LogisticRegression(C=1.,random_state=17,max_iter=1000).fit(z,y,sample_weight=w)
        self.cpu+=process_time()-start
        self.fits+=1
        return fixed_snapshot("mixed",(scale,model))

    def models(self):
        if self._models is not None:
            return self._models
        if not self.history:
            initial=self._fit(self.source_z,self.labels,self.source_weights,self.source_scale)
            self._models=dict.fromkeys(RECIPES,initial)
            return self._models
        x=np.asfortranarray(self.history)
        y=np.asarray(self.answers,dtype=int)
        local_scale=StandardScaler().fit(x)
        w=weights(y,len(self.labels))
        eligible=len(np.unique(y))>=2
        fallback=None if eligible else self._fit(self.source_z,self.labels,self.source_weights,self.source_scale)
        models={}
        for suffix,scale in (("source",self.source_scale),("separate",local_scale)):
            z=scale.transform(x)
            models["pooled-"+suffix]=self._fit(np.vstack((self.source_z,z)),np.r_[self.labels,y],
                                               np.r_[self.source_weights,w],scale)
            models["local-"+suffix]=self._fit(z,y,2*w,scale) if eligible else fallback
        self._models=models
        return models

    def current(self):
        if self.mode=="pooled-midway-cold":
            # 首次回标前保留推荐投票；累计五次回标后使用现场标准化。
            if not self.history:return "observer"
            return "pooled-source" if self.rounds<5 else "pooled-separate"
        return self.mode if self.mode in RECIPES else RECIPES[self.active]

    def snapshot(self):
        if self.current()=="observer":return None
        return self.models()[self.current()]

    def predict(self,x,votes):
        result={name:model_scores(record,x,self.classes).argmax(axis=1)+1 for name,record in self.models().items()}
        if self.mode=="pooled-midway-cold":result["observer"]=self.observer.predict(votes,"uniform")
        return result

    def update(self,x,votes,correct,before):
        y=np.asarray(correct,dtype=int)
        if not len(y):
            return
        if len(x)!=len(y) or any(len(before[a])!=len(y) for a in RECIPES):
            raise ValueError("回标需对应本批预测")
        self.observer.update(y,votes)
        for j,name in enumerate(RECIPES):
            np.add.at(self.loss[j],(y-1,np.asarray(before[name])-1),1)
        if self.mode=="recipe-ongoing" or (self.mode=="recipe-once" and not self.rounds):
            scores=[macro(m) for m in self.loss]
            best=int(np.argmax(scores))
            if scores[best]>=scores[self.active]+.01:
                self.active=best
        self.history.extend(np.asarray(x,dtype=float).tolist())
        self.answers.extend(y.tolist())
        self.rounds+=1
        self._models=None

    def state(self):
        return {"history":self.history,"answers":self.answers,"observer_counts":self.observer.counts.tolist(),
                "loss":self.loss.tolist(),"rounds":self.rounds,"active":self.active}
