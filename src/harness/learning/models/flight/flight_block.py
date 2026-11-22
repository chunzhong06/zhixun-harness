"""在同一航班的原始采样间隔内比较不同读取起点与均值降采样。"""
import numpy as np
from scipy.spatial.distance import cdist
from harness.learning.models.flight.flight_condition_response import ConditionReadout
from harness.learning.models.flight.flight_masked_readings import combine

MEAN="每十六个原始时刻取均值再读部件"
AUGMENT="多取样起点共同训练后固定平均"
PRIMARY="按本航班四种取样起点的复测分散度发布"


def block_mean(rows):
    if len(rows)<16:raise ValueError("原始航班不足一个采样块")
    bounds=np.arange(0,len(rows),16)
    return np.add.reduceat(rows,bounds,axis=0)/np.diff(np.r_[bounds,len(rows)])[:,None]


def samples(rows):
    phases=[rows[offset::16] for offset in (0,4,8,12)]
    return [*phases,block_mean(rows)]


def component_view(extractor,rows):
    z,measure=extractor.measurements(rows)
    ix=np.argsort(cdist(extractor.points,z),axis=1)[:,:max(2,int(np.ceil(len(z)/8)))]
    field=measure[:,np.r_[0:14,26:38]]
    lower,mid,upper=np.percentile(field[ix],[25,50,75],axis=1)
    return np.column_stack((mid,upper-lower,extractor.points))


class BlockReaders:
    def fit(self,x,y,anchor):
        self.original=anchor
        self.mean=ConditionReadout().fit(x[:,4],y)
        self.replicas=ConditionReadout().fit(x[:,:4].reshape(-1,*x.shape[2:]),np.repeat(y,4))
        return self

    @staticmethod
    def predictions(reader,x):
        n=x.shape[0]
        val,sd=reader.readings(x.reshape(-1,*x.shape[2:]))
        return combine(val,sd).reshape(n,x.shape[1])

    def predict(self,x):
        old=self.predictions(self.original,x)
        replicas=self.predictions(self.replicas,x)
        mean=self.predictions(self.mean,x)
        out={"第一个取样起点":old[:,0],"原模型在四个起点等权":old[:,:4].mean(1),"原模型直接用块均值":old[:,4],
             MEAN:mean[:,4],AUGMENT:replicas[:,:4].mean(1)}
        candidates=np.column_stack((out["原模型在四个起点等权"],out[AUGMENT],out[MEAN]))
        spread=np.stack((old[:,:4].std(1),replicas[:,:4].std(1),mean[:,:4].std(1)),axis=1)
        w=(.3+spread)**-2
        out[PRIMARY]=(candidates*w).sum(1)/w.sum(1)
        out["同三工具固定平均"]=candidates.mean(1)
        out["联合部件与块均值固定各半"]=(out[MEAN]+old[:,0])/2
        return out
