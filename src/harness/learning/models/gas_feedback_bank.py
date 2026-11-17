"""随到随预测的气体辅助分类器。拟合阶段只使用官方 fit 行，服务不更新特征统计。"""
import numpy as np
from sklearn.svm import SVC

from harness.learning.inference.gas_views import represent


def fit_discriminant(bank,y):
    center=bank.mean(axis=0)
    scale=np.maximum(bank.std(axis=0),1e-9)
    z=(bank-center)/scale
    classes=np.unique(y)
    means=np.stack([z[y==label].mean(0) for label in classes])
    delta=z-means[np.searchsorted(classes,y)]
    cov=delta.T@delta/(len(y)-len(classes))
    cov=.75*cov+.25*np.diag(np.diag(cov))+1e-6*np.eye(cov.shape[0])
    weights=np.linalg.solve(cov,means.T)
    return center,scale,weights,-.5*np.sum(means*weights.T,axis=1),classes


def discriminate(fitted,x):
    center,scale,w,b,classes=fitted
    return classes[np.argmax(((x-center)/scale)@w+b,axis=1)]


class GasFeedbackBank:
    names=("current","type-2","type-4","relative","shape-full")
    groups={"type-2":(2,3,10,11),"type-4":(6,7,14,15)}

    def __init__(self,training):
        x=np.asarray([r.values for r in training])
        y=np.asarray([r.label for r in training])
        if x.shape!=(len(y),128) or set(y)!=set(range(1,7)) or any(r.role!="fit" for r in training):
            raise ValueError("分类参照需要官方六类气体训练行")
        self.scale=np.maximum(np.median(np.abs(x),axis=0),1e-9)
        self.models={}
        log=self.log_measurements(x)
        for name,channels in self.groups.items():
            self.models[name]=fit_discriminant(log[:,channels,:].reshape(len(x),32),y)
        shape=represent(x,"shape",None,None)
        self.models["relative"]=fit_discriminant(shape,y)
        self.svm=SVC(C=1.,gamma="scale",class_weight="balanced").fit(shape,y)

    def log_measurements(self,x):
        return (np.sign(x)*np.log1p(np.abs(x)/self.scale)).reshape(-1,16,8)

    def predict(self,current,x):
        x=np.asarray(x)
        shape=represent(x,"shape",None,None)
        out={"current":current.predict(x).argmax(axis=1)+1}
        log=self.log_measurements(x)
        for name,channels in self.groups.items():
            out[name]=discriminate(self.models[name],log[:,channels,:].reshape(len(x),32))
        out["relative"]=discriminate(self.models["relative"],shape)
        out["shape-full"]=self.svm.predict(shape)
        return out
