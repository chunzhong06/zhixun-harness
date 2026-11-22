"""源端按整次重复拟合核分数到实际物质的映射。推理仅接收目标行的一次完整暴露。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FIXED = "整周期留组类别映射"
PRIMARY = "估计浓度后决定类别映射"


def target_features(bl,tr,st):
    stage = []
    for a,count in zip((bl,tr,st),(1,5,5),strict=True):
        v=np.asarray(a,dtype=float)
        if v.shape!=(100*count,8,16) or not np.isfinite(v).all():
            raise ValueError("目标器件需提供完整三阶段、8种传感器和16个负载")
        stage.append(np.median(v.reshape(count,20,5,8,16)[:,:,:,:,7],axis=2))
    clean,early,late=stage
    stable=np.median(late,axis=0)
    delta=np.arcsinh((np.median(early[:2],axis=0)-stable)/np.maximum(abs(clean[0]),1e-4))
    return np.stack((stable.ravel(),(delta-delta.mean(0)).ravel()))


def kernel(x,y):
    return make_pipeline(StandardScaler(),SVC(C=4,gamma="scale",random_state=17)).fit(x,y)


class DirectCalibration:
    """安装时用同次配对测量拟合目标到源的校准，发放时只使用目标器件。"""
    def fit(self,source,target,y):
        self.old=StandardScaler().fit(source)
        self.new=StandardScaler().fit(target)
        x=self.old.transform(source)
        self.mapping=Ridge(alpha=1).fit(self.new.transform(target),x)
        self.classifier=kernel(x,y)
        return self

    def scores(self,target):
        return self.classifier.decision_function(self.mapping.predict(self.new.transform(target)))


class CalibratedCycleRelease:
    def fit(self,source,target,y,groups):
        x=self.steady(source)
        z=self.steady(target)
        held=np.zeros((len(y),3))
        for g in np.unique(groups):
            te=groups==g
            held[te]=DirectCalibration().fit(x[~te],z[~te],y[~te]).scores(z[te])
        self.decoder=LogisticRegression(C=1,max_iter=500).fit(held,y)
        self.paper=DirectCalibration().fit(x,z,y)
        return self

    @staticmethod
    def steady(features):
        return features[:,0].reshape(len(features),20,8).transpose(0,2,1).reshape(len(features),160)

    def predict_class(self,features):
        return self.decoder.predict(self.paper.scores(self.steady(features)))


class CycleRelease:
    def fit(self,source,target,y,groups,dose):
        if set(y)!={1,2,3} or len(set(groups))<2 or set(dose)!=set(range(2,8)):
            raise ValueError("装机资料缺少物质、浓度或独立重复")
        self.templates=[]
        held=np.empty((len(y),2,3))
        for col in (0,1):
            def fit(mask):
                x=target[mask,col]
                yy=y[mask]
                if col==0:
                    x=np.vstack((x,source[mask,col])); yy=np.tile(yy,2)
                return kernel(x,yy)
            for g in np.unique(groups):
                test=groups==g
                held[test,col]=fit(~test).decision_function(target[test,col])
            self.templates.append(fit(np.ones(len(y),dtype=bool)))
        self.global_models=[LogisticRegression(C=1,max_iter=500).fit(held[:,i],y) for i in (0,1)]
        self.joint=LogisticRegression(C=1,max_iter=500).fit(held.reshape(len(y),-1),y)
        self.local=[]
        for level in range(2,8):
            w=np.exp(-.65*(level-dose)**2)+.07
            self.local.append(LogisticRegression(C=1,max_iter=500).fit(held.reshape(len(y),-1),y,sample_weight=w/w.mean()))
        self.scale=StandardScaler().fit(target[:,0])
        self.reference=self.scale.transform(target[:,0])
        self.doses=dose.copy()
        return self

    def predict(self,features):
        scores=np.stack([m.decision_function(features[:,i]) for i,m in enumerate(self.templates)],axis=1)
        direct,trans=[m.predict_proba(scores[:,i]) for i,m in enumerate(self.global_models)]
        joint=self.joint.predict_proba(scores.reshape(len(features),-1))
        z=self.scale.transform(features[:,0])
        ds=cdist(z,self.reference,metric="sqeuclidean")/z.shape[1]
        closest=np.stack([ds[:,self.doses==level].min(1) for level in range(2,8)],axis=1)
        # 只用测得的目标行寻找现场浓度档；文件编号和真实浓度不进入决策器。
        level=closest.argmin(1)
        local=np.stack([m.predict_proba(scores.reshape(len(features),-1)) for m in self.local],axis=1)
        row=np.arange(len(features))
        prob={FIXED:direct,"瞬态留组类别映射":trans,"两读法固定共同映射":joint,"各浓度映射固定等权":local.mean(1)}
        for i in range(6):prob[f"固定用第{i+1}浓度映射"]=local[:,i]
        prob[PRIMARY]=(direct+local[row,level])/2
        prob["仅按估计浓度映射"]=local[row,level]
        return {name:1+v.argmax(1) for name,v in prob.items()},level+2
