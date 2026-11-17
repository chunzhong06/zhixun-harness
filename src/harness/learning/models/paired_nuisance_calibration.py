"""配对器件校准与训练干扰水平参照；不把未知查询的干扰标签当作输入。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


def _kernel(x,y):
    return make_pipeline(StandardScaler(),SVC(C=4,gamma='scale',random_state=17)).fit(x,y)


def _scores(model,x):
    values=np.asarray(model.decision_function(x))
    return np.column_stack((-values,values)) if values.ndim==1 else values


class PairedNuisanceClassifier(ClassifierMixin,BaseEstimator):
    """将已有直接标准化、干扰共有响应及类别原型共享给不同表示。

    fit要求合法源阶段的配对特征、类别、采集组和离散干扰水平。
    predict_proba只接当前目标器件特征；干扰水平由源样本邻近关系估计。
    每个训练组从两级模型的内层交叉预测中完整移出。外层评价由调用方
    先排除整组；同一暴露的不同周期必须保持相同group。
    """
    def __init__(self,ridge_alpha=1.,decoder_c=1.):
        self.ridge_alpha=ridge_alpha
        self.decoder_c=decoder_c

    def _fit_bank(self,source,target,y,nuisance):
        old=StandardScaler().fit(source);new=StandardScaler().fit(target)
        a=old.transform(source);b=new.transform(target)
        # 线性核的对偶形式与中心化Ridge等价，避免宽嵌入形成D×D系数矩阵。
        mapping=KernelRidge(alpha=self.ridge_alpha,kernel='linear').fit(b,a)
        direct=_kernel(a,y)
        grid=np.unique(nuisance)
        centers=np.stack([b[nuisance==level].mean(0) for level in grid])
        prototypes=np.stack([[b[(nuisance==level)&(y==label)].mean(0)
                              for label in self.classes_] for level in grid])
        if not np.isfinite(prototypes).all():
            raise ValueError('每个训练干扰水平需覆盖全部类别')
        residual=_kernel(b-centers[np.searchsorted(grid,nuisance)],y)
        return {'old':old,'new':new,'mapping':mapping,'direct':direct,'grid':grid,
                'centers':centers,'prototypes':prototypes,'residual':residual,
                'reference':b,'nuisance':nuisance.copy()}

    @staticmethod
    def _bank_scores(bank,target):
        z=bank['new'].transform(target)
        direct=_scores(bank['direct'],bank['mapping'].predict(z))
        distances=cdist(z,bank['reference'],metric='sqeuclidean')
        closest=np.stack([distances[:,bank['nuisance']==level].min(1) for level in bank['grid']],axis=1)
        level=closest.argmin(1)
        residual=_scores(bank['residual'],z-bank['centers'][level])
        template=-np.mean((z[:,None,:]-bank['prototypes'][level])**2,axis=2)
        template-=template.mean(1,keepdims=True)
        return (direct,residual,template)

    def fit(self,source,target,y,*,groups,nuisance):
        source,target=np.asarray(source,dtype=np.float64),np.asarray(target,dtype=np.float64)
        y,groups,nuisance=map(np.asarray,(y,groups,nuisance))
        if source.ndim!=2 or source.shape!=target.shape or any(v.shape!=(len(source),) for v in (y,groups,nuisance)):
            raise ValueError('配对特征、标签、组和干扰水平必须逐行一致')
        if not np.isfinite(source).all() or not np.isfinite(target).all() or len(np.unique(groups))<2:
            raise ValueError('配对校准需要有限特征和至少两个源采集组')
        self.classes_=np.unique(y)
        if len(self.classes_)<2:raise ValueError('分类至少需要两个类别')
        held=[np.empty((len(y),len(self.classes_))) for _ in range(3)]
        for group in np.unique(groups):
            test=groups==group;fit=~test
            if not np.array_equal(np.unique(y[fit]),self.classes_):raise ValueError('源留组后缺少类别')
            bank=self._fit_bank(source[fit],target[fit],y[fit],nuisance[fit])
            for destination,values in zip(held,self._bank_scores(bank,target[test]),strict=True):
                destination[test]=values
        self.decoders_=[LogisticRegression(C=self.decoder_c,max_iter=500).fit(values,y) for values in held]
        self.bank_=self._fit_bank(source,target,y,nuisance)
        self.n_features_in_=source.shape[1]
        return self

    def predict_proba(self,target):
        check_is_fitted(self,['bank_','decoders_'])
        target=np.asarray(target,dtype=np.float64)
        if target.ndim!=2 or target.shape[1]!=self.n_features_in_ or not np.isfinite(target).all():
            raise ValueError('目标器件特征的形状或有限性不符')
        return np.mean([decoder.predict_proba(values) for decoder,values in
                        zip(self.decoders_,self._bank_scores(self.bank_,target),strict=True)],axis=0)

    def predict(self,target):
        return self.classes_[self.predict_proba(target).argmax(1)]
