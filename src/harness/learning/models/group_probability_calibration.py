"""从合法训练组的留出预测学习概率映射，适用于带采集组信息的分类任务。"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class GroupProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """基础分类器与组间分数校准。

    groups只用于训练划分，不进入预测特征。target_mask可限定用于校准的
    合法装机样本；同组其他器件样本仍参与基础模型拟合。外层评价必须先
    排除其整组，再调用fit，不能传入包含外层评价标签的现成OOF概率。
    """
    def __init__(self, estimator=None, calibration_c=1., random_state=42):
        self.estimator=estimator
        self.calibration_c=calibration_c
        self.random_state=random_state

    def _new_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500,
                                                                 random_state=self.random_state))

    @staticmethod
    def _log_probability(probability):
        values=np.asarray(probability,dtype=np.float64)
        if values.ndim!=2 or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('基础分类器必须返回有限非负概率')
        np.testing.assert_allclose(values.sum(1),1.,rtol=1e-5,atol=1e-6)
        return np.log(values.clip(1e-8,1.))

    def fit(self, x, y, *, groups, target_mask=None):
        x=np.asarray(x);y=np.asarray(y);groups=np.asarray(groups)
        target=np.ones(len(y),dtype=bool) if target_mask is None else np.asarray(target_mask,dtype=bool)
        if x.ndim!=2 or y.shape!=(len(x),) or groups.shape!=y.shape or target.shape!=y.shape:
            raise ValueError('训练特征、标签、采集组及校准掩码必须逐行对应')
        if not np.isfinite(x).all() or not target.any() or len(np.unique(groups))<2:
            raise ValueError('组校准需要有限特征及至少两个合法训练组')
        self.classes_=np.unique(y)
        if len(self.classes_)<2 or not np.array_equal(np.unique(y[target]),self.classes_):
            raise ValueError('校准样本需覆盖基础分类器的全部类别')
        held=np.empty((len(y),len(self.classes_)),dtype=np.float64)
        for group in np.unique(groups):
            validation=groups==group;fitting=~validation
            if not np.array_equal(np.unique(y[fitting]),self.classes_):
                raise ValueError('整组留出后的源训练缺少类别，不能构造完整概率映射')
            queried=validation & target
            if not queried.any():
                continue
            model=self._new_estimator().fit(x[fitting],y[fitting])
            np.testing.assert_array_equal(model.classes_,self.classes_)
            held[queried]=model.predict_proba(x[queried])
        self.calibrator_=make_pipeline(StandardScaler(),LogisticRegression(C=self.calibration_c,
            max_iter=500,random_state=self.random_state)).fit(self._log_probability(held[target]),y[target])
        self.estimator_=self._new_estimator().fit(x,y)
        np.testing.assert_array_equal(self.estimator_.classes_,self.classes_)
        np.testing.assert_array_equal(self.calibrator_.classes_,self.classes_)
        self.n_features_in_=x.shape[1]
        self.calibration_rows_=int(target.sum())
        self.training_group_count_=len(np.unique(groups))
        return self

    def predict_proba(self, x):
        check_is_fitted(self,['estimator_','calibrator_'])
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('预测特征维数或有限性不符合训练约定')
        return self.calibrator_.predict_proba(self._log_probability(self.estimator_.predict_proba(x)))

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(1)]
