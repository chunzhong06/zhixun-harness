"""给现有表格式研究执行器提供文件型测量输入，不把样本标识作为预测特征。"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .group_probability_calibration import GroupProbabilityClassifier
from .paired_nuisance_calibration import PairedNuisanceClassifier


@lru_cache(maxsize=4)
def _archive(path):
    """同一只读资产在进程内只解压一次，缓存不随候选权重反复保存。"""
    with np.load(path,allow_pickle=False) as saved:
        keys=saved['keys'].astype(str).tolist()
        if len(set(keys))!=len(keys):raise ValueError('测量标识重复')
        arrays={name:saved[name].copy() for name in saved.files if name!='keys'}
    if any(name in arrays for name in ('y','target','labels','nuisance','groups')):
        raise ValueError('公开测量资产不能混入标签或训练权限元数据')
    for value in arrays.values():value.flags.writeable=False
    return {key:i for i,key in enumerate(keys)},arrays


def load_reference_view(frame,view,*,paired=False):
    """公开给双方研究者的测量读取接口；query资产只含目标器件。"""
    if not {'sample_key','feature_archive'}.issubset(frame.columns):
        raise ValueError('缺少测量文件或样本标识')
    rows=[]
    for _,record in frame.iterrows():
        index,arrays=_archive(str(Path(record['feature_archive']).resolve()))
        values=arrays[view][index[str(record['sample_key'])]]
        if paired:
            if values.ndim!=3 or values.shape[0]!=2:raise ValueError('拟合需要合法源阶段的配对测量')
        else:
            values=values[1] if values.ndim==3 else values
            if values.ndim!=2:raise ValueError('预测需要重复观测×特征矩阵')
        rows.append(values)
    return np.stack(rows)


def _training_metadata(frame):
    paths=frame['training_metadata'].dropna().unique()
    if len(paths)!=1:raise ValueError('拟合必须引用同一份宿主授权的训练元数据')
    metadata=json.loads(Path(paths[0]).read_bytes())
    try:rows=[metadata[str(key)] for key in frame['sample_key']]
    except KeyError as exc:raise ValueError('样本不在本阶段可用训练元数据中') from exc
    return np.asarray([r['group'] for r in rows]),np.asarray([r['nuisance'] for r in rows])


class FeatureArchiveClassifier(ClassifierMixin,BaseEstimator):
    """同一fit/predict接口支持任意类别和测量宽度，保留物理暴露分组。

    view命名由任务的测量清单声明，mode选择已验证的两种共用校准。
    字符串标识仅用于取数组；实际学习和单暴露预测只使用测量值。
    """
    def __init__(self,view='raw_cycles',mode='paired'):
        self.view=view;self.mode=mode

    def fit(self,frame,y):
        x=load_reference_view(frame,self.view,paired=True)
        y=np.asarray(y);groups,nuisance=_training_metadata(frame)
        if y.shape!=(len(x),):raise ValueError('训练标签没有按物理暴露对齐')
        repeat=x.shape[2]
        if self.mode=='paired':
            self.model_=PairedNuisanceClassifier().fit(x[:,0].reshape(len(x)*repeat,-1),
                x[:,1].reshape(len(x)*repeat,-1),np.repeat(y,repeat),
                groups=np.repeat(groups,repeat),nuisance=np.repeat(nuisance,repeat))
        elif self.mode=='grouped':
            self.model_=GroupProbabilityClassifier().fit(x.reshape(len(x)*2*repeat,-1),np.repeat(y,2*repeat),
                groups=np.repeat(groups,2*repeat),target_mask=np.tile([False]*repeat+[True]*repeat,len(x)))
        else:raise ValueError('未知共用校准模式')
        self.classes_=self.model_.classes_
        self.feature_names_in_=np.asarray(frame.columns,dtype=object)
        return self

    def predict_proba(self,frame):
        check_is_fitted(self,'model_')
        x=load_reference_view(frame,self.view)
        probability=self.model_.predict_proba(x.reshape(len(x)*x.shape[1],-1))
        return probability.reshape(len(x),x.shape[1],-1).mean(1)

    def predict(self,frame):
        return self.classes_[self.predict_proba(frame).argmax(1)]
