"""TabICLv2与原生分类父网先作几何组合，再执行既有源修正规则。

可微训练目标是修正前的几何组合交叉熵；离散源修正保留在实际预测末端。
教师、源上下文和修正规则冻结，父权重及Adam继续真实更新。
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    try:
        from .complete_checkpoint import CompleteModel,FORMAT
    except ImportError:
        from .complete_model import CompleteModel,FORMAT
    from .gas_views import pair_model
    from .tabular_readout import author_package
    from .teacher_cache import prefetch_teacher
else:
    from complete_model import CompleteModel,FORMAT
    from gas_views import pair_model
    from tabular_readout import author_package
    from teacher_cache import prefetch_teacher

KIND='tabicl-classification-parent'


class _ProbabilityInput:
    def predict(self,raw):
        if len(raw)!=len(self.probabilities):
            raise ValueError('源修正的输入与组合概率长度不同')
        return self.probabilities.copy()


class TabICLClassifierModel(CompleteModel):
    def __init__(self,base,auxiliary_arrays):
        self.base=base
        self.auxiliary_arrays={k:v.copy() for k,v in auxiliary_arrays.items()}
        self.state=json.loads(str(self.auxiliary_arrays['complete_state']))
        self.teacher_state=json.loads(str(self.auxiliary_arrays['tabular_classifier_state']))
        if self.state.get('format')!=FORMAT or self.state.get('kind')!=KIND:
            raise ValueError('未知TabICL分类组合格式')
        x,y=self.auxiliary_arrays['classifier_source_x'],self.auxiliary_arrays['classifier_source_y']
        if x.ndim!=2 or x.shape[1]!=len(base.meta['paths']) or y.shape!=(len(x),):
            raise ValueError('分类源上下文与父输入不匹配')
        if not np.isfinite(x).all() or set(y)!=set(range(len(base.meta['classes']))):
            raise ValueError('分类源上下文需覆盖完整类别且输入有限')
        for field in self.teacher_state['tensor_fields'].values():
            if field not in self.auxiliary_arrays or self.auxiliary_arrays[field].dtype.kind!='f':
                raise ValueError('冻结教师数值状态不完整')
        self._teacher=None;self._training_teacher={};self._prediction_cache={};self._repair=None
        self.validate()

    def validate(self):
        self.base.validate()
        if self.base.meta['task']!='classification':
            raise ValueError('TabICL分类组合只能接分类父版')
        weight=self.state['parent_weight']
        if not isinstance(weight,(int,float)) or not np.isfinite(weight) or not 0<weight<=1:
            raise ValueError('实际父分类器权重须在(0,1]内')

    def normalized(self,raw):
        return self.base.normalized(raw)

    def _load_teacher(self):
        if self._teacher is not None:return
        import torch
        torch.set_num_threads(1)
        weights={name:torch.from_numpy(self.auxiliary_arrays[field].copy())
                 for name,field in self.teacher_state['tensor_fields'].items()}
        with TemporaryDirectory(prefix='harness-tabicl-classifier-') as temporary:
            path=Path(temporary)/'teacher.ckpt'
            torch.save({'state_dict':weights,'config':self.teacher_state['model_config']},path)
            model=author_package().TabICLClassifier(model_path=path,n_estimators=8,batch_size=1,
                kv_cache=True,device='cpu',n_jobs=1,allow_auto_download=False,random_state=42,use_amp=False)
            model.fit(self.auxiliary_arrays['classifier_source_x'].astype(np.float32),
                      self.auxiliary_arrays['classifier_source_y'])
        if not np.array_equal(model.classes_,np.arange(len(self.meta['classes']))):
            raise ValueError('教师类别顺序不匹配')
        self._teacher=model

    def prepare_training(self,raw):
        """只预测本轮合法训练行，临时概率不进入检查点。"""
        self._load_teacher();raw=np.asarray(raw,dtype=np.float64)
        for start in range(0,len(raw),128):
            batch=raw[start:start+128]
            probabilities=self._teacher.predict_proba(batch.astype(np.float32))
            for row,probability in zip(batch,probabilities,strict=True):
                self._training_teacher[row.tobytes()]=probability.copy()

    def continuation_output(self,raw,logits):
        try:
            teacher=np.asarray([self._training_teacher[row.tobytes()] for row in np.asarray(raw,dtype=np.float64)])
        except KeyError as exc:
            raise ValueError('分类训练行不在宿主预计算的合法集合') from exc
        return self._combine(logits,teacher)

    def _combine(self,logits,teacher):
        """具体教师组合定义同时供真实训练与独立预测使用。"""
        alpha=self.state['parent_weight']
        combined=alpha*logits+(1-alpha)*np.log(np.maximum(teacher,1e-30))
        return combined,np.full((len(logits),1),alpha)

    def predict_batch(self, raw):
        """内部完整评价可批量预取教师；公开单行入口仍立即返回一行预测。"""
        raw=np.asarray(raw,dtype=np.float64)
        if raw.ndim!=2 or raw.shape[1]!=len(self.meta['paths']) or not np.isfinite(raw).all():
            raise ValueError('分类组合输入无效')
        self._load_teacher()
        prefetch_teacher(self._prediction_cache,raw,self._teacher.predict_proba,width=len(self.meta['classes']))
        return self.predict(raw)

    def predict(self,raw):
        self._load_teacher();raw=np.asarray(raw,dtype=np.float64)
        if raw.ndim!=2 or raw.shape[1]!=len(self.meta['paths']) or not np.isfinite(raw).all():
            raise ValueError('分类组合输入无效')
        teacher=[]
        for row in raw:
            key=row.tobytes()
            if key not in self._prediction_cache:
                value=self._teacher.predict_proba(row[None,:].astype(np.float32))[0]
                if value.shape!=(len(self.meta['classes']),) or not np.isfinite(value).all():
                    raise ValueError('教师分类输出无效')
                self._prediction_cache[key]=value.copy()
            teacher.append(self._prediction_cache[key])
        logits=self.base.forward(self.base.normalized(raw))[1]
        logits,_=self._combine(logits,np.asarray(teacher))
        probability=np.exp(logits-logits.max(axis=1,keepdims=True))
        probability/=probability.sum(axis=1,keepdims=True)
        if 'gas_pair_patch' not in self.meta:return probability
        if self._repair is None:
            self._repair=pair_model(_ProbabilityInput)()
        self._repair.meta=self.base.meta;self._repair.mean=self.base.mean;self._repair.scale=self.base.scale
        self._repair.probabilities=probability
        return self._repair.predict(raw)


def classification_base(model):
    """保留原父网络和源修正；只替换已知policy=parent的空外壳描述，不丢弃其他组件。"""
    if hasattr(model,'auxiliary_arrays'):
        if (model.state.get('kind')!='gas-source-vote' or model.state.get('policy')!='parent'
                or set(model.auxiliary_arrays)!={'complete_state'}):
            raise ValueError('该完整父版的附加分量尚不支持此分类迁移')
        return model.base
    return model


def from_classifier_asset(model,asset_path,*,parent_weight=.25):
    base=classification_base(model)
    with np.load(asset_path,allow_pickle=False) as saved:
        teacher=json.loads(str(saved['tabular_classifier_state']))
        fields=list(teacher['tensor_fields'].values())+['classifier_source_x','classifier_source_y','tabular_classifier_state']
        arrays={field:saved[field].copy() for field in fields}
    state={'format':FORMAT,'kind':KIND,'parent_weight':float(parent_weight),
        'training_objective':'geometric_prediction_before_frozen_source_repair',
        'teacher_parameters':teacher['teacher_parameters'],'teacher_sha256':teacher['teacher_sha256'],
        'source_context_rows':len(arrays['classifier_source_y']),'feature_view':'raw',
        'context_updates':'frozen_source_context','source_repair':'original_parent_metadata'}
    arrays['complete_state']=np.asarray(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False))
    return TabICLClassifierModel(base,arrays)
