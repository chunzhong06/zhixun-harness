"""可编辑训练逻辑与官方父权重之间的接口；保存、计数和预测由宿主负责。"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import ast
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from harness.learning.inference.engine_parent_support import group_rows


def tensor_identity(model):
    """包含优化器与尺度，不以文件名或自报父引用判断载入状态。"""
    arrays = {**model.params, **{'adam_first_' + k: v for k, v in model.first.items()},
              **{'adam_second_' + k: v for k, v in model.second.items()},
              'mean': model.mean, 'scale': model.scale}
    arrays.update(getattr(model, 'auxiliary_arrays', {}))
    return {name: sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}


class ContinuationSession:
    """程序可选择训练方式，不能经公开接口替换父网络或伪造训练历史。

    梯度回调返回逐样本损失与未取均值的输出梯度；宿主计算反向传播和
    Adam更新。支持显式扩展和组件迁移，独立加载器随实际产物一起交付。
    """

    def __init__(self, model, x, y, *, max_epochs=30, groups=None, tabular_resource=None, teacher_cache=None):
        self._model = getattr(model, 'context_base', getattr(model, 'unaligned_engine', model))
        self._gas_partial = getattr(model, 'gas_context', None)
        self._incoming_x = np.asarray(x, dtype=np.float64)
        self._x = model.engine_alignment.transform(self._incoming_x) if hasattr(model, 'unaligned_engine') else self._incoming_x
        self._y = np.asarray(y)
        # 数值程序进入受限环境前，从本轮合法标签准备候选上限。程序内只登记数值。
        self._prepared_life_limit = None
        if self._model.meta.get('profile') == 'C' and getattr(self._model, 'state', {}).get('residual'):
            from harness.learning.training.engine_life_limit import source_limit
            self._prepared_life_limit = source_limit(self._y)
        self._groups = np.asarray(groups).copy() if groups is not None else np.repeat('source', len(self._y))
        if self._groups.shape != (len(self._y),):
            raise ValueError('训练组身份必须逐行对应合法训练资料')
        self._max_epochs = max_epochs
        self._history = []
        self._initial_meta = deepcopy(model.meta)
        self._initial_tensors = tensor_identity(model)
        self._training_origin_tensors = dict(self._initial_tensors)
        self._expected_meta = deepcopy(model.meta)
        self._migration = None
        self._first_step_tensors = None
        self._component_history = []
        self._head_epochs_used = 0
        self._expected_auxiliary = {k: v for k, v in self._initial_tensors.items()
                                    if k in getattr(model, 'auxiliary_arrays', {})}
        self._tabular_resource = tabular_resource
        self._teacher_cache = teacher_cache
        self._prepared_tabular = None
        self._prepared_tabpfn_readout = None
        self._prepared_classifier = None
        if hasattr(model, 'prepare_training'):
            if self._teacher_cache is not None and hasattr(model, 'bind_round_cache'):
                model.bind_round_cache(self._teacher_cache)
            model.prepare_training(self._x)

    def _prepare_tabular(self):
        """文件装载和教师推理均在数值程序的审计边界外由宿主完成。"""
        if self._tabular_resource is None or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('此父版或任务没有可新建的TabICL源组件')
        from harness.learning.inference.tabular_readout import from_asset
        resource = self._tabular_resource
        path = Path(resource['checkpoint'])
        if sha256(path.read_bytes()).hexdigest() != resource['sha256']:
            raise ValueError('共享TabICL组件身份改变')
        prepared = from_asset(self._model, path, parent_weight=.25)
        # 组件上下文必须逐行来自本轮合法训练资料，不能依赖验证标签。
        source = {x.tobytes()+np.float64(y).tobytes() for x, y in zip(self._x, self._y, strict=True)}
        context_x, context_y = prepared.auxiliary_arrays['readout_source_x'], prepared.auxiliary_arrays['readout_source_y']
        if any(x.astype(np.float64).tobytes()+np.float64(y).tobytes() not in source
               for x, y in zip(context_x, context_y, strict=True)):
            raise ValueError('TabICL上下文包含本轮未授权的训练行')
        prepared.prepare_training(self._x)
        self._prepared_tabular = prepared

    def build_tabicl_readout(self, parent_weight=.25):
        """可在真实父续训后接入冻结教师；原父参数、Adam和尺度原样进入组合。"""
        if self._prepared_tabular is None or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('教师只能由宿主预备并接入普通父网络一次')
        before = tensor_identity(self._model)
        model = self._prepared_tabular
        if model.base is not self._model:
            raise ValueError('教师预备期间实际父网络被替换')
        model.state['parent_weight'] = float(parent_weight)
        model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state, sort_keys=True, separators=(',', ':'), allow_nan=False))
        model.validate()
        current = tensor_identity(model)
        if {k:current[k] for k in before} != before:
            raise ValueError('接入教师改变了父权重或优化器')
        self._model = model
        self._expected_auxiliary = {k:v for k,v in current.items() if k in model.auxiliary_arrays}
        if self._first_step_tensors is None:
            self._training_origin_tensors = current
        report = {'kind':'tabicl-parent-combination','parent_weight':model.state['parent_weight'],
                  'retained_parent_tensors':before,'source_context_rows':model.state['source_context_rows'],
                  'teacher_training':'frozen','parent_epochs_before_attachment':len(self._history)}
        self._component_history.append(report)
        return deepcopy(report)

    def _prepare_tabpfn_readout(self):
        """核验新回归资源及同源上下文，在数值程序边界外预计算教师。"""
        if self._tabular_resource is None or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('新TabPFN回归组件须使用原生父版，不能剥离已有完整组合')
        from harness.learning.inference.tabpfn_regressor import from_regressor_asset
        resource = self._tabular_resource
        path = Path(resource['checkpoint'])
        if sha256(path.read_bytes()).hexdigest() != resource['sha256']:
            raise ValueError('共享TabPFN回归组件身份改变')
        prepared = from_regressor_asset(self._model, path)
        source = {x.tobytes() + np.float64(y).tobytes() for x, y in zip(self._x, self._y, strict=True)}
        cx, cy = prepared.auxiliary_arrays['readout_source_x'], prepared.auxiliary_arrays['readout_source_y']
        if any(x.tobytes() + np.float64(y).tobytes() not in source for x, y in zip(cx, cy, strict=True)):
            raise ValueError('TabPFN回归上下文包含本轮未授权的训练行或标签')
        if self._teacher_cache is not None:
            prepared.bind_round_cache(self._teacher_cache)
        prepared.prepare_training(self._x)
        self._prepared_tabpfn_readout = prepared

    def build_tabpfn_readout(self, parent_weight=.25):
        """显式首次接入新教师，保留此刻实际父权重、Adam及尺度；后续从完整组合续训。"""
        if self._prepared_tabpfn_readout is None or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('TabPFN回归教师只能由宿主预备后接入原生父版一次')
        if isinstance(parent_weight, bool):
            raise ValueError('真实回归父网权重不能是布尔值')
        before = tensor_identity(self._model)
        model = self._prepared_tabpfn_readout
        if model.base is not self._model:
            raise ValueError('回归教师预备期间实际父网被替换')
        model.state['parent_weight'] = float(parent_weight)
        model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state, sort_keys=True, separators=(',', ':'), allow_nan=False))
        model.validate()
        current = tensor_identity(model)
        if {k: current[k] for k in before} != before:
            raise ValueError('接入回归教师改变了父权重、Adam或尺度')
        self._model = model
        self._expected_auxiliary = {k: v for k, v in current.items() if k in model.auxiliary_arrays}
        if self._first_step_tensors is None:
            self._training_origin_tensors = current
        report = {'kind': model.state['kind'], 'parent_weight': model.state['parent_weight'],
            'retained_parent_tensors': before, 'source_context_rows': model.state['source_context_rows'],
            'teacher_sha256': model.state['teacher_sha256'], 'teacher_asset_sha256': model.state['teacher_asset_sha256'],
            'teacher_training': 'frozen', 'parent_epochs_before_attachment': len(self._history),
            'transition': 'new_component_from_native_parent',
            'function_preserving': model.state['parent_weight'] == 1,
            'training_objective': 'complete_prediction'}
        if 'bearing_cpu_readout' in model.state:
            report['serving_readout'] = dict(model.state['bearing_cpu_readout'])
        self._component_history.append(report)
        return deepcopy(report)

    def _prepare_tabular_classifier(self,backend='tabicl', *, baseline=False):
        """分类教师只看源上下文；文件和作者推理在数值程序审计边界外预备。"""
        if backend=='tabicl':
            from harness.learning.inference.tabular_classifier import from_classifier_asset
        elif backend=='tabpfn':
            from harness.learning.inference.tabpfn_classifier import TabPFNClassifierModel, from_classifier_asset
        else:
            raise ValueError('未知分类教师后端')
        if baseline and backend=='tabpfn' and isinstance(self._model,TabPFNClassifierModel):
            # 已有原始上下文随实际父产物继承，仅新增的第二组上下文使用本轮训练行。
            prepared=TabPFNClassifierModel(self._model.base,self._model.auxiliary_arrays)
        else:
            resource=self._tabular_resource
            if resource is None or sha256(Path(resource['checkpoint']).read_bytes()).hexdigest()!=resource['sha256']:
                raise ValueError('分类教师资源缺失或身份改变')
            prepared=from_classifier_asset(self._model,resource['checkpoint'])
            source={x.tobytes()+np.float64(y).tobytes() for x,y in zip(self._x,self._y,strict=True)}
            cx,cy=prepared.auxiliary_arrays['classifier_source_x'],prepared.auxiliary_arrays['classifier_source_y']
            if any(x.astype(np.float64).tobytes()+np.float64(y).tobytes() not in source for x,y in zip(cx,cy,strict=True)):
                raise ValueError('分类教师上下文包含未授权训练行')
        if baseline:
            if (backend == 'tabpfn' and not isinstance(self._model,TabPFNClassifierModel)
                    and prepared.teacher_state.get('context_origin') == 'current_train'):
                prepared.add_baseline_context(prepared.auxiliary_arrays['classifier_source_x'],
                                              prepared.auxiliary_arrays['classifier_source_y'])
            else:
                prepared.add_baseline_context(self._x,self._y)
        loaded = False
        if baseline and backend == 'tabpfn':
            from harness.learning.training.gas_training_asset import restore_training_targets
            loaded = restore_training_targets(prepared, self._x)
        if not loaded:
            prepared.prepare_training(self._x)
        self._prepared_classifier=prepared

    def _prepare_tabm(self):
        """宿主仅用当轮训练投影拟合紧凑教师，数值程序拿不到文件或验证数据。"""
        from harness.learning.inference.tabpfn_regressor import TabPFNRegressorModel
        from harness.learning.inference.teacher_cache import prefetch_teacher
        from harness.learning.training.tabm_training import fit_readout
        if (not isinstance(self._model, TabPFNRegressorModel) or self._model.state.get('tabm_readout')
                or self._max_epochs < 7):
            raise ValueError('需要尚未包含 TabM 的完整 TabPFN 回归父版及至少七轮训练预算')
        payload, fitted = fit_readout(self._x, self._y)
        cached = {}
        prefetch_teacher(cached, self._x, fitted.predict, nonnegative=True, batch_size=256)
        self._prepared_tabm = payload, fitted, cached
        self._max_epochs -= 6

    def build_tabm_readout(self):
        """保留完整父版全部分量，引入当轮源读出，下一梯度步作用于组合的误差。"""
        if not hasattr(self, '_prepared_tabm') or self._history:
            raise ValueError('TabM 读出须先由宿主拟合，并在首个父梯度步前接入')
        payload, fitted, cached = self._prepared_tabm
        del self._prepared_tabm
        self._model.attach_tabm(payload, fitted, cached, len(self._x))
        current = tensor_identity(self._model)
        if any(current[key] != digest for key,digest in self._initial_tensors.items() if key != 'complete_state'):
            raise ValueError('新增读出时改变了原父版的权重或冻结状态')
        self._training_origin_tensors = current
        self._expected_auxiliary = {k:current[k] for k in self._model.auxiliary_arrays}
        self._component_history.append({'kind':'tabm-readout', 'teacher_epochs':6, 'source_rows':len(self._x),
                                        'teacher_sha256':self._model.state['tabm_readout']['sha256'],
                                        'parent_weight':.5, 'training_objective':'complete_prediction'})

    def build_tabm_source_support(self, *, head_spread=False):
        """在真实父产物首次接续时，用当轮可见特征确定逐行组合的范围。"""
        model = self._model
        if self._history or not getattr(model, 'state', {}).get('tabm_readout'):
            raise ValueError('须在已有 TabM 的回归父产物上、首个梯度步前配置源范围')
        if type(head_spread) is not bool:
            raise ValueError('head_spread 须为布尔值')
        if head_spread:
            if not hasattr(self,'_prepared_head_support'):
                raise ValueError('需先有源特征范围，并用本轮训练行估计头间分歧')
            model.add_head_support(self._prepared_head_support)
            del self._prepared_head_support
        else:
            model.add_source_support(self._x)
        current = tensor_identity(model)
        if any(current[k] != v for k, v in self._initial_tensors.items() if k != 'complete_state'):
            raise ValueError('设置源范围时原父权重、优化器或教师发生改变')
        self._training_origin_tensors = current
        self._expected_auxiliary = {k: current[k] for k in model.auxiliary_arrays}
        details = {'kind':'tabm-source-support','source_rows':len(self._x),'inside_weight':.75,'outside_weight':.5}
        if head_spread:
            details.update(kind='tabm-head-support',inside_weight=.875,head_supported_outside_weight=.625,
                           source_spread_max=model.state['tabm_readout']['head_support']['max_std'])
        self._component_history.append(details)

    def build_bearing_health_cap(self):
        """冻结源训练的振动严重度读出，随后只按完整预测更新父网。"""
        model=self._model
        readout=getattr(model,'state',{}).get('tabm_readout',{})
        if (self._history or 'head_support' not in readout or 'health_cap' in readout
                or not hasattr(self,'_prepared_health_cap')):
            raise ValueError('需要有多头分歧的轴承父版，并在首个梯度步前接入')
        spec=self._prepared_health_cap
        model.load_health_cap(spec)
        readout['health_cap']=spec
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        current=tensor_identity(model)
        if any(current[k]!=v for k,v in self._initial_tensors.items() if k!='complete_state'):
            raise ValueError('设置寿命上界时已有父权重、优化器或读出已改变')
        self._training_origin_tensors=current
        self._expected_auxiliary={k:current[k] for k in model.auxiliary_arrays}
        self._component_history.append({'kind':'bearing-energy-impulse-cap','source_rows':spec['rows'],'levels':len(spec['thresholds'])})
        del self._prepared_health_cap

    def build_tabicl_classifier(self,parent_weight=.25):
        """保留父网、Adam和原源修正；可微组合在离散源修正之前。"""
        return self._attach_classifier(parent_weight,'tabicl-classification-parent')

    def build_bearing_grid_readout(self):
        """登记来自本轮标签的不同谱窗口寿命校准，保留其他窗口和真实父优化器。"""
        spec = getattr(self,'_prepared_bearing_grid',None)
        if spec is None:
            raise ValueError('本轮源标签未覆盖新的可辨功率谱窗口')
        model = self._model
        prepared = getattr(self, '_prepared_native_grid_model', None)
        if prepared is not None:
            if prepared.base is not model or self._first_step_tensors is None:
                raise ValueError('请先从传入的原生轴承父模型完成真实梯度续训')
            self._model = model = prepared
            del self._prepared_native_grid_model
        else:
            model.load_grid_readout(spec)
            model.state['bearing_grid_readout'] = spec
            model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        self._expected_auxiliary['complete_state'] = sha256(model.auxiliary_arrays['complete_state'].tobytes()).hexdigest()
        if self._first_step_tensors is None:
            self._training_origin_tensors = tensor_identity(model)
        self._component_history.append({'kind':'bearing-spectrum-grid-ridge','grids':[m['bins'] for m in spec['models']]})
        del self._prepared_bearing_grid
        return deepcopy(spec)

    def build_engine_setting_alignment(self):
        """使用合法源行对照得到的编码；以变换后的真实输入进行之后的父网络梯度训练。"""
        prepared = getattr(self, '_prepared_engine_alignment', None)
        if prepared is None or self._history or self._migration:
            raise ValueError('新工况编码只可从传入父版、在本轮首个梯度之前安装')
        entries, aligned_rows = prepared
        self._x = aligned_rows
        if entries:
            self._model.meta['engine_input_alignment'] = deepcopy(entries)
            self._expected_meta['engine_input_alignment'] = deepcopy(entries)
        else:
            self._model.meta.pop('engine_input_alignment', None)
            self._expected_meta.pop('engine_input_alignment', None)
        self._component_history.append({'kind':'source-engine-setting-alignment', 'mappings':deepcopy(entries)})
        del self._prepared_engine_alignment
        return deepcopy(entries)

    def build_gas_partial_context(self, *, view='log', protect_missing=True):
        """在当前源测量支持范围内更新类别中心，保留真实父网络及以前采用的设备上下文。"""
        ready = getattr(self, '_prepared_gas_partial', {})
        key = view, protect_missing
        if type(protect_missing) is not bool or key not in ready or self._history or self._migration:
            raise ValueError('气体小样本上下文须由当前合法训练行在首个梯度前建立')
        spec,self._gas_partial = ready[key]
        self._model.meta['gas_partial_context'] = deepcopy(spec)
        self._expected_meta['gas_partial_context'] = deepcopy(spec)
        latest=spec['banks'][-1]
        self._component_history.append({'kind':'gas-partial-context','view':view,'protect_missing':protect_missing,
                                        'rows':len(latest['labels']),'classes':latest['classes'],'source_sha256':latest['source_sha256']})
        del self._prepared_gas_partial
        return deepcopy(self._component_history[-1])

    def build_tabpfn_classifier(self,parent_weight=.25):
        """算术组合教师与真实父网；梯度经完整组合Jacobian返回父网。"""
        return self._attach_classifier(parent_weight,'tabpfn-classification-parent')

    def build_gas_baseline_ensemble(self,parent_weight=None):
        """接入由本轮合法训练行产生的第二上下文；接续已发布父版时沿用其比例。"""
        if self._prepared_classifier is None or not self._prepared_classifier.state.get('baseline_context'):
            raise ValueError('本底上下文尚未由宿主准备')
        if parent_weight is None:
            parent_weight=self._prepared_classifier.state['parent_weight']
        return self._attach_classifier(parent_weight,'tabpfn-classification-parent',baseline=True)

    def build_gas_drift_readout(self):
        """附加当轮训练出的单向读出，保留原父梯度、教师和此前的源修正。"""
        payload=getattr(self,'_prepared_drift_readout',None)
        model=self._model
        if (payload is None or getattr(model,'state',{}).get('kind')!='tabpfn-classification-parent'
                or not model.state.get('baseline_context') or model.state.get('drift_readout')):
            raise ValueError('该父版或本轮训练行不支持新建气体漂移读出')
        spec,coef=payload
        model.state['drift_readout']=deepcopy(spec)
        model.auxiliary_arrays['drift_coefficients']=coef.copy()
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        model._check_drift_readout()
        for key in ('drift_coefficients','complete_state'):
            self._expected_auxiliary[key]=sha256(model.auxiliary_arrays[key].tobytes()).hexdigest()
        if self._first_step_tensors is None:
            self._training_origin_tensors=tensor_identity(model)
        self._component_history.append({'kind':'gas-drift-readout',**spec})
        del self._prepared_drift_readout
        return deepcopy(spec)

    def build_gas_toluene_readout(self):
        """在现有漂移读出之后复核甲苯，使用当轮训练行单独拟合新的 SVM。"""
        model = self._model
        prepared = getattr(self,'_prepared_toluene',None)
        if (prepared is None or not getattr(model,'state',{}).get('drift_readout')
                or 'toluene_readout' in model.state):
            raise ValueError('该父版或当轮训练行不支持添加甲苯 SVM')
        self._install_gas_readout('toluene_readout','gas-toluene-readout',prepared)
        del self._prepared_toluene

    def build_gas_kinetic_support(self):
        """在甲苯复核之后追加乙醇源覆盖判断，全部参数只由本轮训练行计算。"""
        prepared=getattr(self,'_prepared_gas_kinetics',None)
        state=getattr(self._model,'state',{})
        if prepared is None or 'toluene_readout' not in state or 'kinetic_readout' in state:
            raise ValueError('乙醛源覆盖判断需要具备甲苯读出的完整父版和六类合法训练行')
        self._install_gas_readout('kinetic_readout','gas-kinetic-support',prepared)
        del self._prepared_gas_kinetics
        return deepcopy(prepared[0])

    def _install_gas_readout(self,field,kind,prepared):
        model=self._model
        spec,arrays=prepared
        model.auxiliary_arrays.update(arrays)
        model.state[field] = spec
        model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        for key in (*arrays,'complete_state'):
            self._expected_auxiliary[key] = sha256(model.auxiliary_arrays[key].tobytes()).hexdigest()
        if self._first_step_tensors is None:
            self._training_origin_tensors=tensor_identity(model)
        self._component_history.append({'kind':kind,'source_rows':spec['rows'],'source_x_sha256':spec['source_x_sha256']})

    def build_gas_device_context(self):
        """从新发放标签建立局部教师；沿用实际父版本已有的量化权重和未覆盖的类别。"""
        prepared = getattr(self, '_prepared_gas_device', None)
        if prepared is None:
            raise ValueError('本轮合法源资料未形成可辨的新设备上下文')
        self._install_gas_readout('device_context', 'gas-recent-device-tabpfn', prepared)
        self._model.reset_device_context()
        del self._prepared_gas_device
        return deepcopy(prepared[0])

    def _attach_classifier(self,parent_weight,expected_kind,*,baseline=False):
        from harness.learning.inference.tabular_classifier import classification_base
        if self._prepared_classifier is None or self._prepared_classifier.state.get('kind')!=expected_kind:
            raise ValueError('对应分类教师须先由宿主预备')
        inherited=baseline and getattr(self._model,'state',{}).get('kind')==expected_kind
        base=self._model.base if inherited else classification_base(self._model)
        before=tensor_identity(base)
        metadata_sha=sha256(json.dumps(base.meta,sort_keys=True).encode()).hexdigest()
        old_wrapper=deepcopy(getattr(self._model,'state',None))
        model=self._prepared_classifier;model.base=base
        model.state['parent_weight']=float(parent_weight)
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        model.validate();current=tensor_identity(model)
        if {k:current[k] for k in before}!=before:
            raise ValueError('分类组件改变了父权重、Adam或尺度')
        if inherited:
            for key,value in self._model.auxiliary_arrays.items():
                if key!='complete_state' and not np.array_equal(model.auxiliary_arrays[key],value):
                    raise ValueError('新增本底上下文改变了父版已存组件')
        self._model=model
        self._expected_auxiliary={k:v for k,v in current.items() if k in model.auxiliary_arrays}
        if self._first_step_tensors is None:self._training_origin_tensors=current
        report={'kind':expected_kind,'parent_weight':model.state['parent_weight'],
            'retained_parent_tensors':before,'retained_parent_metadata_sha256':metadata_sha,
            'replaced_wrapper_descriptor':old_wrapper,'teacher_training':'frozen',
            'parent_epochs_before_attachment':len(self._history),'source_context_rows':model.state['source_context_rows'],
            'training_objective':model.state['training_objective']}
        if baseline:
            report['baseline_context']=deepcopy(model.state['baseline_context'])
        self._component_history.append(report)
        return deepcopy(report)

    def build_gas_repair(self, reported, target):
        """首次梯度前用源训练构建类别修正；保留父网络、Adam和原尺度。"""
        if self._history or self._migration or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('源形状修正只用于尚未训练的普通父版')
        from harness.learning.training.gas_components import attach_repair
        before = self._model.predict(self._x[:128])
        model = attach_repair(self._model, self._x, self._y, reported, target)
        current = tensor_identity(model)
        retained = {k: current[k] for k in self._initial_tensors}
        if retained != self._initial_tensors:
            raise ValueError('源形状修正改变了实际父权重或优化器')
        self._model = model
        self._migration = {'kind': 'source-gas-shape-repair', 'reported': reported, 'target': target,
            'retained_parent_tensors': retained,
            'initial_prediction_max_difference': float(np.max(np.abs(before-model.predict(self._x[:128]))))}
        self._expected_meta = deepcopy(model.meta)
        self._training_origin_tensors = current
        self._expected_auxiliary = {k: v for k, v in current.items() if k in model.auxiliary_arrays}
        return deepcopy(self._migration)

    def set_gas_policy(self, policy):
        """已有完整父版可提名不同读出；本轮仍必须真实执行父网络梯度更新。"""
        from harness.learning.training.gas_components import set_policy
        report = set_policy(self._model, policy)
        self._component_history.append(report)
        current = tensor_identity(self._model)
        self._expected_auxiliary = {k: v for k, v in current.items() if k in self._model.auxiliary_arrays}
        if not self._history:
            self._training_origin_tensors = current
        return deepcopy(report)

    def build_engine_life_neighbors(self):
        """把根据输入父版和当轮源样本冻结的检索组件装入当前候选。"""
        model=self._model
        prepared=getattr(self,'_prepared_life_neighbors',None)
        if prepared is None or getattr(model,'_density',None) is None or 'engine_neighbors' in model.state:
            raise ValueError('需要已装配条件密度且尚未装配同工况近邻的父版')
        spec,arrays=prepared
        model.auxiliary_arrays.update(arrays)
        model.state['engine_neighbors']=spec
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        model.load_life_neighbors()
        for key in (*arrays,'complete_state'):
            self._expected_auxiliary[key]=sha256(model.auxiliary_arrays[key].tobytes()).hexdigest()
        if self._first_step_tensors is None:
            self._training_origin_tensors=tensor_identity(model)
        self._component_history.append({'kind':'engine-gradient-neighbors','source_rows':spec['rows'],'source_x_sha256':spec['source_x_sha256']})
        del self._prepared_life_neighbors

    def set_engine_parent_pullback(self, *, enabled=True):
        """在本轮已有梯度更新的候选上改变输出修正，同时保留父网参数与优化器。"""
        model = self._model
        support = getattr(model,'parent_support',None)
        if type(enabled) is not bool or support is None or model.meta.get('profile') != 'C' or not self._history:
            raise ValueError('发动机父网回调需要含原始父网的 C 模型和本轮梯度更新')
        if support.quarter_pullback == enabled:
            raise ValueError('发动机父网回调状态没有改变')
        spec = model.state['engine_parent_support']
        if enabled:
            spec['quarter_pullback'] = True
        else:
            spec.pop('quarter_pullback')
        support.quarter_pullback = enabled
        encoded = json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False)
        model.auxiliary_arrays['complete_state'] = np.asarray(encoded)
        self._expected_auxiliary['complete_state'] = sha256(model.auxiliary_arrays['complete_state'].tobytes()).hexdigest()
        self._component_history.append({'kind':'engine-parent-quarter-pullback','enabled':enabled})

    def set_engine_life_limit(self, *, enabled=True):
        """按当轮标签设置或解除上限；同一父版可另行续训并比较不设限的候选。"""
        model = self._model
        if type(enabled) is not bool or model.meta.get('profile') != 'C' or not getattr(model, 'state', {}).get('residual'):
            raise ValueError('寿命上限仅适用于可继续训练的发动机残差父版')
        limit = self._prepared_life_limit if enabled else None
        old = model.state.get('engine_life_limit')
        if (enabled and limit is None) or old == limit:
            raise ValueError('本轮没有可登记的寿命上限变更')
        if enabled:
            model.state['engine_life_limit'] = limit
        else:
            del model.state['engine_life_limit']
        model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(
            model.state, sort_keys=True, separators=(',',':'), allow_nan=False))
        self._expected_auxiliary['complete_state'] = sha256(model.auxiliary_arrays['complete_state'].tobytes()).hexdigest()
        if self._first_step_tensors is None:
            self._training_origin_tensors = tensor_identity(model)
        self._component_history.append({'kind':'source-engine-life-limit', 'rows':len(self._y),
                                        'previous':old, 'limit':limit})

    def build_engine_condition_density(self):
        """把源训练联合密度接到完整发动机父版，保留已有父网、残差头及其优化器。"""
        model=self._model
        prepared=getattr(self,'_prepared_condition_density',None)
        if (prepared is None or self._history or not getattr(model,'state',{}).get('residual')
                or 'engine_condition_density' in model.state or model.meta.get('profile')!='C'):
            raise ValueError('需要尚未安装条件密度的完整发动机父版，且须在首个梯度步前接入')
        spec,arrays=prepared
        model.auxiliary_arrays.update(arrays)
        model.state['engine_condition_density']=spec
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        model.load_condition_density()
        current=tensor_identity(model)
        if any(current[k]!=v for k,v in self._initial_tensors.items() if k!='complete_state'):
            raise ValueError('附加寿命密度时原父权重、优化器或冻结组件发生变化')
        self._training_origin_tensors=current
        self._expected_auxiliary={k:current[k] for k in model.auxiliary_arrays}
        self._component_history.append({'kind':'engine-condition-density','source_rows':spec['rows'],'weight':spec['weight']})
        del self._prepared_condition_density

    def build_regression_components(self, condition_indices, decimals, *, epochs=60, head_epochs=12):
        """源训练工况处理、真实父网络续训和残差训练；只允许从普通父版构建一次。"""
        if self._history or self._migration or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('完整组件构建只用于尚未训练的普通父版')
        if self._model.meta['task'] != 'regression' or type(epochs) is not int or type(head_epochs) is not int:
            raise ValueError('完整组件需要回归父版与整数训练轮数')
        if not 1 <= epochs <= self._max_epochs or not 1 <= head_epochs <= 12:
            raise ValueError('完整组件训练轮数超出本次预算')
        from harness.learning.training.complete_training import attach_inputs, attach_residual
        old = self._model.predict(self._x[:128])
        # 工况变换和梯度步都会改变预测。后备分支保存进入本轮时实际收到的原生网络。
        incoming = {k:v.copy() for k,v in self._model.params.items()} if self._model.meta.get('profile') == 'C' else None
        self._model = attach_inputs(self._model, self._x, self._groups, condition_indices, decimals)
        current = tensor_identity(self._model)
        retained = {k: current[k] for k in self._initial_tensors}
        if retained != self._initial_tensors:
            raise ValueError('输入组件构建改变了父参数或优化器')
        self._migration = {'kind': 'source-trained-input-components', 'retained_parent_tensors': retained,
            'initial_prediction_max_difference': float(np.max(np.abs(old-self._model.predict(self._x[:128])))),
            'function_preserving': False, 'source_only': True}
        self._training_origin_tensors = current
        for _ in range(epochs):
            self.train_epoch()
        self._component_history.append(attach_residual(self._model, self._x, self._y, self._groups, head_epochs))
        if incoming is not None:
            model = self._model
            known = sorted(group_rows(self._x, model.state['condition']))
            model.state['engine_parent_support'] = {'groups':known, 'calibration':[]}
            model.auxiliary_arrays.update({'engine_parent_'+k:v for k,v in incoming.items()})
            model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state, sort_keys=True, separators=(',',':')))
            model.load_parent_support()
        self._head_epochs_used += head_epochs
        final = tensor_identity(self._model)
        self._expected_auxiliary = {k: final[k] for k in self._model.auxiliary_arrays}
        return deepcopy(self._component_history[-1])

    def train_residual(self, *, epochs=1, learning_rate=None, trainable='readout', loss='huber', beta=3.):
        """恢复完整父版的残差权重与AdamW后续训，保持源工况和输入尺度。

        建头与续训共享每程序12头轮额度。没有保存优化器的旧产物不能沿此
        接口伪造恢复；最终审计仍要求原父网络发生实际梯度更新。
        """
        if type(epochs) is not int or not 1 <= epochs <= 12-self._head_epochs_used:
            raise ValueError('残差构建与续训合计最多12轮')
        from harness.learning.training.residual_continuation import continue_residual
        report = continue_residual(self._model, self._x, self._y, self._groups, epochs=epochs,
            learning_rate=learning_rate, trainable=trainable, loss=loss, beta=beta)
        self._head_epochs_used += epochs
        self._component_history.append(report)
        # 只有经过宿主检验的头更新可以改变附加状态；尺度及工况由该工具核对。
        self._expected_auxiliary = {name: value for name, value in tensor_identity(self._model).items()
                                   if name in self._model.auxiliary_arrays}
        return deepcopy(report)

    @property
    def metadata(self):
        return deepcopy(self._model.meta)

    @property
    def targets(self):
        return self._y.copy()

    @property
    def features(self):
        """原始合法训练特征副本；列序由metadata['paths']给出。"""
        return self._x.copy()

    @property
    def groups(self):
        """合法训练资料的采集单位，只用于分组统计或权重，不加入预测特征。"""
        return self._groups.copy()

    def training_predictions(self):
        if hasattr(self._model, 'training_predictions'):
            result = self._model.training_predictions(self._x).copy()
        else:
            result = self._model.predict(self._x).copy()
        return self._gas_partial.apply(self._x,result) if self._gas_partial is not None else result

    def expand_hidden(self, hidden):
        """在首个梯度前增加隐藏单元；保留原权重和Adam块，新输出连接为零。"""
        model = self._model
        old_hidden = model.meta['hidden']
        if self._history or self._migration is not None:
            raise ValueError('一次程序只能在首个梯度前扩展网络一次')
        if type(hidden) is not int or not old_hidden < hidden <= 128:
            raise ValueError('当前参考加载器支持扩展至128；更大结构需另实现兼容迁移和加载器')
        before = model.predict(self._x[:128])
        rng = np.random.default_rng(np.random.SeedSequence([model.meta['seed'], old_hidden, hidden]))
        extra = hidden - old_hidden
        additions = {'w1': rng.normal(0, 1 / math.sqrt(len(model.meta['paths'])),
                                      size=(len(model.meta['paths']), extra)),
                     'b1': np.zeros(extra), 'w2': np.zeros((extra, model.params['w2'].shape[1]))}
        axes = {'w1': 1, 'b1': 0, 'w2': 0}
        for name, value in additions.items():
            model.params[name] = np.concatenate([model.params[name], value], axis=axes[name])
            model.first[name] = np.concatenate([model.first[name], np.zeros_like(value)], axis=axes[name])
            model.second[name] = np.concatenate([model.second[name], np.zeros_like(value)], axis=axes[name])
        model.meta['hidden'] = hidden
        model.meta['parameter_count'] = sum(value.size for value in model.params.values())
        retained = {}
        for prefix, values in (('', model.params), ('adam_first_', model.first), ('adam_second_', model.second)):
            for name, value in values.items():
                original = value[:, :old_hidden] if name == 'w1' else value[:old_hidden] if name in ('b1', 'w2') else value
                retained[prefix + name] = sha256(original.tobytes()).hexdigest()
        retained.update(mean=sha256(model.mean.tobytes()).hexdigest(), scale=sha256(model.scale.tobytes()).hexdigest())
        retained.update({k: sha256(v.tobytes()).hexdigest()
                         for k, v in getattr(model, 'auxiliary_arrays', {}).items()})
        if retained != self._initial_tensors:
            raise ValueError('扩展没有完整保留父权重与优化器块')
        model.validate()
        delta = float(np.max(np.abs(before - model.predict(self._x[:128]))))
        if delta > 1e-9:
            raise ValueError('零输出连接扩展改变了父网络初始预测')
        self._migration = {'kind': 'zero-output-hidden-expansion', 'old_hidden': old_hidden,
                           'new_hidden': hidden, 'retained_parent_tensors': retained,
                           'initial_prediction_max_difference': delta}
        self._expected_meta = deepcopy(model.meta)
        self._training_origin_tensors = tensor_identity(model)
        return deepcopy(self._migration)

    def train_epoch(self, *, learning_rate=None, batch_size=None, trainable=None,
                    loss_gradient=None, sample_weights=None):
        if len(self._history) >= self._max_epochs:
            raise ValueError('本次源码试验达到训练轮数上限')
        model = self._model
        if learning_rate is not None:
            if isinstance(learning_rate, bool) or not 0 < float(learning_rate) <= 1:
                raise ValueError('学习率必须在(0, 1]内')
            model.meta['learning_rate'] = float(learning_rate)
        if batch_size is not None:
            if type(batch_size) is not int or not 1 <= batch_size <= 65536:
                raise ValueError('批大小无效')
            model.meta['batch_size'] = batch_size
        names = tuple(model.params) if trainable is None else tuple(trainable)
        if not names or len(set(names)) != len(names) or set(names) - set(model.params):
            raise ValueError('训练参数必须是已有参数的非空子集')
        weights = np.ones(len(self._y)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
        if weights.shape != self._y.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('样本权重必须对应合法训练行且非负')
        weights = weights / weights.mean()
        # 接续已交付模型时才建立后备工况的校准；首次建模与本程序之后的梯度步不重复拟合。
        if not self._history and getattr(model,'parent_support',None) is not None:
            fitted = model.parent_support.fit_released(self._x,self._y)
            if fitted:
                model.state['engine_parent_support']['calibration'] = [
                    {'key':key,'slope':slope,'offset':offset}
                    for key,(slope,offset) in sorted(model.parent_support.calibration.items())]
                model.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
                digest = sha256(model.auxiliary_arrays['complete_state'].tobytes()).hexdigest()
                self._expected_auxiliary['complete_state'] = self._training_origin_tensors['complete_state'] = digest
                self._component_history.append({'kind':'engine-parent-calibration','released_groups':fitted})
        x = model.normalized(self._x)
        regression = model.meta['task'] == 'regression'
        target = ((self._y - model.meta['target_mean']) / model.meta['target_scale']) if regression else self._y
        rng = np.random.default_rng(np.random.SeedSequence([model.meta['seed'], model.meta['epochs']]))
        order = rng.permutation(len(x))
        started = time.monotonic()
        initial_steps = model.meta['steps']
        loss_sum = 0.0
        if self._first_step_tensors is None:
            self._first_step_tensors = tensor_identity(model)
            if self._first_step_tensors != self._training_origin_tensors:
                raise ValueError('第一次梯度更新前改变了父模型状态')
        for offset in range(0, len(order), model.meta['batch_size']):
            indices = order[offset:offset + model.meta['batch_size']]
            batch, expected = x[indices], target[indices]
            hidden, logits = model.forward(batch)
            output_derivative = None
            if hasattr(model, 'continuation_output'):
                # 完整父版的冻结残差头也影响最终误差；梯度穿过该头回到真实父网络。
                logits, output_derivative = model.continuation_output(self._x[indices], logits)
            if self._gas_partial is not None:
                logits, output_derivative = self._gas_partial.continuation(self._x[indices], logits, output_derivative)
            if loss_gradient is not None:
                losses, delta = loss_gradient(logits.copy(), expected.copy())
                losses, delta = np.asarray(losses), np.asarray(delta, dtype=np.float64)
            elif regression:
                error = logits[:, 0] - expected
                losses, delta = error ** 2, (2 * error)[:, None]
            else:
                logits -= logits.max(axis=1, keepdims=True)
                probability = np.exp(logits)
                probability /= probability.sum(axis=1, keepdims=True)
                losses = -np.log(np.maximum(probability[np.arange(len(indices)), expected], 1e-300))
                delta = probability
                delta[np.arange(len(indices)), expected] -= 1
            if losses.shape != (len(indices),) or delta.shape != logits.shape or not np.isfinite(losses).all() or not np.isfinite(delta).all():
                raise ValueError('自定义损失须返回逐样本损失及同形状有限梯度')
            loss_sum += float(np.sum(losses * weights[indices]))
            delta = delta * weights[indices, None] / len(indices)
            if output_derivative is not None:
                if np.ndim(output_derivative)==3:
                    if output_derivative.shape!=(len(indices),delta.shape[1],delta.shape[1]) or not np.isfinite(output_derivative).all():
                        raise ValueError('组合输出Jacobian的形状或数值无效')
                    delta=np.einsum('ni,nij->nj',delta,output_derivative)
                else:
                    delta *= output_derivative
            gradients = {'w2': hidden.T @ delta, 'b2': delta.sum(axis=0)}
            hidden_delta = (delta @ model.params['w2'].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(value ** 2)) for name, value in gradients.items() if name in names))
            if not math.isfinite(norm):
                raise ValueError('梯度范数不是有限值')
            model.meta['steps'] += 1
            for name in names:
                gradient = gradients[name] * min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * gradient
                model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
                first = model.first[name] / (1 - .9 ** model.meta['steps'])
                second = model.second[name] / (1 - .999 ** model.meta['steps'])
                model.params[name] -= model.meta['learning_rate'] * first / (np.sqrt(second) + 1e-8)
            model.validate()
        model.meta['epochs'] += 1
        record = {'epoch': model.meta['epochs'], 'initial_steps': initial_steps,
                  'step': model.meta['steps'], 'loss': loss_sum / len(x),
                  'trainable': list(names), 'custom_gradient': loss_gradient is not None,
                  'sample_weights_sha256': sha256(weights.tobytes()).hexdigest(),
                  'learning_rate': model.meta['learning_rate'], 'batch_size': model.meta['batch_size'],
                  'seconds': time.monotonic() - started}
        self._history.append(record)
        return deepcopy(record)

    def audit(self):
        if not self._history:
            raise ValueError('训练程序没有执行实际梯度更新')
        for key in ('format', 'task', 'profile', 'paths', 'classes', 'hidden', 'seed', 'target_mean', 'target_scale', 'parameter_count'):
            if self._model.meta[key] != self._expected_meta[key]:
                raise ValueError('当前续训接口不支持隐式结构、任务或尺度迁移：' + key)
        current = tensor_identity(self._model)
        if any(current[name] != self._initial_tensors[name] for name in ('mean', 'scale')):
            raise ValueError('续训改变了原标准化状态')
        if self._model.meta.get('engine_input_alignment') != self._expected_meta.get('engine_input_alignment'):
            raise ValueError('发动机输入映射发生未登记改变')
        if self._model.meta.get('gas_partial_context') != self._expected_meta.get('gas_partial_context'):
            raise ValueError('气体局部分类器发生未登记改变')
        if any(current[name] != expected for name, expected in self._expected_auxiliary.items()):
            raise ValueError('本次父网络续训改变了冻结的输入处理或残差头')
        return {'initial_steps': self._initial_meta['steps'], 'final_steps': self._model.meta['steps'],
                'initial_epochs': self._initial_meta['epochs'], 'final_epochs': self._model.meta['epochs'],
                'loaded_tensors': self._initial_tensors, 'before_first_gradient': self._first_step_tensors,
                'migration': deepcopy(self._migration),
                'component_history': deepcopy(self._component_history),
                'final_tensors': current, 'history': deepcopy(self._history),
                'parameter_changes': [k for k in self._model.params if current[k] != self._initial_tensors[k]]}


def validate_program(source):
    """只接受训练与数值计算程序；这是计算接口约束，不是操作系统沙箱。"""
    tree = ast.parse(source)
    forbidden = {'open', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars', 'type', 'input', 'breakpoint'}
    io_names = {'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt', 'fromfile', 'tofile', 'memmap', 'ctypes', 'ctypeslib', 'dump', 'dumps'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in forbidden):
            raise ValueError('训练源码使用了非数值执行能力：' + node.id)
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in io_names):
            raise ValueError('训练源码不能读写文件或访问内部状态：' + node.attr)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if any(name not in ('numpy', 'math') for name in names) or getattr(node, 'level', 0):
                raise ValueError('训练源码仅需numpy或math')
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError('训练源码不能替换宿主对象或全局状态')
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'train' for n in tree.body):
        raise ValueError('必须提供train(session)函数')
    return tree


def run_program(source, session):
    tree = validate_program(source)
    numerical_methods = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    if 'build_gas_partial_context' in numerical_methods:
        from harness.learning.inference.gas_partial_readout import PartialContext, fit, prepare
        model=session._model
        prepared=prepare(model,session._incoming_x,session._y)
        session._prepared_gas_partial={}
        if prepared is not None:
            for view in ('log','shape'):
                for protect in (True,False):
                    spec=fit(prepared,view,protect)
                    if spec is not None:
                        session._prepared_gas_partial[view,protect]=spec,PartialContext(spec,model)
    if 'build_engine_setting_alignment' in numerical_methods and session._model.meta.get('profile') == 'C':
        from harness.learning.inference.engine_setting_alignment import SettingAlignment, fit
        model = session._model
        entries = fit(model, session._incoming_x, session._y, model.meta.get('engine_input_alignment', []))
        if entries is not None:
            transform = SettingAlignment(model.meta['paths'], entries)
            session._prepared_engine_alignment = entries, transform.transform(session._incoming_x)
    if 'build_engine_life_neighbors' in numerical_methods:
        from harness.learning.inference.engine_neighbors import fit
        session._prepared_life_neighbors=fit(session._model,session._x,session._y)
    if 'build_bearing_health_cap' in numerical_methods:
        from harness.learning.inference.bearing_health_readout import fit
        session._prepared_health_cap=fit(session._x,session._y,session._model.meta['paths'])
    if 'build_bearing_grid_readout' in numerical_methods:
        from harness.learning.inference.bearing_grid_readout import fit
        model = session._model
        native = not hasattr(model, 'auxiliary_arrays') and model.meta.get('profile') == 'B'
        compatible = getattr(model, 'state', {}).get('kind') in ('tabpfn-regression-parent', 'bearing-native-grid-parent')
        session._prepared_bearing_grid = fit(model, session._x, session._y) if native or compatible else None
        if native and session._prepared_bearing_grid is not None:
            from harness.learning.inference.complete_checkpoint import NativeBearingGridModel
            session._prepared_native_grid_model = NativeBearingGridModel.from_base(model, session._prepared_bearing_grid)
    if ('build_tabm_source_support' in numerical_methods
            and getattr(session._model,'_support_bounds',None) is not None
            and getattr(session._model,'_head_limit',None) is None):
        session._prepared_head_support=session._model.prepare_head_spread(session._x)
    if 'build_engine_condition_density' in numerical_methods:
        from harness.learning.inference.engine_condition_density import fit_readout
        session._prepared_condition_density=fit_readout(session._x,session._y,session._model.state['condition'])
    if 'build_gas_toluene_readout' in numerical_methods:
        from harness.learning.inference.gas_toluene_readout import fit_readout
        session._prepared_toluene=fit_readout(session._x,session._y,session._model.meta['classes'])
    if 'build_gas_kinetic_support' in numerical_methods:
        from harness.learning.inference.gas_kinetic_support import fit
        session._prepared_gas_kinetics=fit(session._x,session._y,session._model.meta['classes'])
    if 'build_gas_device_context' in numerical_methods:
        from harness.learning.inference.gas_device_context import prepare
        model = session._model
        session._prepared_gas_device = (prepare(session._x, session._y, model.auxiliary_arrays['classifier_source_x'],
            model.meta['classes'], model.state.get('device_context'))
            if getattr(model, 'state', {}).get('kind') == 'tabpfn-classification-parent' else None)
    if 'build_gas_drift_readout' in numerical_methods:
        from harness.learning.inference.gas_drift_readout import fit_readout
        session._prepared_drift_readout=fit_readout(session._x,session._y,session._model.meta['classes'])
    if 'build_tabm_readout' in numerical_methods:
        session._prepare_tabm()
    readout_methods = numerical_methods & {'build_tabicl_readout', 'build_tabpfn_readout'}
    if len(readout_methods) > 1:
        raise ValueError('一次训练程序只接入一种回归教师')
    if 'build_tabicl_readout' in readout_methods:
        session._prepare_tabular()
    if 'build_tabpfn_readout' in readout_methods:
        session._prepare_tabpfn_readout()
    classifier_methods=numerical_methods & {'build_tabicl_classifier','build_tabpfn_classifier','build_gas_baseline_ensemble'}
    if len(classifier_methods)>1:
        raise ValueError('一次训练程序只接入一种分类教师')
    if classifier_methods:
        session._prepare_tabular_classifier('tabicl' if 'build_tabicl_classifier' in classifier_methods else 'tabpfn',
                                            baseline='build_gas_baseline_ensemble' in classifier_methods)
    if numerical_methods & {'build_gas_repair', 'set_gas_policy'}:
        from harness.learning.training import gas_components
    if numerical_methods & {'build_regression_components', 'train_residual'}:
        from harness.learning.training.complete_training import prepare_runtime
        if 'train_residual' in numerical_methods:
            from harness.learning.training import residual_continuation
        prepare_runtime()
    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in ('numpy', 'math') or level:
            raise ImportError('仅允许数值模块')
        return np if name == 'numpy' else math
    safe = {name: getattr(__import__('builtins'), name) for name in
            ('range', 'len', 'min', 'max', 'abs', 'float', 'int', 'str', 'bool', 'dict', 'list', 'tuple',
             'enumerate', 'zip', 'sum', 'sorted', 'round', 'print', 'ValueError')}
    safe['__import__'] = importer
    namespace = {'__builtins__': safe, 'np': np, 'math': math}
    # 导入和随机数运行库在安装审计钩子前载入，避免把正常延迟加载误当源码访问。
    np.random.default_rng(0).permutation(2)
    # NumPy 的 unique 首次检查掩码时会加载 numpy.ma，须在训练程序启动前预热。
    np.unique(np.asarray([0, 1]))
    active = [True]
    def guard(event, arguments):
        if active[0] and (event == 'open' or event.startswith(('socket.', 'subprocess.')) or event in ('os.system', 'ctypes.dlopen')):
            raise PermissionError('训练程序不需要文件、网络或外部进程访问')
    compiled = compile(tree, '<continuation-program>', 'exec')
    sys.addaudithook(guard)
    try:
        exec(compiled, namespace)
        namespace['train'](session)
    finally:
        active[0] = False
    return session.audit()
