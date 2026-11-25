"""冻结TabPFN回归教师与真实原生父网的完整输出组合。

检查点保存压缩教师、原始源上下文、父网及Adam；作者的临时拟合缓存在
每次新进程首次使用时由同一冻结源上下文重建，不携带目标样本或pickle。
组合的非负父输出及其梯度复用已经核验的TabICL回归组合数学定义。
"""
from hashlib import sha256
import json
import os
from pathlib import Path
from platform import machine
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from .frozen_attention import _enable_frozen_cpu_inference
    from .tabular_readout import TabICLReadoutModel
    from .tabpfn_classifier import author_package
    from .quantized_teacher import MAX_TEACHER_BYTES, restore_teacher, teacher_content_sha256
    from .teacher_cache import frozen_regression_values, prefetch_teacher
    try:
        from .complete_checkpoint import FORMAT
    except ImportError:
        from .complete_model import FORMAT
else:
    from frozen_attention import _enable_frozen_cpu_inference
    from tabular_readout import TabICLReadoutModel
    from tabpfn_classifier import author_package
    from quantized_teacher import MAX_TEACHER_BYTES, restore_teacher, teacher_content_sha256
    from teacher_cache import frozen_regression_values, prefetch_teacher
    from complete_model import FORMAT


KIND = 'tabpfn-regression-parent'
RESOURCE_FORMAT = 'harness.tabpfn-regressor-resource.v1'
RESOURCE_FIELDS = ('teacher_asset', 'teacher_license', 'readout_source_x',
                   'readout_source_y', 'tabpfn_regressor_state')


def _array_sha256(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class TabPFNRegressorModel(TabICLReadoutModel):
    """继承回归组合的真实父梯度，单独核验资源、装载及教师缓存边界。"""

    def __init__(self, base, auxiliary_arrays):
        if hasattr(base, 'auxiliary_arrays'):
            raise ValueError('TabPFN回归组件不能剥离或嵌套已有完整父版')
        self.base = base
        self.auxiliary_arrays = {k: v.copy() for k, v in auxiliary_arrays.items()}
        expected_fields = {*RESOURCE_FIELDS, 'complete_state'}
        self._cpu_readout = None
        if 'bearing_cpu_readout' in self.auxiliary_arrays:
            expected_fields.add('bearing_cpu_readout')
        if set(self.auxiliary_arrays) not in (expected_fields, expected_fields | {'tabm_readout'}):
            raise ValueError('TabPFN回归资源字段不完整或含未知持久状态')
        for name in ('complete_state', 'tabpfn_regressor_state', 'teacher_license'):
            value = self.auxiliary_arrays[name]
            if value.shape != () or value.dtype.kind not in 'US':
                raise ValueError('回归描述和许可证必须是字符串标量')
        self.state = json.loads(str(self.auxiliary_arrays['complete_state']))
        self.teacher_state = json.loads(str(self.auxiliary_arrays['tabpfn_regressor_state']))
        teacher = self.teacher_state
        if self.state.get('format') != FORMAT or self.state.get('kind') != KIND:
            raise ValueError('未知TabPFN回归组合格式')
        if teacher.get('format') != RESOURCE_FORMAT or teacher.get('feature_view') != 'raw':
            raise ValueError('未知TabPFN回归资源格式或输入视图')
        if (base.meta['task'] != 'regression' or teacher.get('feature_paths') != base.meta['paths']):
            raise ValueError('教师原始特征列序与真实回归父网不符')
        if (type(teacher.get('n_estimators')) is not int or teacher['n_estimators'] != 8
                or type(teacher.get('random_state')) is not int or teacher['random_state'] != 42
                or teacher.get('fit_mode') != 'fit_with_cache'
                or teacher.get('inference_precision') != 'float32'):
            raise ValueError('回归教师须保持固定8估计器、seed42及float32缓存拟合配置')
        if (type(teacher.get('teacher_parameters')) is not int or teacher['teacher_parameters'] <= 0
                or not _valid_sha256(teacher.get('teacher_sha256'))):
            raise ValueError('回归教师来源权重身份或参数量无效')
        x, y = self.auxiliary_arrays['readout_source_x'], self.auxiliary_arrays['readout_source_y']
        if (x.dtype != np.float64 or y.dtype != np.float64 or x.ndim != 2 or not len(x)
                or x.shape[1] != len(base.meta['paths']) or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any()):
            raise ValueError('回归源上下文须为原始float64行及有限非负真实RUL')
        if (teacher.get('source_x_sha256') != _array_sha256(x)
                or teacher.get('source_y_sha256') != _array_sha256(y)):
            raise ValueError('冻结回归源上下文摘要不符')
        payload = self.auxiliary_arrays['teacher_asset']
        if (payload.dtype != np.uint8 or payload.ndim != 1 or not 0 < len(payload) <= MAX_TEACHER_BYTES
                or sha256(payload.tobytes()).hexdigest() != teacher.get('asset_sha256')):
            raise ValueError('冻结回归教师压缩资产身份不符')
        if not str(self.auxiliary_arrays['teacher_license']).strip():
            raise ValueError('回归教师缺少原模型许可证')
        if (self.state.get('teacher_sha256') != teacher['teacher_sha256']
                or self.state.get('teacher_asset_sha256') != teacher['asset_sha256']
                or self.state.get('source_context_rows') != len(y)
                or self.state.get('training_objective') != 'complete_prediction'
                or self.state.get('context_updates') != 'frozen_source_context'):
            raise ValueError('完整回归描述与冻结资源不一致')
        self._teacher = None
        spec_cpu = self.state.get('bearing_cpu_readout')
        if (spec_cpu is None) != ('bearing_cpu_readout' not in self.auxiliary_arrays):
            raise ValueError('轴承 CPU 读出与检查点字段不对应')
        if spec_cpu is not None:
            payload = self.auxiliary_arrays['bearing_cpu_readout']
            if (base.meta.get('profile') != 'B' or spec_cpu.get('format') != 'bearing-cpu-rbf-v1'
                    or payload.dtype != np.uint8 or payload.ndim != 1
                    or _array_sha256(payload) != spec_cpu.get('sha256')):
                raise ValueError('轴承 CPU 读出格式或摘要无效')
            self._cpu_readout = self._cpu_readout_class()(payload,len(base.meta['paths']))
        self._training_teacher = {}
        self._prediction_cache = {}
        self._tabm = None
        self._tabm_training = {}
        self._support_bounds = None
        self._head_limit = None
        self._training_spread = {}
        self._health_cap = None
        self._grid_readout = None
        if 'bearing_grid_readout' in self.state:
            self.load_grid_readout(self.state['bearing_grid_readout'])
        spec = self.state.get('tabm_readout')
        if (spec is None) != ('tabm_readout' not in self.auxiliary_arrays):
            raise ValueError('TabM 组合描述与权重不对应')
        if spec is not None:
            payload = self.auxiliary_arrays['tabm_readout']
            if spec['weight'] != .5 or _array_sha256(payload) != spec['sha256']:
                raise ValueError('TabM 组合身份与检查点不符')
            if __package__:
                from .tabm_readout import TabMReadout
            else:
                from tabm_readout import TabMReadout
            self._tabm = TabMReadout(payload, len(base.meta['paths']))
            if 'source_support' in spec:
                self._load_support(spec['source_support'])
            if 'head_support' in spec:
                self._load_head_support(spec['head_support'])
            if 'health_cap' in spec:
                self.load_health_cap(spec['health_cap'])
        self._round_cache = None
        self.round_cache_hits = {'training': 0, 'evaluation': 0}
        self.validate()

    def validate(self):
        # 冻结大数组仅在构造/恢复时核验；每个真实父梯度步只查可变父状态。
        self.base.validate()
        weight = self.state['parent_weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError('真实回归父网权重必须在(0,1]内')

    def _load_teacher(self):
        if self._teacher is not None:
            return
        import torch
        threads = os.environ.get('ZHIXUN_SDK_PREDICT_THREADS')
        torch.set_num_threads(int(threads) if threads in ('2','4') else 1)
        torch.manual_seed(42)
        variables = ('HF_HUB_OFFLINE', 'TABPFN_NO_BROWSER', 'SKB_DATA_DIRECTORY', 'TABPFN_MODEL_CACHE_DIR')
        previous = {name: os.environ.get(name) for name in variables}
        try:
            with TemporaryDirectory(prefix='harness-tabpfn-regressor-') as temporary:
                os.environ.update(HF_HUB_OFFLINE='1', TABPFN_NO_BROWSER='1',
                                  SKB_DATA_DIRECTORY=temporary, TABPFN_MODEL_CACHE_DIR=temporary)
                package = author_package()
                if __package__:
                    from .bearing_fit_state import load
                else:
                    from bearing_fit_state import load
                path = Path(temporary) / 'tabpfn-v3.5-fast-20260909.safetensors'
                restore_teacher(self.auxiliary_arrays['teacher_asset'], path)
                # 文件头的映射顺序可跨进程改变，展开后核数值内容而非容器字节。
                decoded_sha = self.teacher_state.get('decoded_teacher_content_sha256')
                if decoded_sha is not None and teacher_content_sha256(path) != decoded_sha:
                    raise ValueError('回归教师展开内容摘要不符')
                prepared = load(self.teacher_state,package,path)
                if prepared is not None:
                    _enable_frozen_cpu_inference(prepared)
                    self._teacher = prepared
                    self.fit_state_restored = True
                    return
                model = package.TabPFNRegressor(model_path=path, n_estimators=8, device='cpu',
                    random_state=42, fit_mode='fit_with_cache', n_preprocessing_jobs=1,
                    inference_precision=torch.float32)
                x, y = self.auxiliary_arrays['readout_source_x'], self.auxiliary_arrays['readout_source_y']
                model.fit(x.astype(np.float32), y.copy())
                self.fit_state_restored = False
                if model.n_features_in_ != x.shape[1] or model.n_estimators_ != 8:
                    raise ValueError('作者回归器的实际输入或估计器数不符')
                _enable_frozen_cpu_inference(model)
            self._teacher = model
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _raw(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.meta['paths']) or not np.isfinite(raw).all():
            raise ValueError('TabPFN回归组合输入无效')
        return raw

    def bind_round_cache(self, folder):
        """仅训练宿主可绑定临时目录；独立预测实例从交付检查点自行推理。"""
        self._round_cache = folder

    @staticmethod
    def _cpu_readout_class():
        if __package__:
            from .bearing_cpu_readout import BearingCPUReadout
        else:
            from bearing_cpu_readout import BearingCPUReadout
        return BearingCPUReadout

    def _teacher_predict(self, rows):
        if self._cpu_readout is not None:
            return self._cpu_readout.predict(rows)
        self._load_teacher()
        backend = getattr(self._teacher, '_zhixun_cpu_backend', None)
        if backend is None:
            return self._teacher.predict(rows)
        active = backend.active
        token = active.set(True)
        try:
            return self._teacher.predict(rows)
        finally:
            active.reset(token)

    def _prefetch_batch(self, raw, cache, role):
        if self._cpu_readout is not None:
            prefetch_teacher(cache,raw,self._teacher_predict,nonnegative=True,batch_size=256)
            return
        if self._round_cache is None:
            prefetch_teacher(cache, raw, self._teacher_predict, nonnegative=True, batch_size=128)
            return
        namespace = '\n'.join(('regression-cpu-v1', role, machine(),
                               os.environ.get('ZHIXUN_SDK_PREDICT_THREADS', '1'),
                               str(self.auxiliary_arrays['tabpfn_regressor_state'])))
        if frozen_regression_values(self._round_cache, namespace, raw, cache, self._teacher_predict,
                                    training=role == 'training'):
            self.round_cache_hits[role] += 1

    def prepare_training(self, raw):
        """宿主只预计算本轮合法训练行；预测缓存及作者拟合缓存不进入检查点。"""
        raw = self._raw(raw)
        self._prefetch_batch(raw, self._training_teacher, 'training')
        if self._tabm is not None:
            prefetch_teacher(self._tabm_training, raw, self._tabm.predict, nonnegative=True, batch_size=256)
        if self._head_limit is not None:
            self.prepare_head_spread(raw)

    def attach_tabm(self, payload, model, cached, source_rows):
        if self._tabm is not None:
            raise ValueError('父产物已经包含 TabM 读出')
        self._tabm, self._tabm_training = model, cached
        self.auxiliary_arrays['tabm_readout'] = payload
        self.state['tabm_readout'] = {'weight':.5, 'sha256':_array_sha256(payload), 'source_rows':source_rows}
        self.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(self.state,sort_keys=True,separators=(',',':')))

    def _tabm_train_prediction(self, raw):
        return np.asarray([self._tabm_training[row.tobytes()] for row in np.asarray(raw,dtype=np.float64)])

    def _load_support(self, spec):
        bounds = np.asarray(spec['bounds'], dtype=np.float32)
        if (spec.get('format') != 'source-minmax-v1' or type(spec.get('rows')) is not int or spec['rows'] < 1
                or bounds.shape != (2, len(self.meta['paths'])) or not np.isfinite(bounds).all()
                or np.any(bounds[0] > bounds[1])):
            raise ValueError('TabM 源特征范围与实际训练行或输入列不一致')
        self._support_bounds = bounds

    def add_source_support(self, raw):
        """只从当前训练投影冻结逐列范围，旧产物与后续接续保留原组合语义。"""
        if self._tabm is None or self._support_bounds is not None:
            raise ValueError('源范围只可附加到尚未配置范围的 TabM 回归父产物')
        x = self._raw(raw).astype(np.float32)
        spec = {'format': 'source-minmax-v1', 'rows': len(x), 'bounds': [x.min(0).tolist(), x.max(0).tolist()]}
        self._load_support(spec)
        self.state['tabm_readout']['source_support'] = spec
        self.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(self.state, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def prepare_head_spread(self, raw):
        raw = self._raw(raw)
        if self._tabm is None or self._support_bounds is None or not len(raw):
            raise ValueError('多头分歧需要已有源范围的 TabM 父版和合法训练行')
        _,spread = self._tabm.predict_with_spread(raw)
        self._training_spread = {row.tobytes():value for row,value in zip(raw,spread,strict=True)}
        return {'format':'tabm-head-support-v1','rows':len(raw),'max_std':float(spread.max())}

    def _load_head_support(self, spec):
        limit = spec.get('max_std')
        if (self._support_bounds is None or spec.get('format')!='tabm-head-support-v1'
                or type(spec.get('rows')) is not int or spec['rows']<1 or type(limit) not in (int,float)
                or not np.isfinite(limit) or limit<0):
            raise ValueError('TabM 源训练头分歧描述无效')
        self._head_limit = limit

    def add_head_support(self, spec):
        if self._head_limit is not None:
            raise ValueError('父版已经包含多头分歧范围')
        self._load_head_support(spec)
        self.state['tabm_readout']['head_support'] = spec
        self.auxiliary_arrays['complete_state'] = np.asarray(json.dumps(self.state,sort_keys=True,separators=(',',':'),allow_nan=False))

    def load_health_cap(self,spec):
        if __package__:
            from .bearing_health_readout import BearingHealthReadout
        else:
            from bearing_health_readout import BearingHealthReadout
        self._health_cap=BearingHealthReadout(spec,self.meta['paths'])

    def _tabm_weights(self, raw, spread=None):
        if self._support_bounds is None:
            return np.full(len(raw), .5)
        low, high = self._support_bounds
        varying = low < high
        x = np.asarray(raw, dtype=np.float32)[:, varying]
        inside = varying.any() & ((low[varying] <= x) & (x <= high[varying])).all(axis=1)
        if self._head_limit is not None:
            if spread is None:
                spread = np.asarray([self._training_spread[row.tobytes()] for row in self._raw(raw)])
            return np.where(inside,.875,np.where(spread<=self._head_limit,.625,.5))
        return np.where(inside, .75, .5)

    def load_grid_readout(self, spec):
        if __package__:
            from .bearing_grid_readout import GridReadout
        else:
            from bearing_grid_readout import GridReadout
        self._grid_readout = GridReadout(spec,self.meta['paths'])

    def training_before_grid(self, raw):
        values = super().training_predictions(raw)
        if self._tabm is None:
            return values
        weight = self._tabm_weights(raw)
        result=(1 - weight)*values + weight*self._tabm_train_prediction(raw)
        return result if self._health_cap is None else np.minimum(result,self._health_cap.upper(raw))

    def training_predictions(self, raw):
        values = self.training_before_grid(raw)
        return values if self._grid_readout is None else self._grid_readout.apply(raw,values)[0]

    def _continuation_before_grid(self, raw, logits):
        values, derivative = super().continuation_output(raw, logits)
        if self._tabm is None:
            return values, derivative
        scaled = (self._tabm_train_prediction(raw)-self.meta['target_mean']) / self.meta['target_scale']
        weight = self._tabm_weights(raw)[:, None]
        result=(1 - weight)*values + weight*scaled[:, None]
        derivative=(1 - weight)*derivative
        if self._health_cap is not None:
            ceiling=(self._health_cap.upper(raw)[:,None]-self.meta['target_mean'])/self.meta['target_scale']
            derivative=derivative*(result<ceiling)
            result=np.minimum(result,ceiling)
        return result,derivative

    def continuation_output(self, raw, logits):
        values, derivative = self._continuation_before_grid(raw, logits)
        if self._grid_readout is None:
            return values, derivative
        scale, center = self.meta['target_scale'], self.meta['target_mean']
        minutes = values[:,0]*scale + center
        calibrated, gradient = self._grid_readout.apply(raw,minutes)
        return (values + ((calibrated-minutes)/scale)[:,None], derivative*gradient[:,None])

    def predict_batch(self, raw):
        """内部评价批量预取同一冻结教师；不在查询之间拟合统计或更新参数。"""
        raw = self._raw(raw)
        self._prefetch_batch(raw, self._prediction_cache, 'evaluation')
        return self.predict(raw)

    def _predict_before_grid(self, raw):
        raw = self._raw(raw)
        prefetch_teacher(self._prediction_cache, raw, self._teacher_predict, nonnegative=True, batch_size=1)
        teacher = np.asarray([self._prediction_cache[row.tobytes()] for row in raw])
        alpha = self.state['parent_weight']
        parent = alpha * self.base.predict(raw) + (1 - alpha) * teacher
        if self._tabm is None:
            return parent
        if self._head_limit is None:
            values,spread = self._tabm.predict(raw),None
        else:
            values,spread = self._tabm.predict_with_spread(raw)
        weight = self._tabm_weights(raw,spread)
        result=(1 - weight)*parent + weight*values
        return result if self._health_cap is None else np.minimum(result,self._health_cap.upper(raw))

    def predict(self, raw):
        raw = self._raw(raw)
        result = self._predict_before_grid(raw)
        return result if self._grid_readout is None else self._grid_readout.apply(raw,result)[0]


def from_regressor_asset(base, asset_path, *, parent_weight=.25):
    """首次组件只从真实原生父网新建；禁止取旧组合.base冒充完整模型接续。"""
    if hasattr(base, 'auxiliary_arrays'):
        raise ValueError('新回归组件须从原生父版开始，不能剥离已有完整组合')
    if isinstance(parent_weight, bool):
        raise ValueError('真实回归父网权重不能是布尔值')
    with np.load(asset_path, allow_pickle=False) as saved:
        if set(saved.files) != set(RESOURCE_FIELDS):
            raise ValueError('回归共享资源须仅含教师、许可、源上下文和描述')
        arrays = {field: saved[field].copy() for field in RESOURCE_FIELDS}
    teacher = json.loads(str(arrays['tabpfn_regressor_state']))
    state = {'format': FORMAT, 'kind': KIND, 'parent_weight': float(parent_weight),
        'training_objective': 'complete_prediction', 'teacher_parameters': teacher['teacher_parameters'],
        'teacher_sha256': teacher['teacher_sha256'], 'teacher_asset_sha256': teacher['asset_sha256'],
        'source_context_rows': len(arrays['readout_source_y']), 'feature_view': 'raw',
        'context_updates': 'frozen_source_context', 'initial_component_origin': 'new_component_from_native_parent'}
    arrays['complete_state'] = np.asarray(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return TabPFNRegressorModel(base, arrays)
