"""冻结TabPFN回归教师与真实原生父网的完整输出组合。

检查点保存压缩教师、原始源上下文、父网及Adam；作者的临时拟合缓存在
每次新进程首次使用时由同一冻结源上下文重建，不携带目标样本或pickle。
组合的非负父输出及其梯度复用已经核验的TabICL回归组合数学定义。
"""
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from .tabular_readout import TabICLReadoutModel
    from .tabpfn_classifier import author_package
    from .quantized_teacher import restore_teacher, teacher_content_sha256
    from .teacher_cache import prefetch_teacher
    try:
        from .complete_checkpoint import FORMAT
    except ImportError:
        from .complete_model import FORMAT
else:
    from tabular_readout import TabICLReadoutModel
    from tabpfn_classifier import author_package
    from quantized_teacher import restore_teacher, teacher_content_sha256
    from teacher_cache import prefetch_teacher
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
        if set(self.auxiliary_arrays) != {*RESOURCE_FIELDS, 'complete_state'}:
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
        if (payload.dtype != np.uint8 or payload.ndim != 1 or not 0 < len(payload) <= 54 * 1024**2
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
        self._training_teacher = {}
        self._prediction_cache = {}
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
        torch.set_num_threads(1)
        torch.manual_seed(42)
        variables = ('HF_HUB_OFFLINE', 'TABPFN_NO_BROWSER', 'SKB_DATA_DIRECTORY', 'TABPFN_MODEL_CACHE_DIR')
        previous = {name: os.environ.get(name) for name in variables}
        try:
            with TemporaryDirectory(prefix='harness-tabpfn-regressor-') as temporary:
                os.environ.update(HF_HUB_OFFLINE='1', TABPFN_NO_BROWSER='1',
                                  SKB_DATA_DIRECTORY=temporary, TABPFN_MODEL_CACHE_DIR=temporary)
                package = author_package()
                path = Path(temporary) / 'tabpfn-v3.5-fast-20260909.safetensors'
                restore_teacher(self.auxiliary_arrays['teacher_asset'], path)
                # 文件头的映射顺序可跨进程改变，展开后核数值内容而非容器字节。
                decoded_sha = self.teacher_state.get('decoded_teacher_content_sha256')
                if decoded_sha is not None and teacher_content_sha256(path) != decoded_sha:
                    raise ValueError('回归教师展开内容摘要不符')
                model = package.TabPFNRegressor(model_path=path, n_estimators=8, device='cpu',
                    random_state=42, fit_mode='fit_with_cache', n_preprocessing_jobs=1,
                    inference_precision=torch.float32)
                x, y = self.auxiliary_arrays['readout_source_x'], self.auxiliary_arrays['readout_source_y']
                model.fit(x.astype(np.float32), y.copy())
                if model.n_features_in_ != x.shape[1] or model.n_estimators_ != 8:
                    raise ValueError('作者回归器的实际输入或估计器数不符')
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

    def prepare_training(self, raw):
        """宿主只预计算本轮合法训练行；预测缓存及作者拟合缓存不进入检查点。"""
        raw = self._raw(raw)
        self._load_teacher()
        prefetch_teacher(self._training_teacher, raw, self._teacher.predict, nonnegative=True, batch_size=128)

    def predict_batch(self, raw):
        """内部评价批量预取同一冻结教师；不在查询之间拟合统计或更新参数。"""
        raw = self._raw(raw)
        self._load_teacher()
        prefetch_teacher(self._prediction_cache, raw, self._teacher.predict, nonnegative=True, batch_size=128)
        return self.predict(raw)

    def predict(self, raw):
        raw = self._raw(raw)
        self._load_teacher()
        prefetch_teacher(self._prediction_cache, raw, self._teacher.predict, nonnegative=True, batch_size=1)
        teacher = np.asarray([self._prediction_cache[row.tobytes()] for row in raw])
        alpha = self.state['parent_weight']
        return alpha * self.base.predict(raw) + (1 - alpha) * teacher


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
