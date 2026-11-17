"""冻结TabICLv2原始特征读出与可训练父网络组合，所有持久状态保存于NPZ。

教师只用已验证的源上下文；父梯度对完整组合误差计算。训练期教师输出
由宿主在数值程序启动前缓存，避免每个小批次重复推理或在程序内访问文件。
"""
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    try:
        from .complete_checkpoint import CompleteModel, FORMAT
    except ImportError:
        from .complete_model import CompleteModel, FORMAT
else:
    from complete_model import CompleteModel, FORMAT


KIND = 'tabicl-parent-combination'


def author_package():
    """正式独立包直接导入随包作者代码；研发环境复用已有安装，不自动下载。"""
    # 平台也可能按文件路径启动baselines/predict.py，此时sys.path只含baselines目录。
    bundled_root=Path(__file__).resolve().parents[1]
    if not (__package__ or '').startswith('harness.') and (bundled_root/'tabicl/__init__.py').is_file():
        if str(bundled_root) not in sys.path:sys.path.insert(0,str(bundled_root))
    try:
        return importlib.import_module('tabicl')
    except ModuleNotFoundError as exc:
        if exc.name != 'tabicl' or not (__package__ or '').startswith('harness.'):
            raise
        from harness.paths import workspace_root
        runtime = workspace_root()/'engineering/.environments/tabicl-runtime'
        if not (runtime/'tabicl/__init__.py').is_file():
            raise RuntimeError('缺少已固定版本的TabICL运行库') from exc
        sys.path.insert(0, str(runtime))
        return importlib.import_module('tabicl')


class TabICLReadoutModel(CompleteModel):
    def __init__(self, base, auxiliary_arrays):
        self.base = base
        self.auxiliary_arrays = {k: v.copy() for k, v in auxiliary_arrays.items()}
        self.state = json.loads(str(self.auxiliary_arrays['complete_state']))
        self.teacher_state = json.loads(str(self.auxiliary_arrays['tabular_readout_state']))
        if self.state.get('format') != FORMAT or self.state.get('kind') != KIND:
            raise ValueError('未知TabICL父版组合格式')
        if self.teacher_state['feature_view'] != 'raw' or base.meta['task'] != 'regression':
            raise ValueError('当前接入仅支持原始特征回归读出')
        x, y = self.auxiliary_arrays['readout_source_x'], self.auxiliary_arrays['readout_source_y']
        if x.ndim != 2 or x.shape[1] != len(base.meta['paths']) or y.shape != (len(x),):
            raise ValueError('教师源上下文与父输入语义不符')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('教师源上下文包含非有限值')
        for field in self.teacher_state['tensor_fields'].values():
            if field not in self.auxiliary_arrays or self.auxiliary_arrays[field].dtype.kind != 'f':
                raise ValueError('教师数值权重缺失或类型不符')
        self._teacher = None
        self._training_teacher = {}
        self._prediction_cache = {}
        self.validate()

    def validate(self):
        # 每个父梯度步只检查可变父状态；冻结教师的大数组在装载及宿主审计时核验。
        self.base.validate()
        weight = self.state['parent_weight']
        if not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError('父预测权重必须在(0,1]内')

    def normalized(self, raw):
        return self.base.normalized(raw)

    def _load_teacher(self):
        if self._teacher is not None:
            return
        import torch
        package = author_package()
        torch.set_num_threads(1)
        weights = {name: torch.from_numpy(self.auxiliary_arrays[field].copy())
                   for name, field in self.teacher_state['tensor_fields'].items()}
        with TemporaryDirectory(prefix='harness-tabicl-runtime-') as temporary:
            path = Path(temporary)/'teacher.ckpt'
            torch.save({'state_dict': weights, 'config': self.teacher_state['model_config']}, path)
            model = package.TabICLRegressor(model_path=path, n_estimators=self.teacher_state['n_estimators'],
                batch_size=1, kv_cache=True, device='cpu', n_jobs=1, allow_auto_download=False,
                random_state=42, use_amp=False)
            model.fit(self.auxiliary_arrays['readout_source_x'].astype(np.float32),
                      self.auxiliary_arrays['readout_source_y'])
        self._teacher = model

    def prepare_training(self, raw):
        """只预测合法训练行；原始标签不传给教师的预测步骤，缓存不写入产物。"""
        self._load_teacher()
        raw = np.asarray(raw, dtype=np.float64)
        for start in range(0, len(raw), 128):
            batch = raw[start:start+128]
            values = np.maximum(0., self._teacher.predict(batch.astype(np.float32)))
            for row, value in zip(batch, values, strict=True):
                self._training_teacher[row.tobytes()] = float(value)

    def _training_values(self, raw):
        try:
            return np.asarray([self._training_teacher[row.tobytes()] for row in np.asarray(raw, dtype=np.float64)])
        except KeyError as exc:
            raise ValueError('训练行不在宿主预计算的合法资料内') from exc

    def continuation_output(self, raw, logits):
        teacher = self._training_values(raw)
        native = logits[:, 0]*self.meta['target_scale']+self.meta['target_mean']
        alpha = self.state['parent_weight']
        combined = alpha*np.maximum(0., native)+(1-alpha)*teacher
        scaled = (combined-self.meta['target_mean'])/self.meta['target_scale']
        return scaled[:, None], (alpha*(native > 0))[:, None]

    def training_predictions(self, raw):
        alpha = self.state['parent_weight']
        return alpha*self.base.predict(raw)+(1-alpha)*self._training_values(raw)

    def predict(self, raw):
        self._load_teacher()
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.meta['paths']) or not np.isfinite(raw).all():
            raise ValueError('TabICL组合输入无效')
        teacher = []
        for row in raw:
            key = row.tobytes()
            if key not in self._prediction_cache:
                value = float(self._teacher.predict(row[None, :].astype(np.float32))[0])
                if not np.isfinite(value):
                    raise ValueError('教师预测不是有限值')
                self._prediction_cache[key] = max(0., value)
            teacher.append(self._prediction_cache[key])
        alpha = self.state['parent_weight']
        return alpha*self.base.predict(raw)+(1-alpha)*np.asarray(teacher)


def from_asset(base, asset_path, *, parent_weight):
    """从宿主核验的共享原型提取教师状态；原型里的旧父网络不参与迁移。"""
    with np.load(asset_path, allow_pickle=False) as saved:
        teacher = json.loads(str(saved['tabular_readout_state']))
        fields = list(teacher['tensor_fields'].values())+['readout_source_x','readout_source_y','tabular_readout_state']
        arrays = {field: saved[field].copy() for field in fields}
    state = {'format': FORMAT, 'kind': KIND, 'parent_weight': float(parent_weight),
        'training_objective': 'complete_prediction', 'teacher_parameters': teacher['teacher_parameters'],
        'teacher_sha256': teacher['teacher_sha256'], 'source_context_rows': len(arrays['readout_source_y']),
        'feature_view': 'raw', 'context_updates': 'frozen_source_context'}
    arrays['complete_state'] = np.asarray(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return TabICLReadoutModel(base, arrays)
