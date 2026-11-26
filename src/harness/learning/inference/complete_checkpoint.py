"""完整预测链的数值检查点；保留原生权重、Adam及冻结输入处理/残差网络。

本模块只依赖NumPy，可随独立预测入口运行。旧残差产物未保存优化器，
因此迁移后保持该网络冻结；续训对完整输出求导并更新真实父网络参数。
"""
from collections import defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile

import numpy as np


FORMAT = 'harness.complete-checkpoint.v1'
PARAMETERS = ('w1', 'b1', 'w2', 'b2')
BASE_KEYS = {*PARAMETERS, *('adam_first_' + k for k in PARAMETERS),
             *('adam_second_' + k for k in PARAMETERS), 'mean', 'scale', 'metadata'}


def is_complete(path):
    with np.load(path, allow_pickle=False) as arrays:
        return 'complete_state' in arrays.files


def with_input_alignment(model):
    if 'engine_input_alignment' not in model.meta or hasattr(model, 'unaligned_engine'):
        return model
    if __package__:
        from .engine_setting_alignment import AlignedEngine
    else:
        from engine_setting_alignment import AlignedEngine
    return AlignedEngine(model)


def delivery_model(model):
    if 'gas_partial_context' in model.meta and not hasattr(model, 'context_base'):
        if __package__:
            from .gas_partial_readout import GasContextModel
        else:
            from gas_partial_readout import GasContextModel
        return GasContextModel(model)
    return with_input_alignment(model)


def prediction_sources(package, checkpoint):
    """为实际模型选择加载入口；完整模型的预测代码不依赖研究目录或旧pickle。"""
    files = {'baselines/' + name: package.read('baselines/' + name)
             for name in ('__init__.py', 'model.py', 'data.py', 'predict.py')}
    with np.load(checkpoint, allow_pickle=False) as arrays:
        complete = 'complete_state' in arrays.files
        metadata = json.loads(str(arrays['metadata']))
        aligned = 'engine_input_alignment' in metadata
        partial = 'gas_partial_context' in metadata
    if complete or aligned or partial:
        files['baselines/core_model.py'] = files['baselines/model.py']
        files['baselines/complete_model.py'] = Path(__file__).read_bytes()
        files['baselines/gas_views.py'] = Path(__file__).with_name('gas_views.py').read_bytes()
        files['baselines/model.py'] = ("if __package__:\n"
            "    from .core_model import Model as BaseModel\n"
            "    from .complete_model import load_checkpoint\n"
            "else:\n"
            "    from core_model import Model as BaseModel\n"
            "    from complete_model import load_checkpoint\n\n"
            "class Model(BaseModel):\n"
            "    @classmethod\n"
            "    def load(cls, path):\n"
            "        return load_checkpoint(path, BaseModel)\n").encode('utf-8')
        with np.load(checkpoint, allow_pickle=False) as arrays:
            state = json.loads(str(arrays['complete_state'])) if complete else {}
            kind = state.get('kind')
        if kind == 'engine-source-distribution':
            files['baselines/engine_source_forest.py'] = Path(__file__).with_name('engine_source_forest.py').read_bytes()
        if aligned:
            files['baselines/engine_setting_alignment.py'] = Path(__file__).with_name('engine_setting_alignment.py').read_bytes()
        if partial:
            for name in ('gas_partial_readout.py','gas_device_context.py'):
                files['baselines/'+name] = Path(__file__).with_name(name).read_bytes()
        if 'engine_condition_density' in state:
            files['baselines/engine_condition_density.py'] = Path(__file__).with_name('engine_condition_density.py').read_bytes()
        if 'engine_neighbors' in state:
            files['baselines/engine_neighbors.py'] = Path(__file__).with_name('engine_neighbors.py').read_bytes()
        if 'bearing_grid_readout' in state:
            files['baselines/bearing_grid_readout.py'] = Path(__file__).with_name('bearing_grid_readout.py').read_bytes()
            if any('health_cap' in item for item in state['bearing_grid_readout']['models']):
                files['baselines/bearing_health_readout.py'] = Path(__file__).with_name('bearing_health_readout.py').read_bytes()
            if any('terminal' in item for item in state['bearing_grid_readout']['models']):
                files['baselines/bearing_terminal_readout.py'] = Path(__file__).with_name('bearing_terminal_readout.py').read_bytes()
        if kind in ('tabpfn-classification-parent', 'tabpfn-regression-parent'):
            from .tabpfn_classifier import prediction_dependencies
            for name in ('tabular_readout.py','tabular_classifier.py','tabpfn_classifier.py',
                         'quantized_teacher.py','probability_mixture.py','teacher_cache.py','frozen_attention.py'):
                files['baselines/'+name]=Path(__file__).with_name(name).read_bytes()
            if kind=='tabpfn-classification-parent':
                files['baselines/gas_fit_state.py']=Path(__file__).with_name('gas_fit_state.py').read_bytes()
                files['baselines/gas_drift_readout.py']=Path(__file__).with_name('gas_drift_readout.py').read_bytes()
                if 'device_context' in state:
                    files['baselines/gas_device_context.py'] = Path(__file__).with_name('gas_device_context.py').read_bytes()
                with np.load(checkpoint,allow_pickle=False) as arrays:
                    if 'gas_svm_vectors' in arrays.files:
                        files['baselines/gas_toluene_readout.py']=Path(__file__).with_name('gas_toluene_readout.py').read_bytes()
                    if 'kinetic_readout' in state:
                        files['baselines/gas_kinetic_support.py']=Path(__file__).with_name('gas_kinetic_support.py').read_bytes()
            if kind == 'tabpfn-regression-parent':
                files['baselines/bearing_fit_state.py'] = Path(__file__).with_name('bearing_fit_state.py').read_bytes()
                if 'bearing_cpu_readout' in state:
                    files['baselines/bearing_cpu_readout.py'] = Path(__file__).with_name('bearing_cpu_readout.py').read_bytes()
                files['baselines/tabpfn_regressor.py'] = Path(__file__).with_name('tabpfn_regressor.py').read_bytes()
                with np.load(checkpoint, allow_pickle=False) as arrays:
                    has_tabm = 'tabm_readout' in arrays.files
                if has_tabm:
                    from importlib.metadata import distribution
                    files['baselines/tabm_readout.py'] = Path(__file__).with_name('tabm_readout.py').read_bytes()
                    if 'health_cap' in state.get('tabm_readout',{}):
                        files['baselines/bearing_health_readout.py'] = Path(__file__).with_name('bearing_health_readout.py').read_bytes()
                    for dependency in ('tabm', 'rtdl_num_embeddings'):
                        dist = distribution(dependency)
                        for item in dist.files:
                            if str(item)==dependency+'.py' or 'licenses' in item.parts:
                                files[str(item).replace('\\','/')] = dist.locate_file(item).read_bytes()
            files.update(prediction_dependencies())
            with np.load(checkpoint,allow_pickle=False) as arrays:
                files['TabPFN-model-LICENSE']=str(arrays['teacher_license']).encode('utf-8')
        if 'engine_parent_support' in state:
            files['baselines/engine_parent_support.py'] = Path(__file__).with_name('engine_parent_support.py').read_bytes()
        if kind in ('tabicl-parent-combination','tabicl-classification-parent'):
            from .tabular_readout import author_package
            files['baselines/tabular_readout.py'] = Path(__file__).with_name('tabular_readout.py').read_bytes()
            files['baselines/teacher_cache.py'] = Path(__file__).with_name('teacher_cache.py').read_bytes()
            if kind == 'tabicl-classification-parent':
                files['baselines/tabular_classifier.py'] = Path(__file__).with_name('tabular_classifier.py').read_bytes()
            package_root = Path(author_package().__file__).parent
            for source in package_root.rglob('*.py'):
                files['tabicl/'+source.relative_to(package_root).as_posix()] = source.read_bytes()
            licenses = list(package_root.parent.glob('tabicl-*.dist-info/licenses/LICENSE'))
            if len(licenses) != 1:
                raise ValueError('须保留固定作者库的许可证来源')
            files['TabICL-LICENSE'] = licenses[0].read_bytes()
            # 独立预测不能依赖研发PYTHONPATH；复用已安装的轻量纯Python依赖和许可。
            runtime_root = package_root.parent
            for dependency in ('einops', 'huggingface_hub', 'tqdm'):
                source_root = runtime_root/dependency
                metadata_roots = list(runtime_root.glob(dependency+'-*.dist-info'))
                if not source_root.is_dir() or len(metadata_roots) != 1:
                    raise ValueError('缺少固定版本的独立预测依赖：'+dependency)
                for source in source_root.rglob('*'):
                    if source.is_file() and source.suffix in ('.py','.pyi','.json'):
                        files[dependency+'/'+source.relative_to(source_root).as_posix()] = source.read_bytes()
                for source in metadata_roots[0].rglob('*'):
                    if source.is_file() and (source.name in ('METADATA','WHEEL','LICENSE','LICENSE.txt') or
                                             'licenses' in source.parts):
                        files[source.relative_to(runtime_root).as_posix()] = source.read_bytes()
    return files


def load_checkpoint(path, base_class):
    """原NPZ使用原加载器；扩展NPZ显式恢复全部组件，禁止静默丢弃附加状态。"""
    if not is_complete(path):
        return delivery_model(base_class.load(path))
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
        raise ValueError('完整模型必须是64 MiB以内的普通文件')
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist()]
        if len(names) != len(set(names)) or sum(item.file_size for item in archive.infolist()) > 64 * 1024**2:
            raise ValueError('完整检查点目录或展开大小无效')
    with np.load(path, allow_pickle=False) as arrays:
        if not BASE_KEYS <= set(arrays.files):
            raise ValueError('缺少真实父模型状态')
        state = json.loads(str(arrays['complete_state']))
        if state.get('kind') == 'gas-source-vote':
            if __package__:
                from .gas_views import pair_model
            else:
                from gas_views import pair_model
            base_class = pair_model(base_class)
        base = base_class.__new__(base_class)
        base.meta = json.loads(str(arrays['metadata']))
        base.params = {k: arrays[k].copy() for k in PARAMETERS}
        base.first = {k: arrays['adam_first_' + k].copy() for k in PARAMETERS}
        base.second = {k: arrays['adam_second_' + k].copy() for k in PARAMETERS}
        base.mean, base.scale = arrays['mean'].copy(), arrays['scale'].copy()
        extra = {k: arrays[k].copy() for k in arrays.files if k not in BASE_KEYS}
    if state.get('kind') == 'tabicl-parent-combination':
        if __package__:
            from .tabular_readout import TabICLReadoutModel
        else:
            from tabular_readout import TabICLReadoutModel
        model_class = TabICLReadoutModel
    elif state.get('kind') == 'tabicl-classification-parent':
        if __package__:
            from .tabular_classifier import TabICLClassifierModel
        else:
            from tabular_classifier import TabICLClassifierModel
        model_class = TabICLClassifierModel
    elif state.get('kind') == 'tabpfn-classification-parent':
        if __package__:
            from .tabpfn_classifier import TabPFNClassifierModel
        else:
            from tabpfn_classifier import TabPFNClassifierModel
        model_class = TabPFNClassifierModel
    elif state.get('kind') == 'tabpfn-regression-parent':
        if __package__:
            from .tabpfn_regressor import TabPFNRegressorModel
        else:
            from tabpfn_regressor import TabPFNRegressorModel
        model_class = TabPFNRegressorModel
    elif state.get('kind') == 'engine-source-distribution':
        if __package__:
            from .engine_source_forest import SourceForestModel
        else:
            from engine_source_forest import SourceForestModel
        model_class = SourceForestModel
    elif state.get('kind') == 'bearing-native-grid-parent':
        model_class = NativeBearingGridModel
    else:
        model_class = GasVoteModel if state.get('kind') == 'gas-source-vote' else CompleteModel
    return delivery_model(model_class(base, extra))


class CompleteModel:
    def __init__(self, base, auxiliary_arrays):
        self.base = base
        self.auxiliary_arrays = {k: v.copy() for k, v in auxiliary_arrays.items()}
        self.state = json.loads(str(self.auxiliary_arrays['complete_state']))
        if self.state.get('format') != FORMAT or base.meta['task'] != 'regression':
            raise ValueError('当前完整状态仅支持寿命回归及显式格式')
        self._neighbors = defaultdict(list)
        repair = self.state['repair']
        self._counts = {tuple(k): n for k, n in repair['counts']}
        if len(repair['columns']) >= 3:
            for key in self._counts:
                for j in range(len(key)):
                    self._neighbors[j, key[:j] + key[j+1:]].append(key)
        self._density = None
        if 'engine_condition_density' in self.state or any(k.startswith('engine_gmr_') for k in self.auxiliary_arrays):
            self.load_condition_density()
        self._life_neighbors=None
        if 'engine_neighbors' in self.state or any(k.startswith('engine_neighbor_') for k in self.auxiliary_arrays):
            self.load_life_neighbors()
        self.parent_support = None
        if 'engine_parent_support' in self.state or any(k.startswith('engine_parent_') for k in self.auxiliary_arrays):
            self.load_parent_support()
        self.validate()

    def load_parent_support(self):
        if __package__:
            from .engine_parent_support import EngineParentSupport
        else:
            from engine_parent_support import EngineParentSupport
        self.parent_support = EngineParentSupport(self.base, self.auxiliary_arrays, self.state)

    def load_life_neighbors(self):
        if self._density is None:
            raise ValueError('寿命近邻需要完整的发动机条件密度父版')
        if __package__:
            from .engine_neighbors import LifeNeighbors
        else:
            from engine_neighbors import LifeNeighbors
        self._life_neighbors=LifeNeighbors(self.state.get('engine_neighbors'),self.auxiliary_arrays,self.state['condition'])

    def load_condition_density(self):
        if self.meta.get('profile')!='C' or not self.state.get('residual'):
            raise ValueError('发动机条件密度需要完整的发动机残差父版')
        if __package__:
            from .engine_condition_density import ConditionDensity
        else:
            from engine_condition_density import ConditionDensity
        self._density = ConditionDensity(self.state.get('engine_condition_density'),self.auxiliary_arrays,
                                         len(self.meta['paths']),self.state['condition'])

    @property
    def meta(self): return self.base.meta

    @property
    def params(self): return self.base.params

    @property
    def first(self): return self.base.first

    @property
    def second(self): return self.base.second

    @property
    def mean(self): return self.base.mean

    @property
    def scale(self): return self.base.scale

    def validate(self):
        self.base.validate()
        width = len(self.meta['paths'])
        if 'engine_life_limit' in self.state:
            limit = self.state['engine_life_limit']
            if (self.meta.get('profile') != 'C' or not self.state.get('residual')
                    or type(limit) not in (float,int) or not np.isfinite(limit) or limit <= 0):
                raise ValueError('发动机寿命上限需要有效的循环数与完整残差父版')
        if self.state['repair']['width'] != width:
            raise ValueError('输入修复维度与父模型不同')
        for name, values in self.auxiliary_arrays.items():
            if name != 'complete_state' and (values.dtype.kind not in 'fi' or not np.isfinite(values).all()):
                raise ValueError('完整模型存在无效数值状态：' + name)
        if self.state.get('residual'):
            n = width * 2 + int(self.state['residual']['include_parent'])
            if self.auxiliary_arrays['residual_mean'].shape != (n,) or np.any(self.auxiliary_arrays['residual_std'] <= 0):
                raise ValueError('残差网络输入刻度无效')

    def native_features(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.meta['paths']) or not np.isfinite(x).all():
            raise ValueError('完整模型输入宽度或数值无效')
        if getattr(self,'_density',None) is not None:
            x = self._density.map(x)
        return self._native_features_mapped(x)

    def _native_features_mapped(self, values):
        x = values.copy()
        columns = self.state['repair']['columns']
        for row in x:
            key = tuple(row[columns])
            if key in self._counts:
                continue
            matches = [(columns[j], neighbor[j], self._counts[neighbor])
                       for j in range(len(columns))
                       for neighbor in self._neighbors.get((j, key[:j] + key[j+1:]), ())]
            if len(matches) == 1 and matches[0][2] >= 8:
                column, value, _ = matches[0]
                row[column] = value
        condition = self.state['condition']
        indices = condition['indices']
        signals = [i for i in range(x.shape[1]) if i not in indices]
        statistics = {tuple(row['key']): row for row in condition['groups']}
        groups = defaultdict(list)
        for i, row in enumerate(x):
            key = tuple(round(float(row[j]), d) for j, d in zip(indices, condition['decimals']))
            groups[key].append(i)
        for key, rows in groups.items():
            stats = statistics.get(key, condition['global'])
            selected = np.ix_(rows, signals)
            x[selected] = (x[selected] - np.asarray(stats['mean'])) / np.asarray(stats['scale'])
        return (x - self.auxiliary_arrays['center_mean']) / self.auxiliary_arrays['center_scale']

    def normalized(self, values):
        native = self.native_features(values)
        # 保留旧程序的还原/归一化顺序，使迁移不改变其浮点运算语义。
        return self.base.normalized(native * self.scale + self.mean)

    def forward(self, values):
        return self.base.forward(values)

    def continuation_output(self, raw, logits):
        """返回完整输出的标准化值和对父网络logit的导数，冻结头仍参与反向传播。"""
        if not self.state.get('residual'):
            return logits, np.ones_like(logits)
        original_raw = raw
        native_value = logits[:, 0] * self.meta['target_scale'] + self.meta['target_mean']
        parent = np.maximum(0., native_value)
        derivative = (native_value > 0).astype(np.float64)
        if self.state.get('residual'):
            state = self.state['residual']
            arrays = self.auxiliary_arrays
            raw = np.asarray(raw,dtype=np.float64)
            if raw.shape!=(len(logits),len(self.meta['paths'])) or not np.isfinite(raw).all():
                raise ValueError('残差模型需要与父预测逐行对应的有限输入')
            if self._density is not None:
                raw = self._density.map(raw)
            raw_features = np.column_stack((raw, self._native_features_mapped(raw))).astype(np.float32)
            parent = parent.astype(np.float32)
            features = np.column_stack((raw_features, parent)) if state['include_parent'] else raw_features
            features = (features - arrays['residual_mean']) / arrays['residual_std']
            pre0 = features @ arrays['residual_0_weight'].T + arrays['residual_0_bias']
            pre2 = np.maximum(0., pre0) @ arrays['residual_2_weight'].T + arrays['residual_2_bias']
            residual = (np.maximum(0., pre2) @ arrays['residual_4_weight'].T + arrays['residual_4_bias'])[:, 0]
            if state['include_parent']:
                grad2 = (pre2 > 0) * arrays['residual_4_weight'][0]
                grad0 = (grad2 @ arrays['residual_2_weight']) * (pre0 > 0)
                slope = (grad0 @ arrays['residual_0_weight'][:, -1]) / arrays['residual_std'][-1]
                derivative *= 1 + slope * ((residual > state['lo']) & (residual < state['hi']))
            parent = parent + np.clip(residual, state['lo'], state['hi'])
            derivative *= parent > 0
        value = np.maximum(0., parent)
        if self._density is not None:
            value,derivative = self._density.mix(raw,value,derivative)
        if getattr(self,'_life_neighbors',None) is not None:
            value,derivative = self._life_neighbors.mix(raw,value,derivative)
        limit = self.state.get('engine_life_limit')
        if limit is not None:
            derivative = derivative * (value < limit)
            value = np.minimum(value, limit)
        # 源训练支持范围之外仍按该工况自己的后备父版和校准交付。
        if self.parent_support is not None:
            value,derivative = self.parent_support.mix(original_raw,value,derivative)
        scaled = (value.astype(np.float64) - self.meta['target_mean']) / self.meta['target_scale']
        return scaled[:, None], derivative[:, None]

    def predict(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        _, logits = self.forward(self.normalized(raw))
        scaled, _ = self.continuation_output(raw, logits)
        return np.maximum(0., scaled[:, 0] * self.meta['target_scale'] + self.meta['target_mean'])

    def save(self, path):
        self.validate()
        arrays = {**self.params, **{'adam_first_' + k: v for k, v in self.first.items()},
                  **{'adam_second_' + k: v for k, v in self.second.items()}, 'mean': self.mean, 'scale': self.scale,
                  'metadata': np.array(json.dumps(self.meta, sort_keys=True, separators=(',', ':'), allow_nan=False)),
                  **self.auxiliary_arrays}
        with Path(path).open('xb') as stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())


class NativeBearingGridModel(CompleteModel):
    """原生轴承模型继续保有全部权重与 Adam，仅在已学习的采集口径校准输出。"""
    def __init__(self, base, auxiliary_arrays):
        self.base = base
        self.auxiliary_arrays = dict(auxiliary_arrays)
        self.state = json.loads(str(auxiliary_arrays['complete_state']))
        self.validate()
        self.load_grid_readout(self.state['bearing_grid_readout'])

    def validate(self):
        self.base.validate()
        if (self.meta.get('profile') != 'B' or self.meta['task'] != 'regression'
                or self.state.get('format') != FORMAT or self.state.get('kind') != 'bearing-native-grid-parent'
                or set(self.auxiliary_arrays) != {'complete_state'}):
            raise ValueError('原生轴承校准与父模型状态不相容')

    @classmethod
    def from_base(cls, base, spec):
        state = {'format': FORMAT, 'kind': 'bearing-native-grid-parent', 'bearing_grid_readout': spec}
        return cls(base, {'complete_state': np.asarray(json.dumps(state, sort_keys=True, separators=(',', ':'),
                                                                  allow_nan=False))})

    def load_grid_readout(self, spec):
        if __package__:
            from .bearing_grid_readout import GridReadout
        else:
            from bearing_grid_readout import GridReadout
        self._grid_readout = GridReadout(spec, self.meta['paths'])

    def normalized(self, raw):
        return self.base.normalized(raw)

    def training_before_grid(self, raw):
        return self.base.predict(raw)

    def predict(self, raw):
        return self._grid_readout.apply(raw, self.base.predict(raw))[0]

    def continuation_output(self, raw, logits):
        scale, center = self.meta['target_scale'], self.meta['target_mean']
        minutes = logits[:, 0] * scale + center
        fitted, slope = self._grid_readout.apply(raw, minutes, native_logits=True)
        return logits + ((fitted-minutes)/scale)[:, None], slope[:, None]


def migrate_engine(base, repair, condition, center, residual=None, *, provenance):
    """仅迁移本地可信旧模型的数值状态；不拟合、不创造缺失的残差Adam历史。"""
    def stats(scaler):
        return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
    state = {'format': FORMAT, 'repair': repair.state(), 'condition': {
        'indices': list(condition.condition_indices), 'decimals': list(condition.decimals),
        'global': stats(condition.global_),
        'groups': [{'key': list(key), **stats(value)} for key, value in sorted(condition.groups_.items())]},
        'residual': None, 'provenance': deepcopy(provenance)}
    arrays = {'center_mean': center.mean_.copy(), 'center_scale': center.scale_.copy()}
    if residual is not None:
        expected = [f'input_{i}' for i in range(24)] + [f'native_{i}' for i in range(24)]
        if list(residual.raw_columns_) != expected or residual.parent_column_ != 'parent_value':
            raise ValueError('旧残差模型的特征语义不匹配')
        state['residual'] = {'include_parent': bool(residual.include_parent), 'lo': float(residual.lo_),
                             'hi': float(residual.hi_), 'training': 'frozen_legacy_no_optimizer_saved'}
        arrays.update(residual_mean=residual.mean_.copy(), residual_std=residual.std_.copy())
        for name, tensor in residual.net_.state_dict().items():
            arrays['residual_' + name.replace('.', '_')] = tensor.detach().cpu().numpy().copy()
    arrays['complete_state'] = np.array(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return CompleteModel(base, arrays)


class GasVoteModel(CompleteModel):
    """源训练的五路分类器加原生父网络；不携带目标批次回标或现场拟合统计。"""
    def __init__(self, base, auxiliary_arrays):
        self.base = base
        self.auxiliary_arrays = {k: v.copy() for k, v in auxiliary_arrays.items()}
        self.state = json.loads(str(self.auxiliary_arrays['complete_state']))
        self.validate()

    def validate(self):
        self.base.validate()
        if (self.meta['task'] != 'classification' or len(self.meta['paths']) != 128
                or self.state.get('format') != FORMAT or self.state.get('kind') != 'gas-source-vote'):
            raise ValueError('气体完整父版的结构不匹配')
        policy = self.state.get('policy', 'majority')
        if policy not in ('parent', 'majority'):
            raise ValueError('未知气体读出策略')
        if policy == 'majority' and not {'gas_log_scale', 'gas_svm_support', 'gas_type2_mean',
                                        'gas_type4_mean', 'gas_relative_mean'} <= self.auxiliary_arrays.keys():
            raise ValueError('多数投票缺少源训练分量')
        for k, v in self.auxiliary_arrays.items():
            if k != 'complete_state' and (v.dtype.kind not in 'fi' or not np.isfinite(v).all()):
                raise ValueError('源分类器数值状态无效')

    def normalized(self, values): return self.base.normalized(values)

    def continuation_output(self, raw, logits):
        # 硬投票不提供可微梯度。更新真实父分类器，完整投票结果在探索/确认阶段评价。
        return logits, np.ones_like(logits)

    def source_votes(self, x):
        if __package__:
            from .gas_views import represent
        else:
            from gas_views import represent
        arrays = self.auxiliary_arrays
        shape = represent(x, 'shape', None, None)
        log = (np.sign(x) * np.log1p(np.abs(x) / arrays['gas_log_scale'])).reshape(-1, 16, 8)
        views = {'type2': log[:, (2, 3, 10, 11), :].reshape(len(x), 32),
                 'type4': log[:, (6, 7, 14, 15), :].reshape(len(x), 32), 'relative': shape}
        votes = []
        for name, values in views.items():
            prefix = 'gas_' + name + '_'
            z = (values - arrays[prefix + 'mean']) / arrays[prefix + 'scale']
            scores = z @ arrays[prefix + 'coef'] + arrays[prefix + 'bias']
            votes.append(arrays[prefix + 'classes'][scores.argmax(axis=1)].astype(int))
        support = arrays['gas_svm_support']
        distance = np.maximum(0., (shape**2).sum(1)[:, None] + (support**2).sum(1) - 2 * shape @ support.T)
        kernel = np.exp(-self.state['svm_gamma'] * distance)
        counts = arrays['gas_svm_counts'].astype(int)
        starts = np.r_[0, np.cumsum(counts)]
        scores = np.zeros((len(x), len(counts)), dtype=int)
        index = 0
        for i in range(len(counts)):
            for j in range(i+1, len(counts)):
                a, b = slice(starts[i], starts[i+1]), slice(starts[j], starts[j+1])
                margin = (kernel[:, a] @ arrays['gas_svm_dual'][j-1, a]
                          + kernel[:, b] @ arrays['gas_svm_dual'][i, b] + arrays['gas_svm_intercept'][index])
                scores[:, i] += margin > 0
                scores[:, j] += margin <= 0
                index += 1
        votes.append(arrays['gas_svm_classes'][scores.argmax(axis=1)].astype(int))
        return votes

    def predict(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        parent_probability = self.base.predict(raw)
        if self.state.get('policy', 'majority') == 'parent':
            return parent_probability
        current = parent_probability.argmax(axis=1) + 1
        votes = np.stack([current, *self.source_votes(raw)])
        count = np.stack([(votes == c).sum(axis=0) for c in range(1, 7)], axis=1)
        chosen = np.where(count[np.arange(len(raw)), current-1] == count.max(1), current-1, count.argmax(1))
        # 未回标旧观察者的平滑计分为票数乘log(2)，平手按原生父版类别优先。
        probability = np.exp((count - count.max(1, keepdims=True)) * np.log(2.))
        for i, c in enumerate(chosen):
            if probability[i].argmax() != c:
                probability[i, c] = np.nextafter(probability[i].max(), np.inf)
        probability /= probability.sum(1, keepdims=True)
        for i, c in enumerate(chosen):
            if probability[i].argmax() != c:
                probability[i, c] = np.nextafter(probability[i].max(), np.inf)
        return probability


def migrate_gas(base, bank, *, provenance):
    """封存已有源训练组件；目标批次反馈计数不能进入比赛源训练版。"""
    state = {'format': FORMAT, 'kind': 'gas-source-vote', 'svm_gamma': float(bank.svm._gamma),
             'training_objective': 'native_classifier_surrogate', 'provenance': deepcopy(provenance)}
    arrays = {'gas_log_scale': bank.scale.copy(), 'gas_svm_support': bank.svm.support_vectors_.copy(),
              'gas_svm_dual': bank.svm._dual_coef_.copy(), 'gas_svm_intercept': bank.svm._intercept_.copy(),
              'gas_svm_counts': bank.svm.n_support_.copy(), 'gas_svm_classes': bank.svm.classes_.copy()}
    for name, fitted in bank.models.items():
        for field, value in zip(('mean', 'scale', 'coef', 'bias', 'classes'), fitted):
            arrays['gas_' + name.replace('-', '') + '_' + field] = value.copy()
    arrays['complete_state'] = np.array(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return GasVoteModel(base, arrays)
