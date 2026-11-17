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


def prediction_sources(package, checkpoint):
    """为实际模型选择加载入口；完整模型的预测代码不依赖研究目录或旧pickle。"""
    files = {'baselines/' + name: package.read('baselines/' + name)
             for name in ('__init__.py', 'model.py', 'data.py', 'predict.py')}
    if is_complete(checkpoint):
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
            kind = json.loads(str(arrays['complete_state'])).get('kind')
        if kind in ('tabpfn-classification-parent', 'tabpfn-regression-parent'):
            from .tabpfn_classifier import prediction_dependencies
            for name in ('tabular_readout.py','tabular_classifier.py','tabpfn_classifier.py',
                         'quantized_teacher.py','probability_mixture.py','teacher_cache.py'):
                files['baselines/'+name]=Path(__file__).with_name(name).read_bytes()
            if kind == 'tabpfn-regression-parent':
                files['baselines/tabpfn_regressor.py'] = Path(__file__).with_name('tabpfn_regressor.py').read_bytes()
            files.update(prediction_dependencies())
            with np.load(checkpoint,allow_pickle=False) as arrays:
                files['TabPFN-model-LICENSE']=str(arrays['teacher_license']).encode('utf-8')
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
        return base_class.load(path)
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
    else:
        model_class = GasVoteModel if state.get('kind') == 'gas-source-vote' else CompleteModel
    return model_class(base, extra)


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
        self.validate()

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
        x = np.asarray(values, dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != len(self.meta['paths']) or not np.isfinite(x).all():
            raise ValueError('完整模型输入宽度或数值无效')
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
        native_value = logits[:, 0] * self.meta['target_scale'] + self.meta['target_mean']
        parent = np.maximum(0., native_value)
        derivative = (native_value > 0).astype(np.float64)
        if self.state.get('residual'):
            state = self.state['residual']
            arrays = self.auxiliary_arrays
            raw_features = np.column_stack((raw, self.native_features(raw))).astype(np.float32)
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
