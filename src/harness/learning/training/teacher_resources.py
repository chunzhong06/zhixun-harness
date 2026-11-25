"""比赛入口装配已核验共享教师；只接受当前合法训练资料中的源上下文。"""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

from harness.paths import PACKAGE_ROOT, workspace_root


def checkpoint_kind(parent):
    with np.load(parent, allow_pickle=False) as arrays:
        return json.loads(str(arrays['complete_state'])).get('kind') if 'complete_state' in arrays.files else None


def default_program_seconds(parent, resource=None):
    """程序预算包含保存后重载和完整预测；TabPFN沿用既有接入验证的30分钟预算。"""
    kind = checkpoint_kind(parent)
    tabpfn = kind in ('tabpfn-classification-parent', 'tabpfn-regression-parent')
    if resource is not None:
        with np.load(resource['checkpoint'], allow_pickle=False) as arrays:
            tabpfn = tabpfn or bool({'tabpfn_classifier_state', 'tabpfn_regressor_state'} & set(arrays.files))
    # 完整Linux验证确认旧15分钟预算会在预测期间截断有效的TabPFN候选。
    # SDK调用方仍须按宿主给出的本轮时限缩减，此值不是赛方资源上限。
    if tabpfn:
        return 1800
    return 900 if resource or kind in ('tabicl-parent-combination', 'tabicl-classification-parent') else 90


def registered_tabular_resource(profile, parent, train, *, manifest=None):
    """优先装配合法的既有上下文；气体静态量化权重也可用当轮行重新拟合。"""
    kind = checkpoint_kind(parent)
    if profile not in ('A','B') or (profile == 'B' and kind is not None) or (profile == 'A' and kind not in (None,'gas-source-vote')):
        return None
    manifest = Path(manifest or os.environ.get('HARNESS_TABULAR_RESOURCES',
                    PACKAGE_ROOT/'resources/tabular-resources.json'))
    record = json.loads(manifest.read_bytes())
    if record.get('format') != 'harness.tabular-resources.v1':
        raise ValueError('未知共享教师清单格式')
    selected = record['tasks'].get(profile)
    if selected is None:
        return None
    path = Path(selected['checkpoint'])
    if not path.is_absolute():
        path = workspace_root()/path
    path = path.resolve()
    if sha256(path.read_bytes()).hexdigest() != selected['sha256']:
        raise ValueError('默认共享教师资源身份改变')
    from harness.adapters.platforms.baselines.core_model import Model
    from harness.adapters.platforms.baselines.data import read_dataset
    from harness.learning.inference.complete_checkpoint import load_checkpoint
    model = load_checkpoint(parent, Model)
    if profile == 'A':
        from harness.learning.inference.tabular_classifier import classification_base
        try:
            classification_base(model)
        except ValueError:
            # 默认装配只迁移已核验的父外壳；带其他有效分量的版本仍正常续训。
            return None
    data = read_dataset(train, model.meta['task'], model.meta['classes'], model.meta['paths'])
    with np.load(path, allow_pickle=False) as arrays:
        prefix = 'classifier' if profile == 'A' else 'readout'
        has_rows = prefix+'_source_x' in arrays.files and prefix+'_source_y' in arrays.files
        if has_rows:
            source = {x.tobytes()+np.float64(y).tobytes() for x, y in zip(data['x'], data['y'], strict=True)}
            if all(x.astype(np.float64).tobytes()+np.float64(y).tobytes() in source
                   for x, y in zip(arrays[prefix+'_source_x'], arrays[prefix+'_source_y'], strict=True)):
                return {'checkpoint':str(path), 'sha256':selected['sha256']}
        if profile == 'B' and not has_rows and 'tabpfn_regressor_state' in arrays.files:
            spec = json.loads(str(arrays['tabpfn_regressor_state']))
            if spec.get('feature_paths') == model.meta['paths'] and len(data['ids']) >= 2:
                from harness.learning.inference import bearing_fit_state
                prepared = bearing_fit_state.manifest()
                if prepared is not None and bearing_fit_state.issued_indices(data,prepared) is None:
                    return None
                return {'checkpoint':str(path), 'sha256':selected['sha256'], 'context_train_sha256':data['sha256']}
        if profile == 'A' and 'tabpfn_classifier_state' in arrays.files:
            teacher = json.loads(str(arrays['tabpfn_classifier_state']))
            width = arrays['classifier_source_x'].shape[1] if has_rows else teacher.get('context_width')
            if (teacher.get('precision') == 'sinq_storage_float32_inference'
                    and teacher.get('classes') == model.meta['classes']
                    and data['x'].shape[1] == width
                    and np.array_equal(np.unique(data['y']), np.arange(len(model.meta['classes'])))):
                return {'checkpoint':str(path), 'sha256':selected['sha256'], 'context_train_sha256':data['sha256']}
    print('本轮合法输入未覆盖所需教师上下文或类别，本轮沿用普通父版续训。', file=sys.stderr)
    return None


def export_gas_teacher_weights(source, destination, *, estimators=None):
    """为下一次镜像构建导出静态 SINQ 教师，仅携带作者权重、许可、输入宽度和已验证容量。"""
    with np.load(source,allow_pickle=False) as registered:
        if 'tabpfn_classifier_state' not in registered:
            return False
        spec = json.loads(str(registered['tabpfn_classifier_state']))
        if spec.get('precision') != 'sinq_storage_float32_inference':
            return False
        if 'classifier_source_x' in registered:
            rows,width = registered['classifier_source_x'].shape
        else:
            rows,width = spec['context_capacity'],spec['context_width']
        fields = ('format','backend','author_library','original_weight_sha256','asset_sha256','precision',
                  'teacher_parameters','n_estimators','recommended_parent_weight','label_encoding','classes')
        asset_spec = {key:spec[key] for key in fields}
        if estimators is not None:
            if type(estimators) is not int or estimators not in (4, 8):
                raise ValueError('当前双上下文气体组合支持四个或八个估计器')
            asset_spec['n_estimators'] = estimators
        asset_spec.update(context_capacity=rows,context_width=width)
        with destination.open('xb') as stream:
            np.savez(stream,teacher_asset=registered['teacher_asset'],teacher_license=registered['teacher_license'],
                     tabpfn_classifier_state=np.asarray(json.dumps(asset_spec,sort_keys=True)))
    return True


def export_bearing_teacher_weights(source, destination):
    """归档只带教师参数和上下文容量，训练 ID、标签及特征不随镜像分发。"""
    with np.load(source, allow_pickle=False) as data:
        spec = json.loads(str(data['tabpfn_regressor_state']))
        if spec['format'] != 'harness.tabpfn-regressor-resource.v1':
            raise ValueError('未知回归教师格式')
        for key in ('source_ids', 'source_train_sha256', 'source_x_sha256', 'source_y_sha256'):
            del spec[key]
        spec['context_capacity'] = len(data['readout_source_y'])
        with destination.open('xb') as out:
            np.savez(out, teacher_asset=data['teacher_asset'], teacher_license=data['teacher_license'],
                     tabpfn_regressor_state=np.asarray(json.dumps(spec, sort_keys=True)))


def _bearing_reference_rows(train, segments):
    """在连续寿命段内兼顾时间覆盖和振动变化，每段仍至多取 128 条。"""
    fields = ('rms', 'kurtosis', 'spectral_centroid', 'band1_energy_0_2k',
              'band2_energy_2k_6k', 'band3_energy_6k_12k')
    paths = [json.dumps([['key', 'features'], ['key', axis], ['key', field]], separators=(',', ':'))
             for axis in ('horizontal', 'vertical') for field in fields]
    columns = {path: index for index, path in enumerate(train['paths'])}
    if not all(path in columns for path in paths):
        return np.concatenate([s[np.linspace(0, len(s)-1, min(len(s), 128), dtype=int)] for s in segments])
    raw = train['x'][:, [columns[path] for path in paths]]
    center = np.median(raw, axis=0)
    width = np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0)
    scaled = np.arcsinh((raw-center)/np.where(width > 0, width, 1.))
    selected = []
    for segment in segments:
        if len(segment) <= 128:
            selected.extend(segment)
            continue
        padded = np.pad(scaled[segment], ((4, 4), (0, 0)), mode='edge')
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 9, axis=0), axis=-1)
        changes = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
        distance = np.r_[0., np.cumsum(np.minimum(changes, np.quantile(changes, .95)))]
        timeline = np.linspace(0., 1., len(segment))
        if distance[-1] == 0:
            selected.extend(segment[np.linspace(0, len(segment)-1, 128, dtype=int)])
            continue
        progress = (timeline + distance/distance[-1])/2
        previous = -1
        for j, target in enumerate(np.linspace(0., 1., 128)):
            index = int(np.searchsorted(progress, target))
            # 为其余配额留足不同的源行，首末观测也各自保留。
            choices = np.clip([index-1, index], previous+1, len(segment)-128+j)
            previous = int(choices[np.argmin(abs(progress[choices]-target))])
            selected.append(segment[previous])
    return np.asarray(selected, dtype=int)


def rebuild_bearing_resource(resource, train, destination):
    """从当轮合法训练数据重建固定容量的轴承教师参照。"""
    if resource['context_train_sha256'] != train['sha256']:
        raise ValueError('回归教师的训练输入身份改变')
    with np.load(resource['checkpoint'], allow_pickle=False) as data:
        spec = json.loads(str(data['tabpfn_regressor_state']))
        capacity = spec.pop('context_capacity')
        labels = train['y']
        if type(capacity) is not int or capacity < 2 or len(labels) < 2 or spec['feature_paths'] != train['paths']:
            raise ValueError('回归上下文容量或输入列不符')
        from harness.learning.inference import bearing_fit_state
        prepared = bearing_fit_state.manifest()
        if prepared is None:
            rows = np.arange(len(labels))
        else:
            rows = bearing_fit_state.issued_indices(train,prepared)
            if rows is None:
                raise ValueError('现场训练数据未覆盖已登记的轴承源上下文')
            rows = np.asarray(rows,dtype=int)
        if len(rows) > capacity:
            cuts = np.flatnonzero(np.diff(labels) != -1)+1
            segments = np.split(rows, cuts)
            if all(len(s) > 1 for s in segments):
                rows = _bearing_reference_rows(train, segments)
            if len(rows) > capacity:
                rows = rows[np.linspace(0,len(rows)-1,capacity,dtype=int)]
        x, y = train['x'][rows], labels[rows]
        spec.update(source_ids=[train['ids'][i] for i in rows], source_train_sha256=train['sha256'],
                    source_x_sha256=sha256(x.tobytes()).hexdigest(), source_y_sha256=sha256(y.tobytes()).hexdigest())
        with destination.open('xb') as out:
            np.savez(out, teacher_asset=data['teacher_asset'], teacher_license=data['teacher_license'],
                     readout_source_x=x, readout_source_y=y, tabpfn_regressor_state=np.asarray(json.dumps(spec,sort_keys=True)))
    return {'checkpoint':str(destination.resolve()), 'sha256':sha256(destination.read_bytes()).hexdigest()}


def rebuild_gas_resource(resource, train, destination, classes):
    """以平台本轮训练集替换两张上下文数组。量化权重、许可和作者拟合配置原样保留。"""
    if resource['context_train_sha256'] != train['sha256']:
        raise ValueError('重建上下文与登记的本轮训练输入不一致')
    with np.load(resource['checkpoint'], allow_pickle=False) as registered:
        spec = json.loads(str(registered['tabpfn_classifier_state']))
        labels = train['y']
        if 'classifier_source_x' in registered:
            capacity,width = registered['classifier_source_x'].shape
        else:
            capacity,width = spec.pop('context_capacity'),spec.pop('context_width')
        if (spec.get('format') != 'harness.tabpfn-classifier-resource.v1'
                or spec.get('precision') != 'sinq_storage_float32_inference' or spec.get('classes') != classes
                or type(capacity) is not int or capacity < len(classes) or width != train['x'].shape[1]
                or not np.array_equal(np.unique(labels),np.arange(len(classes)))):
            raise ValueError('当轮训练无法构建相同类别契约的 SINQ 教师')
        # 超过已验证的上下文容量时，各类保留一行，剩余名额从当轮训练确定性抽取。
        indices = np.arange(len(labels))
        if len(indices) > capacity:
            rng = np.random.default_rng(42)
            retained = np.array([rng.choice(indices[labels == c]) for c in range(len(classes))])
            pool = np.setdiff1d(indices,retained,assume_unique=True)
            indices = np.sort(np.r_[retained, rng.choice(pool,capacity-len(retained),replace=False)])
        x, y = train['x'][indices], labels[indices]
        spec.update(context_origin='current_train', source_train_sha256=train['sha256'],
                    source_ids=[train['ids'][i] for i in indices],
                    source_x_sha256=sha256(x.tobytes()).hexdigest(), source_y_sha256=sha256(y.tobytes()).hexdigest())
        # 目录由本轮排他锁创建；完成保存后才把路径交给训练及恢复入口。
        with destination.open('xb') as stream:
            np.savez(stream, teacher_asset=registered['teacher_asset'], teacher_license=registered['teacher_license'],
                     classifier_source_x=x,classifier_source_y=y,tabpfn_classifier_state=np.asarray(json.dumps(spec,sort_keys=True)))
    return {'checkpoint':str(destination.resolve()), 'sha256':sha256(destination.read_bytes()).hexdigest()}
