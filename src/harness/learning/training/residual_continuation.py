"""恢复完整回归父版的残差 AdamW；不拟合输入尺度、不替换父网络。

调用方须在训练源码的审计钩子启用前导入本模块并预热 PyTorch 优化器。
本模块只使用传入的合法训练数组；总轮数预算由 ContinuationSession 负责。
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np
import torch


_PARAMETERS = ('0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')
_BATCH_SIZE = 1024
_ANCHORS_PER_GROUP = 128


def _digest(value):
    return sha256(np.asarray(value).tobytes()).hexdigest()


def _array_key(name):
    return 'residual_' + name.replace('.', '_')


def _optimizer_key(name, field):
    return 'residual_optimizer_' + name.replace('.', '_') + '_' + field


def _optimizer_summary(arrays):
    return {name: {
        'step': int(float(arrays[_optimizer_key(name, 'step')])),
        'exp_avg_sha256': _digest(arrays[_optimizer_key(name, 'exp_avg')]),
        'exp_avg_sq_sha256': _digest(arrays[_optimizer_key(name, 'exp_avg_sq')]),
    } for name in _PARAMETERS}


def _frozen_identity(model, mutable):
    values = {'parent_' + name: value for name, value in model.params.items()}
    values.update({'parent_adam_first_' + name: value for name, value in model.first.items()})
    values.update({'parent_adam_second_' + name: value for name, value in model.second.items()})
    values.update(parent_mean=model.mean, parent_scale=model.scale)
    values.update({name: value for name, value in model.auxiliary_arrays.items()
                   if name not in mutable})
    identities = {name: _digest(value) for name, value in values.items()}
    identities['parent_metadata'] = sha256(json.dumps(
        model.meta, sort_keys=True, allow_nan=False).encode()).hexdigest()
    state = deepcopy(model.state)
    # 只允许修改残差续训记录；工况、修复规则与原发布来源均保持不变。
    state.pop('residual')
    identities['input_state'] = sha256(json.dumps(
        state, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return identities


def continue_residual(model, x, y, groups, *, epochs=1, learning_rate=None,
                      trainable='readout', loss='huber', beta=3.):
    """续训已有残差头并返回权重、优化器、随机状态及冻结块的审计摘要。

    readout 只更新末层，all 更新已有全部残差参数。两者均恢复所有 AdamW
    矩和逐参数步数，冻结参数的优化器状态也保持原样。旧版缺少优化器时拒绝。
    huber 沿用原 SmoothL1；scaled_squared 为 ((残差预测-(y-父预测))/
    原 target_scale)^2。两项损失在部署截断之前计算，沿用原残差训练语义。
    成功后才写回模型；任何训练或验证异常均保留传入的模型与全局 Torch RNG。
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError('残差续训轮数须为正整数，总预算由宿主另外核对')
    if trainable not in ('readout', 'all') or loss not in ('huber', 'scaled_squared'):
        raise ValueError('残差续训参数块或损失类型无效')
    if isinstance(beta, bool) or not math.isfinite(float(beta)) or float(beta) <= 0:
        raise ValueError('Huber beta须为有限正数')
    if not hasattr(model, 'auxiliary_arrays') or model.meta['task'] != 'regression':
        raise ValueError('残差续训需要已有完整回归父版')
    state = model.state.get('residual')
    if not state or state.get('training') != 'source_trained_optimizer_saved' or state.get('optimizer') != 'AdamW':
        raise ValueError('父残差头没有可恢复的真实AdamW状态，不能重初始化冒充续训')
    arrays = model.auxiliary_arrays
    required = {'residual_mean', 'residual_std', 'residual_random_state'}
    required.update(_array_key(name) for name in _PARAMETERS)
    required.update(_optimizer_key(name, key) for name in _PARAMETERS
                    for key in ('exp_avg', 'exp_avg_sq', 'step'))
    if not required <= arrays.keys():
        raise ValueError('父残差头缺少权重、优化器矩、步数或随机状态')
    rate = state.get('learning_rate') if learning_rate is None else learning_rate
    if isinstance(rate, bool) or not math.isfinite(float(rate)) or not 0 < float(rate) <= 1:
        raise ValueError('残差学习率必须在(0,1]内')
    decay = state.get('weight_decay')
    if isinstance(decay, bool) or not math.isfinite(float(decay)) or float(decay) < 0:
        raise ValueError('父残差AdamW权重衰减无效')
    scale = float(model.meta['target_scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('父目标尺度无效')
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y), np.asarray(groups)
    if (x.ndim != 2 or not len(x) or y.shape != (len(x),) or groups.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('残差续训需要对应的有限训练特征、目标和设备身份')
    keep = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        keep.extend(rows[np.linspace(0, len(rows)-1, min(_ANCHORS_PER_GROUP, len(rows))).astype(int)])
    keep = np.asarray(keep, dtype=int)
    if not len(keep):
        raise ValueError('残差续训未取得合法设备锚点')
    native = model.native_features(x)
    parent = model.base.predict(native * model.scale + model.mean).astype(np.float32)
    raw = np.column_stack((x, native)).astype(np.float32)
    features = np.column_stack((raw, parent)) if state['include_parent'] else raw
    mean, std = arrays['residual_mean'], arrays['residual_std']
    if (mean.shape != (features.shape[1],) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('父残差输入尺度无效')
    xx = torch.tensor((features[keep] - mean) / std, dtype=torch.float32)
    yy = torch.tensor((np.asarray(y, dtype=np.float32)-parent)[keep, None], dtype=torch.float32)
    if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
        raise ValueError('残差训练数组超出有限单精度范围')
    selected = _PARAMETERS if trainable == 'all' else ('4.weight', '4.bias')
    mutable = {'complete_state', 'residual_random_state'}
    mutable.update(_array_key(name) for name in selected)
    mutable.update(_optimizer_key(name, key) for name in selected
                   for key in ('exp_avg', 'exp_avg_sq', 'step'))
    frozen_before = _frozen_identity(model, mutable)
    initial_weights = {name: _digest(arrays[_array_key(name)]) for name in _PARAMETERS}
    initial_optimizer = _optimizer_summary(arrays)
    initial_rng = _digest(arrays['residual_random_state'])
    random_state = arrays['residual_random_state']
    if (random_state.ndim != 1 or random_state.dtype.kind not in 'iu'
            or np.any(random_state < 0) or np.any(random_state > 255)):
        raise ValueError('父残差Torch随机状态无效')

    def losses(prediction, target):
        if loss == 'huber':
            return torch.nn.functional.smooth_l1_loss(prediction, target, beta=float(beta), reduction='none')
        return ((prediction-target)/scale)**2

    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        # 构造容器时产生的随机初值立即被完整覆盖，随后恢复已发布的 RNG。
        n0, n1 = arrays['residual_0_weight'].shape[0], arrays['residual_2_weight'].shape[0]
        net = torch.nn.Sequential(torch.nn.Linear(features.shape[1], n0), torch.nn.ReLU(),
                                  torch.nn.Linear(n0, n1), torch.nn.ReLU(), torch.nn.Linear(n1, 1))
        tensors, saved_optimizer = {}, {}
        for index, (name, parameter) in enumerate(net.named_parameters()):
            value = arrays[_array_key(name)]
            if value.shape != tuple(parameter.shape) or value.dtype != np.dtype('float32') or not np.isfinite(value).all():
                raise ValueError('父残差权重形状或数值无效：' + name)
            tensors[name] = torch.from_numpy(value.copy())
            optimizer_state = {}
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                value = arrays[_optimizer_key(name, key)]
                expected_shape = () if key == 'step' else tuple(parameter.shape)
                if value.shape != expected_shape or value.dtype != np.dtype('float32') or not np.isfinite(value).all():
                    raise ValueError('父残差优化器状态形状或数值无效：' + name + '/' + key)
                if key == 'step' and (float(value) < 1 or float(value) != int(float(value))):
                    raise ValueError('父残差优化器没有真实正整数更新步数')
                if key == 'exp_avg_sq' and np.any(value < 0):
                    raise ValueError('父残差AdamW二阶矩不能为负')
                optimizer_state[key] = torch.from_numpy(value.copy())
            saved_optimizer[index] = optimizer_state
            parameter.requires_grad_(name in selected)
        net.load_state_dict(tensors, strict=True)
        # 原构建器使用这组AdamW默认值；不能用新优化器的零矩覆盖已发布的矩。
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(rate), weight_decay=float(decay))
        groups_state = optimizer.state_dict()['param_groups']
        optimizer.load_state_dict({'state': saved_optimizer, 'param_groups': groups_state})
        torch.set_rng_state(torch.from_numpy(random_state.astype(np.uint8)))

        def evaluate_loss():
            with torch.no_grad():
                return sum(float(losses(net(xb), yb).sum())
                           for xb, yb in zip(xx.split(_BATCH_SIZE), yy.split(_BATCH_SIZE), strict=True)) / len(xx)

        initial_loss = evaluate_loss()
        history = []
        for epoch in range(epochs):
            loss_sum, steps = 0., 0
            for indices in torch.randperm(len(xx)).split(_BATCH_SIZE):
                batch_loss = losses(net(xx[indices]), yy[indices])
                value = batch_loss.mean()
                if not torch.isfinite(value):
                    raise ValueError('残差续训损失不是有限值')
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise ValueError('残差续训梯度不是有限值')
                optimizer.step()
                loss_sum += float(batch_loss.detach().sum())
                steps += 1
            history.append({'epoch': state['epochs']+epoch+1, 'steps': steps,
                            'loss': loss_sum / len(xx)})
        final_loss = evaluate_loss()
        if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
            raise ValueError('残差续训完整训练损失不是有限值')
        updated = {name: value.copy() for name, value in arrays.items()}
        for name, parameter in net.named_parameters():
            updated[_array_key(name)] = parameter.detach().numpy().copy()
            for key, value in optimizer.state[parameter].items():
                updated[_optimizer_key(name, key)] = value.detach().numpy().copy()
        updated['residual_random_state'] = torch.get_rng_state().numpy().astype(np.int64)

    next_state = deepcopy(model.state)
    next_state['residual'].update(epochs=state['epochs']+epochs, learning_rate=float(rate),
        continuation_policy='inherited_adamw_residual_' + trainable, last_loss=loss,
        last_beta=float(beta), last_target_scale=scale)
    updated['complete_state'] = np.array(json.dumps(next_state, sort_keys=True, separators=(',', ':'), allow_nan=False))
    old_state = model.state
    model.auxiliary_arrays, model.state = updated, next_state
    try:
        model.validate()
        frozen_after = _frozen_identity(model, mutable)
        if frozen_after != frozen_before:
            raise ValueError('残差续训改变了冻结父网络、输入处理或未选参数块')
    except BaseException:
        model.auxiliary_arrays, model.state = arrays, old_state
        raise
    return {'component': 'residual', 'epochs': epochs, 'trainable': trainable,
        'loss': loss, 'loss_definition': 'smooth_l1(residual,y-parent,beta)' if loss == 'huber'
            else 'mean(((residual-(y-parent))/parent_target_scale)^2)',
        'beta': float(beta), 'target_scale': scale, 'learning_rate': float(rate), 'weight_decay': float(decay),
        'anchors_per_group': _ANCHORS_PER_GROUP, 'anchors': len(keep),
        'anchor_indices_sha256': _digest(keep), 'batch_size': _BATCH_SIZE,
        'steps': sum(row['steps'] for row in history), 'history': history,
        'initial_loss': initial_loss, 'final_loss': final_loss,
        'initial_weights': initial_weights,
        'final_weights': {name: _digest(updated[_array_key(name)]) for name in _PARAMETERS},
        'initial_optimizer': initial_optimizer, 'final_optimizer': _optimizer_summary(updated),
        'initial_rng_sha256': initial_rng, 'final_rng_sha256': _digest(updated['residual_random_state']),
        'frozen_before': frozen_before, 'frozen_after': frozen_after,
        'optimizer_restored': True, 'global_rng_restored': True, 'seconds': time.monotonic()-started}
