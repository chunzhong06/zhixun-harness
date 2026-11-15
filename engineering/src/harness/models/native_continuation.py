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
    Adam更新。首版沿用官方网络结构，使产物可由原预测器独立载入。
    """

    def __init__(self, model, x, y, *, max_epochs=30, groups=None):
        self._model = model
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y)
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
        self._expected_auxiliary = {k: v for k, v in self._initial_tensors.items()
                                    if k in getattr(model, 'auxiliary_arrays', {})}

    def build_regression_components(self, condition_indices, decimals, *, epochs=60, head_epochs=12):
        """源训练工况处理、真实父网络续训和残差训练；只允许从普通父版构建一次。"""
        if self._history or self._migration or hasattr(self._model, 'auxiliary_arrays'):
            raise ValueError('完整组件构建只用于尚未训练的普通父版')
        if self._model.meta['task'] != 'regression' or type(epochs) is not int or type(head_epochs) is not int:
            raise ValueError('完整组件需要回归父版与整数训练轮数')
        if not 1 <= epochs <= self._max_epochs or not 1 <= head_epochs <= 12:
            raise ValueError('完整组件训练轮数超出本次预算')
        from harness.models.complete_training import attach_inputs, attach_residual
        old = self._model.predict(self._x[:128])
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
        final = tensor_identity(self._model)
        self._expected_auxiliary = {k: final[k] for k in self._model.auxiliary_arrays}
        return deepcopy(self._component_history[-1])

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
        return self._model.predict(self._x).copy()

    def expand_hidden(self, hidden):
        """在首个梯度前增加隐藏单元；保留原权重和Adam块，新输出连接为零。"""
        model = self._model
        old_hidden = model.meta['hidden']
        if self._history or self._migration is not None:
            raise ValueError('一次程序只能在首个梯度前扩展网络一次')
        if type(hidden) is not int or not old_hidden < hidden <= 128:
            raise ValueError('扩展宽度须大于父网络且不超过官方格式上限128')
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
    if any(isinstance(n, ast.Attribute) and n.attr == 'build_regression_components' for n in ast.walk(tree)):
        from harness.models.complete_training import prepare_runtime
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
