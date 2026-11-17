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
    Adam更新。支持显式扩展和组件迁移，独立加载器随实际产物一起交付。
    """

    def __init__(self, model, x, y, *, max_epochs=30, groups=None, tabular_resource=None):
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
        self._head_epochs_used = 0
        self._expected_auxiliary = {k: v for k, v in self._initial_tensors.items()
                                    if k in getattr(model, 'auxiliary_arrays', {})}
        self._tabular_resource = tabular_resource
        self._prepared_tabular = None
        self._prepared_tabpfn_readout = None
        self._prepared_classifier = None
        if hasattr(model, 'prepare_training'):
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
        self._component_history.append(report)
        return deepcopy(report)

    def _prepare_tabular_classifier(self,backend='tabicl'):
        """分类教师只看源上下文；文件和作者推理在数值程序审计边界外预备。"""
        if backend=='tabicl':
            from harness.learning.inference.tabular_classifier import from_classifier_asset
        elif backend=='tabpfn':
            from harness.learning.inference.tabpfn_classifier import from_classifier_asset
        else:
            raise ValueError('未知分类教师后端')
        resource=self._tabular_resource
        if resource is None or sha256(Path(resource['checkpoint']).read_bytes()).hexdigest()!=resource['sha256']:
            raise ValueError('分类教师资源缺失或身份改变')
        prepared=from_classifier_asset(self._model,resource['checkpoint'])
        source={x.tobytes()+np.float64(y).tobytes() for x,y in zip(self._x,self._y,strict=True)}
        cx,cy=prepared.auxiliary_arrays['classifier_source_x'],prepared.auxiliary_arrays['classifier_source_y']
        if any(x.astype(np.float64).tobytes()+np.float64(y).tobytes() not in source for x,y in zip(cx,cy,strict=True)):
            raise ValueError('分类教师上下文包含未授权训练行')
        prepared.prepare_training(self._x)
        self._prepared_classifier=prepared

    def build_tabicl_classifier(self,parent_weight=.25):
        """保留父网、Adam和原源修正；可微组合在离散源修正之前。"""
        return self._attach_classifier(parent_weight,'tabicl-classification-parent')

    def build_tabpfn_classifier(self,parent_weight=.25):
        """算术组合教师与真实父网；梯度经完整组合Jacobian返回父网。"""
        return self._attach_classifier(parent_weight,'tabpfn-classification-parent')

    def _attach_classifier(self,parent_weight,expected_kind):
        from harness.learning.inference.tabular_classifier import classification_base
        if self._prepared_classifier is None or self._prepared_classifier.state.get('kind')!=expected_kind:
            raise ValueError('对应分类教师须先由宿主预备')
        base=classification_base(self._model)
        before=tensor_identity(base)
        metadata_sha=sha256(json.dumps(base.meta,sort_keys=True).encode()).hexdigest()
        old_wrapper=deepcopy(getattr(self._model,'state',None))
        model=self._prepared_classifier;model.base=base
        model.state['parent_weight']=float(parent_weight)
        model.auxiliary_arrays['complete_state']=np.asarray(json.dumps(model.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        model.validate();current=tensor_identity(model)
        if {k:current[k] for k in before}!=before:
            raise ValueError('分类组件改变了父权重、Adam或尺度')
        self._model=model
        self._expected_auxiliary={k:v for k,v in current.items() if k in model.auxiliary_arrays}
        if self._first_step_tensors is None:self._training_origin_tensors=current
        report={'kind':expected_kind,'parent_weight':model.state['parent_weight'],
            'retained_parent_tensors':before,'retained_parent_metadata_sha256':metadata_sha,
            'replaced_wrapper_descriptor':old_wrapper,'teacher_training':'frozen',
            'parent_epochs_before_attachment':len(self._history),'source_context_rows':model.state['source_context_rows'],
            'training_objective':model.state['training_objective']}
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
            return self._model.training_predictions(self._x).copy()
        return self._model.predict(self._x).copy()

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
    readout_methods = numerical_methods & {'build_tabicl_readout', 'build_tabpfn_readout'}
    if len(readout_methods) > 1:
        raise ValueError('一次训练程序只接入一种回归教师')
    if 'build_tabicl_readout' in readout_methods:
        session._prepare_tabular()
    if 'build_tabpfn_readout' in readout_methods:
        session._prepare_tabpfn_readout()
    classifier_methods=numerical_methods & {'build_tabicl_classifier','build_tabpfn_classifier'}
    if len(classifier_methods)>1:
        raise ValueError('一次训练程序只接入一种分类教师')
    if classifier_methods:
        session._prepare_tabular_classifier('tabpfn' if 'build_tabpfn_classifier' in classifier_methods else 'tabicl')
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
