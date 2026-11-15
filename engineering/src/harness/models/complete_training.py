"""从实际父网络和合法源数据构建完整组件；不读取离线强权重或评价标签。"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.source_combinations import SourceCombinations
from harness.models.complete_checkpoint import migrate_engine
from harness.models.robust_residual_network import RobustResidualNetwork


def prepare_runtime():
    """受限源码运行前载入优化器依赖；预热不接触任务数据或任务模型。"""
    torch.set_num_threads(1)
    pd.DataFrame([[0., 1.]], columns=['signal', 'parent'])[['signal']].to_numpy(dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        layer = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(layer.parameters())
        layer(torch.zeros(1, 1)).sum().backward()
        optimizer.step()


def attach_inputs(base, x, groups, indices, decimals):
    selected, healthy = [], []
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        if len(rows) < 8:
            raise ValueError('每个源设备至少需要8条有序训练记录')
        healthy.extend(rows[:8])
        selected.extend(rows[np.linspace(0, len(rows)-1, min(8, len(rows))).astype(int)])
    selected = np.sort(selected)
    repair = SourceCombinations(x[selected])
    condition = ConditionStandardScaler(tuple(indices), tuple(decimals)).fit(x[healthy])
    center = StandardScaler().fit(condition.transform(x[selected]))
    return migrate_engine(base, repair, condition, center, provenance={
        'construction': 'actual_parent_and_legal_training_only', 'source_rows': len(x),
        'anchor_rows': len(selected), 'healthy_rows': len(healthy)})


def attach_residual(model, x, y, groups, epochs):
    native = model.native_features(x)
    parent = model.base.predict(native * model.scale + model.mean)
    width = x.shape[1]
    columns = [f'input_{i}' for i in range(width)] + [f'native_{i}' for i in range(width)] + ['parent_value']
    frame = pd.DataFrame(np.column_stack((x, native, parent)), columns=columns)
    residual = RobustResidualNetwork(epochs=epochs).fit(frame, y, groups=groups)
    arrays = model.auxiliary_arrays
    arrays.update(residual_mean=residual.mean_.copy(), residual_std=residual.std_.copy())
    for name, value in residual.net_.state_dict().items():
        arrays['residual_' + name.replace('.', '_')] = value.detach().numpy().copy()
    for (name, _), state in zip(residual.net_.named_parameters(), residual.optimizer_state_['state'].values(), strict=True):
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            arrays['residual_optimizer_' + name.replace('.', '_') + '_' + key] = state[key].detach().numpy().copy()
    arrays['residual_random_state'] = residual.random_state_.numpy().astype(np.int64)
    model.state['residual'] = {'include_parent': True, 'lo': float(residual.lo_), 'hi': float(residual.hi_),
        'training': 'source_trained_optimizer_saved', 'epochs': epochs, 'optimizer': 'AdamW',
        'learning_rate': .002, 'weight_decay': .001, 'seed': 17,
        'continuation_policy': 'freeze_head_and_continue_parent_through_full_output'}
    arrays['complete_state'] = np.array(json.dumps(model.state, sort_keys=True, separators=(',', ':')))
    model.validate()
    return {'component': 'residual', 'epochs': epochs, 'anchors': residual.anchors_, 'optimizer_saved': True}
