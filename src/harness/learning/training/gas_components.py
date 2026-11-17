"""用合法源训练构建气体形状修正，或改变已有组件的读出方式。"""
from copy import deepcopy
import json
import numpy as np

from harness.learning.inference.complete_checkpoint import FORMAT, GasVoteModel
from harness.learning.inference.gas_views import build_pair_memory, pair_model


def attach_repair(base, x, y, reported, target):
    """保留实际父网络及Adam；源参照只决定指定类别分歧的修正。"""
    if (base.meta['task'] != 'classification' or len(base.meta['classes']) != 6
            or len(base.meta['paths']) != 128 or hasattr(base, 'auxiliary_arrays')):
        raise ValueError('源形状修正需要普通官方六类气体父版')
    if (type(reported) is not int or type(target) is not int
            or reported not in range(1, 7) or target not in range(1, 7) or reported == target):
        raise ValueError('修正方向须是不同的六类编号')
    if 'gas_pair_patch' in base.meta:
        raise ValueError('已有修正规则不能被新构建动作覆盖')
    model_class = pair_model(type(base))
    wrapped = model_class.__new__(model_class)
    wrapped.__dict__ = deepcopy(base.__dict__)
    wrapped.meta['gas_pair_patch'] = build_pair_memory(
        {'method': 'shape-blend', 'reported': reported, 'target': target}, x, y, base.mean, base.scale)
    state = {'format': FORMAT, 'kind': 'gas-source-vote', 'policy': 'parent',
             'training_objective': 'native_classifier_surrogate',
             'provenance': {'construction': 'actual_parent_and_legal_training_only',
                            'training_rows': len(y), 'target_feedback_rows': 0}}
    return GasVoteModel(wrapped, {'complete_state': np.array(json.dumps(state, sort_keys=True, separators=(',', ':')))})


def set_policy(model, policy):
    """仅改变读出；原父参数与源分量保留，缺少投票分量时不能启用多数投票。"""
    if not isinstance(model, GasVoteModel) or policy not in ('parent', 'majority'):
        raise ValueError('读出选择需要完整气体父版及已知策略')
    old = deepcopy(model.state)
    model.state['policy'] = policy
    try:
        model.validate()
    except Exception:
        model.state = old
        raise
    model.auxiliary_arrays['complete_state'] = np.array(json.dumps(model.state, sort_keys=True, separators=(',', ':')))
    return {'component': 'gas_readout', 'before': old.get('policy', 'majority'), 'after': policy}
