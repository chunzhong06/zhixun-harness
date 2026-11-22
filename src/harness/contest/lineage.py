"""把平台实际传入的父版与历史发布记录对应，避免用模型名推断继承关系。"""
from hashlib import sha256
import json

import numpy as np


def parent_handoff(parent_sha, history):
    """历史只能解释来历；即使平台回退或传来外部父版，也不替换实际输入。"""
    previous = history[-1] if history else None
    matches = [row.get('platform_stage_id', row.get('stage')) for row in history
               if row.get('published_sha256') == parent_sha]
    previous_sha = previous.get('published_sha256') if previous else None
    relation = ('first_observed_round' if previous is None else
                'previous_publication' if parent_sha == previous_sha else
                'earlier_publication' if matches else 'different_supplied_parent')
    return {'actual_parent_sha256': parent_sha, 'previous_publication_sha256': previous_sha,
            'relation': relation, 'matching_history_stages': matches}


def previous_score_context(parent_sha, validation_sha, before, history):
    """只有同权重、同评价文件的相邻分数才直接核对；变化集合在本轮重新比较。"""
    if not history:
        return {'comparison': 'no_previous_round'}
    previous = history[-1]
    previous_validation = (previous.get('data_identity') or {}).get('validation_sha256')
    if previous.get('published_sha256') != parent_sha:
        comparison = 'different_parent'
    elif previous_validation is None:
        comparison = 'previous_validation_unknown'
    elif previous_validation != validation_sha:
        comparison = 'changed_validation'
    else:
        comparison = 'same_parent_same_validation'
    result = {'comparison': comparison, 'previous_validation_sha256': previous_validation,
              'current_validation_sha256': validation_sha}
    previous_metric = ((previous.get('selection') or {}).get('candidate') or {}).get('score')
    if comparison == 'same_parent_same_validation' and previous_metric is not None:
        key = 'macro_f1' if 'macro_f1' in before else 'rmse'
        result.update(metric=key, previous_published_score=previous_metric,
                      current_parent_score=before[key], difference=before[key]-previous_metric)
    return result


def checkpoint_state(path):
    """分别核对权重、Adam与尺度；整个文件身份仍由调用方单独记录。"""
    def digest(arrays, keys, extra=None):
        value = sha256()
        for key in sorted(keys):
            array = arrays[key]
            value.update(json.dumps([key, array.dtype.str, array.shape], separators=(',', ':')).encode())
            value.update(array.tobytes(order='C'))
        if extra is not None:
            value.update(json.dumps(extra, sort_keys=True, separators=(',', ':')).encode())
        return value.hexdigest()

    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays['metadata']))
        parameters = ('w1', 'b1', 'w2', 'b2')
        optimizer = [prefix+name for prefix in ('adam_first_', 'adam_second_') for name in parameters]
        base = {*parameters, *optimizer, 'mean', 'scale', 'metadata'}
        return {'parent_weights_sha256': digest(arrays, parameters),
                'parent_adam_sha256': digest(arrays, optimizer),
                'scaling_sha256': digest(arrays, ('mean', 'scale'),
                    {key: metadata[key] for key in ('paths', 'target_mean', 'target_scale')}),
                'auxiliary_state_sha256': digest(arrays, set(arrays.files)-base),
                'epochs': metadata['epochs'], 'steps': metadata['steps']}
