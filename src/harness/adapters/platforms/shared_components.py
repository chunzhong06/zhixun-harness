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
    """显式资源仍由原接口核验；默认资源遇到不适配的任务或新资料时不启用。"""
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
    source = {x.tobytes()+np.float64(y).tobytes() for x, y in zip(data['x'], data['y'], strict=True)}
    with np.load(path, allow_pickle=False) as arrays:
        prefix = 'classifier' if profile == 'A' else 'readout'
        compatible = all(x.astype(np.float64).tobytes()+np.float64(y).tobytes() in source
                         for x, y in zip(arrays[prefix+'_source_x'], arrays[prefix+'_source_y'], strict=True))
    if not compatible:
        print('共享教师源上下文未全部包含在本轮合法训练资料内，本轮沿用普通父版续训。', file=sys.stderr)
        return None
    return {'checkpoint':str(path), 'sha256':selected['sha256']}
