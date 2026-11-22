"""可编辑的通用续训程序；按任务类型和训练组工作，不嵌入数据集名称或验证分数。"""


REFERENCE_PROGRAMS = {
    'official-one-epoch': '''def train(session):
    session.train_epoch()
''',
    'stable-fine-tuning': '''def train(session):
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001)
''',
    'balanced-fine-tuning': '''import numpy as np
def train(session):
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(4):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'expanded-balanced-fine-tuning': '''import numpy as np
def train(session):
    metadata = session.metadata
    outputs = len(metadata['classes']) if metadata['task'] == 'classification' else 1
    capacity = min(128, (99999 - outputs) // (len(metadata['paths']) + outputs + 1))
    if metadata['hidden'] < capacity:
        session.expand_hidden(capacity)
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(4):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'extended-balanced-fine-tuning': '''import numpy as np
def train(session):
    metadata = session.metadata
    outputs = len(metadata['classes']) if metadata['task'] == 'classification' else 1
    capacity = min(64, (99999 - outputs) // (len(metadata['paths']) + outputs + 1))
    if metadata['hidden'] < capacity:
        session.expand_hidden(capacity)
    identity = session.targets if metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    'long-balanced-fine-tuning': '''import numpy as np
def train(session):
    identity = session.targets if session.metadata['task'] == 'classification' else session.groups
    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for epoch in range(12):
        session.train_epoch(learning_rate=0.001, sample_weights=weights)
''',
    # 来源：真实Luna在官方轴承源训练上的候选，经全公开验证及独立交付复验。
    # 作为可比较工具保留；不将一次开发集收益推广成所有回归任务的固定选择。
    'expanded-balanced-huber-fine-tuning': '''import numpy as np
def train(session):
    metadata = session.metadata
    outputs = len(metadata['classes']) if metadata['task'] == 'classification' else 1
    capacity = min(64, (99999 - outputs) // (len(metadata['paths']) + outputs + 1))
    if metadata['hidden'] < capacity:
        session.expand_hidden(capacity)
    _, inverse, counts = np.unique(session.groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    def huber_gradient(logits, target):
        error = logits[:, 0] - target
        absolute = np.abs(error)
        threshold = 1.0
        loss = np.where(absolute <= threshold, 0.5 * error ** 2, threshold * (absolute - 0.5 * threshold))
        gradient = np.clip(error, -threshold, threshold)[:, None]
        return loss, gradient
    for epoch in range(24):
        session.train_epoch(learning_rate=0.001, sample_weights=weights, loss_gradient=huber_gradient)
''',
}


def reference_programs(task=None):
    """调用方取得副本并冻结源码身份；现场仍须实际执行、比较和决定采用。"""
    return {name: source for name, source in REFERENCE_PROGRAMS.items()
            if name != 'expanded-balanced-huber-fine-tuning' or task == 'bearing'}


ENGINE_UNIFORM = ('def train(session):\n'
                  '    for epoch in range(4):\n'
                  '        session.train_epoch(learning_rate=0.0005, batch_size=256)\n')


def engine_reference_programs(diagnostics, capabilities):
    programs = {}
    action = capabilities.get('engine_parent_pullback_action')
    if action in ('apply','remove'):
        name = '发动机父网有界回调与续训' if action == 'apply' else '发动机撤除父网回调并续训'
        programs[name] = ('def train(session):\n    session.train_epoch(learning_rate=0.000005, batch_size=256)\n'
                          f'    session.set_engine_parent_pullback(enabled={action == "apply"})\n')
    if capabilities.get('engine_life_limit_available'):
        programs['发动机源寿命上限与父版续训'] = (
            'def train(session):\n    session.train_epoch(learning_rate=0.000005, batch_size=256)\n    session.set_engine_life_limit()\n')
    if capabilities.get('engine_life_limit_removable'):
        programs['发动机解除上限并续训'] = (
            'def train(session):\n    session.set_engine_life_limit(enabled=False)\n    session.train_epoch(learning_rate=0.000005, batch_size=256)\n')
    if capabilities.get('engine_setting_alignment_available'):
        programs['发动机按父预测校准工况编码'] = (
            'def train(session):\n    session.build_engine_setting_alignment()\n    session.train_epoch(learning_rate=0.0001)\n')
    if capabilities.get('engine_neighbors_available'):
        programs['发动机同工况近邻与父版续训'] = (
            'def train(session):\n    session.train_epoch(learning_rate=0.000005, batch_size=256)\n    session.build_engine_life_neighbors()\n')
    elif capabilities.get('engine_condition_density_available'):
        programs['发动机分工况寿命密度与父版续训'] = (
            'def train(session):\n    session.build_engine_condition_density()\n    session.train_epoch(learning_rate=0.000005, batch_size=256)\n')
    return programs


def starting_reference(task, references, capabilities):
    """按此前有效组件预定一次实际起点，不读取本轮确认或服务答案。"""
    epochs = capabilities['max_epochs']
    if task == 'gas':
        names = ('气体缺类样本与父版联合读出','气体新设备上下文与父版续训','气体响应尺度与源支持续训','气体甲苯复核与父版续训','气体本底判别与父版续训','气体本底集成父版接续','气体本底集成与源修正续训','TabPFN分类组合父版接续','TabPFN分类组合与源修正续训',
                 'TabICLv2分类组合父版接续','TabICLv2分类组合与源修正续训') + (('balanced-source-shape-repair',) if epochs >= 2 else ()) + ('source-shape-confusion-repair', 'gas-parent-readout-continuation')
    elif task == 'bearing' and capabilities.get('bearing_grid_readout_available'):
        names = ('轴承新采集窗口校准与父版续训',)
    elif task == 'bearing' and capabilities.get('tabpfn_regression_parent'):
        names = ('轴承新采集窗口校准与父版续训','轴承能量冲击上界与父版续训','TabM按特征与多头分歧组合续训','TabM按源特征范围组合续训','TabM与完整回归父版联合续训','TabPFN回归组合父版接续')
    elif task == 'bearing' and epochs >= 25 and capabilities.get('tabpfn_regression_component'):
        names = ('TabPFN原始读出与父版续训',)
    elif task == 'bearing' and epochs >= 25 and capabilities.get('tabicl_source_component'):
        names = ('TabICLv2原始读出与父版续训',)
    elif task == 'bearing' and epochs >= 24 and capabilities.get('build_components_before_first_epoch'):
        names = ('expanded-balanced-huber-fine-tuning',)
    elif task == 'bearing' and epochs >= 12 and capabilities.get('build_components_before_first_epoch'):
        names = ('extended-balanced-fine-tuning',)
    elif task == 'engine' and capabilities.get('engine_setting_alignment_available'):
        names = ('发动机按父预测校准工况编码',)
    elif task == 'engine' and capabilities.get('engine_neighbors_available'):
        names=('发动机同工况近邻与父版续训',)
    elif task == 'engine' and capabilities.get('engine_after_uniform') and capabilities.get('engine_condition_density_available'):
        names = ('发动机分工况寿命密度与父版续训','发动机按行等权续训四轮')
    elif task == 'engine' and epochs >= 4 and capabilities.get('anonymous_engine_parent'):
        names = ('发动机按行等权续训四轮',)
    elif task == 'engine' and epochs >= 60 and capabilities.get('build_components_before_first_epoch'):
        names = ('源工况组件后均衡续训', '源工况完整父版续训')
    else:
        names = ()
    return next((name for name in (*names, 'official-one-epoch') if name in references), None)


def bearing_reference_programs(diagnostics, capabilities):
    """按实际资源选择作者后端；新组合保留已有父网，后续只接续完整模型。"""
    programs = {}
    if capabilities.get('bearing_grid_readout_available'):
        programs['轴承新采集窗口校准与父版续训'] = (
            'def train(session):\n    session.train_epoch(learning_rate=0.0001)\n    session.build_bearing_grid_readout()\n')
    if capabilities.get('tabpfn_regression_parent'):
        if capabilities.get('bearing_health_cap_available'):
            programs['轴承能量冲击上界与父版续训']=(
                'def train(session):\n    session.build_bearing_health_cap()\n    session.train_epoch(learning_rate=0.0001)\n')
        programs['TabPFN回归组合父版接续'] = 'def train(session):\n    session.train_epoch(learning_rate=0.0001)\n'
        if capabilities.get('tabm_readout_available'):
            programs['TabM与完整回归父版联合续训'] = (
                'def train(session):\n    session.build_tabm_readout()\n    session.train_epoch(learning_rate=0.0001)\n')
        if capabilities.get('tabm_source_support_available'):
            programs['TabM按源特征范围组合续训'] = (
                'def train(session):\n    session.build_tabm_source_support()\n    session.train_epoch(learning_rate=0.0001)\n')
        if capabilities.get('tabm_head_support_available'):
            programs['TabM按特征与多头分歧组合续训'] = (
                'def train(session):\n    session.build_tabm_source_support(head_spread=True)\n    session.train_epoch(learning_rate=0.0001)\n')
    if capabilities['max_epochs'] >= 25:
        if capabilities.get('tabpfn_regression_component'):
            name, method = 'TabPFN原始读出与父版续训', 'build_tabpfn_readout'
            weight = float(capabilities.get('regressor_parent_weight', .25))
        elif capabilities.get('tabicl_source_component'):
            name, method, weight = 'TabICLv2原始读出与父版续训', 'build_tabicl_readout', .25
        else:
            return programs
        programs[name] = (REFERENCE_PROGRAMS['expanded-balanced-huber-fine-tuning'] +
            f'\n    session.{method}(parent_weight={weight!r})\n' +
            '    session.train_epoch(learning_rate=0.0001)\n' +
            ('    session.train_epoch(learning_rate=0.0001)\n'
             if method == 'build_tabicl_readout' and capabilities['max_epochs'] >= 26 else ''))
    return programs


def gas_reference_programs(diagnostics, capabilities):
    """按合法探索混淆提出一个源修正，或复用已有父分量；不读取确认标签。"""
    programs = {}
    if capabilities.get('gas_partial_context_available'):
        programs['气体缺类样本与父版联合读出'] = (
            "def train(session):\n    session.build_gas_partial_context(view='log', protect_missing=True)\n    session.train_epoch(learning_rate=0.0001)\n")
        programs['气体已见类别中心与父版续训'] = (
            "def train(session):\n    session.build_gas_partial_context(view='log', protect_missing=False)\n    session.train_epoch(learning_rate=0.0001)\n")
    if capabilities.get('gas_source_repair'):
        confusion = diagnostics['confusion_true_rows_prediction_columns']
        count, actual, predicted = max((confusion[a][p], a, p) for a in range(6) for p in range(6) if a != p)
        if count:
            programs['source-shape-confusion-repair'] = (
                'def train(session):\n'
                f'    session.build_gas_repair({predicted+1}, {actual+1})\n'
                '    session.train_epoch(learning_rate=0.001)\n')
            # Luna有效程序中的类均衡与少量目标类加权，修正方向仍取合法评价混淆。
            programs['balanced-source-shape-repair'] = (
                'import numpy as np\n'
                'def train(session):\n'
                f'    session.build_gas_repair({predicted+1}, {actual+1})\n'
                '    _, inverse, counts = np.unique(session.targets, return_inverse=True, return_counts=True)\n'
                '    weights = 1.0 / counts[inverse]\n'
                f'    weights[session.targets == {actual}] *= 1.15\n'
                '    weights /= np.mean(weights)\n'
                '    for _ in range(2):\n'
                '        session.train_epoch(learning_rate=0.0005, sample_weights=weights)\n')
    if 'parent' in capabilities.get('gas_readout_choices', []):
        programs['gas-parent-readout-continuation'] = (
            "def train(session):\n    session.set_gas_policy('parent')\n"
            '    session.train_epoch(learning_rate=0.001)\n')
    if capabilities.get('tabicl_classification_component'):
        attachment = ('    session.build_tabicl_classifier(parent_weight=0.25)\n'
                      '    session.train_epoch(learning_rate=0.0001)\n')
        if 'balanced-source-shape-repair' in programs:
            if capabilities['max_epochs'] >= 3:
                programs['TabICLv2分类组合与源修正续训'] = programs['balanced-source-shape-repair'] + attachment
        else:
            programs['TabICLv2分类组合与源修正续训'] = 'def train(session):\n' + attachment
    if capabilities.get('tabicl_classification_parent'):
        # 冻结教师与原源修正，保留实际父版的网络、Adam和尺度继续更新。
        programs['TabICLv2分类组合父版接续'] = 'def train(session):\n    session.train_epoch(learning_rate=0.0001)\n'
    if capabilities.get('tabpfn_classification_component'):
        # 比例随已核验资源提供；现场程序仍可显式比较，旧资源保持原0.25起点。
        parent_weight=float(capabilities.get('classifier_parent_weight',.25))
        attachment=(f'    session.build_tabpfn_classifier(parent_weight={parent_weight!r})\n'
                    '    session.train_epoch(learning_rate=0.0001)\n')
        if 'balanced-source-shape-repair' in programs:
            if capabilities['max_epochs']>=3:
                programs['TabPFN分类组合与源修正续训']=programs['balanced-source-shape-repair']+attachment
        else:
            programs['TabPFN分类组合与源修正续训']='def train(session):\n'+attachment
    if capabilities.get('tabpfn_classification_parent'):
        programs['TabPFN分类组合父版接续']='def train(session):\n    session.train_epoch(learning_rate=0.0001)\n'
        if capabilities.get('gas_toluene_readout_available'):
            programs['气体甲苯复核与父版续训']=programs['TabPFN分类组合父版接续']+'    session.build_gas_toluene_readout()\n'
        if capabilities.get('gas_device_context_available'):
            programs['气体新设备上下文与父版续训'] = programs['TabPFN分类组合父版接续']+'    session.build_gas_device_context()\n'
        if capabilities.get('gas_kinetic_support_available'):
            start=programs.get('气体甲苯复核与父版续训',programs['TabPFN分类组合父版接续'])
            programs['气体响应尺度与源支持续训']=start+'    session.build_gas_kinetic_support()\n'
    if capabilities.get('gas_baseline_context_available'):
        if capabilities.get('tabpfn_classification_parent'):
            programs['气体本底集成父版接续']=('def train(session):\n    session.build_gas_baseline_ensemble()\n'
                                            '    session.train_epoch(learning_rate=0.0001)\n')
        elif capabilities.get('tabpfn_classification_component') and 'TabPFN分类组合与源修正续训' in programs:
            programs['气体本底集成与源修正续训']=programs['TabPFN分类组合与源修正续训'].replace(
                'session.build_tabpfn_classifier(', 'session.build_gas_baseline_ensemble(')
    if capabilities.get('gas_drift_readout_available'):
        for name in ('气体本底集成父版接续','气体本底集成与源修正续训','TabPFN分类组合父版接续'):
            if name in programs:
                programs['气体本底判别与父版续训']=programs[name]+'    session.build_gas_drift_readout()\n'
                break
    return programs
