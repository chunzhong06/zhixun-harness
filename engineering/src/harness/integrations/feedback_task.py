"""将任务宿主已收到的合法反馈打包给共同源码执行器；默认仍只有源训练标签。"""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import cloudpickle
import pandas as pd
from harness.runtime.storage import atomic_json


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare_feedback_input(bundle, destination, ledger, target_features, validation_ids, confirmation_ids):
    """从已回标账本导出训练包，评价ID始终排除；没有未来答案参数。

    target_features只含无标签输入；validation/confirmation提供输入身份。
    源模型及源资料须由任务宿主验证来历。训练清单随执行器在运行生成代码前
    固定；生成代码改写文件后，新接口拒绝使用改变的内容。
    """
    bundle, destination = Path(bundle), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    task = json.loads((bundle / 'task/task.json').read_bytes())
    source = pd.read_csv(bundle / 'task/train.csv')
    columns = [k for k in source.columns if k != 'target']
    if ledger.source_ids != task['train_ids']:
        raise ValueError('源训练身份和反馈账本不一致')
    validation_ids, confirmation_ids = list(validation_ids), list(confirmation_ids)
    held_out = validation_ids + confirmation_ids
    if len(set(held_out)) != len(held_out) or set(held_out).intersection(ledger.source_ids):
        raise ValueError('探索、确认和源训练身份必须互不重叠')
    for name in ['train.csv', 'task.json']:
        shutil.copyfile(bundle / 'task' / name, destination / name)
    for name, ids in [('validation.csv', validation_ids), ('test.csv', confirmation_ids)]:
        pd.DataFrame([target_features[i] for i in ids], columns=columns).to_csv(destination / name, index=False)
    if not ledger.allow_feedback_training:
        # 官方/评分专用任务不提供父模型附加入口或任何反馈标签文件。
        atomic_json(destination / 'feedback-training.json', {'schema_version': 1, 'enabled': False})
        return {'enabled': False, 'feedback_rows': 0}
    state = ledger.state()
    task['source_description'] = task.get('description', '')
    task['description'] = ('当前任务显式允许使用已返回且列入feedback.csv的标签训练。'
                           '探索和确认身份已从训练包排除，评价标签留在外层；原数据划分说明仅用于追溯。')
    atomic_json(destination / 'task.json', task)
    ids, labels, batches = [], [], []
    for batch, feedback in enumerate(state['feedback'], 1):
        for i, y in zip(feedback['ids'], feedback['labels'], strict=True):
            if i not in held_out:
                ids.append(i); labels.append(y); batches.append(batch)
    feedback = pd.DataFrame([target_features[i] for i in ids], columns=columns)
    feedback['target'] = labels
    feedback.to_csv(destination / 'feedback.csv', index=False)
    atomic_json(destination / 'feedback-identities.json', {'ids': ids, 'batches': batches})
    atomic_json(destination / 'source-identities.json', {'ids': ledger.source_ids})
    shutil.copyfile(bundle / 'parent.pkl', destination / 'parent.pkl')
    parent = cloudpickle.loads((destination / 'parent.pkl').read_bytes())
    classes = [v.item() if hasattr(v, 'item') else v for v in getattr(parent, 'classes_', [])]
    files = ['train.csv', 'feedback.csv', 'feedback-identities.json', 'source-identities.json', 'parent.pkl']
    manifest = {'schema_version': 1, 'enabled': True,
                'policies': ['source_only', 'cumulative_refit', 'recent_mix_refit'],
                'feature_columns': columns, 'parent_feature_columns': [c for c in columns if c != 'parent_value'],
                'feedback_rows': len(ids), 'source_rows': len(source),
                'parent_capabilities': {'type': type(parent).__name__, 'classes': classes,
                                        'predict_proba': hasattr(parent, 'predict_proba')},
                'policy_definitions': {
                    'source_only': '保持源父模型，不进行新拟合。',
                    'cumulative_refit': '源加全部合法反馈；每行sample_weight都是1。',
                    'recent_mix_refit': '源加最近两批合法反馈；源每行权重1，反馈总权重等于源总权重。'},
                'held_out_ids': held_out, 'observed_batches': len(state['feedback']),
                'files': {name: checksum(destination / name) for name in files}}
    atomic_json(destination / 'feedback-training.json', manifest)
    return manifest


def reference_sources():
    """对驱动公开完整程序；重训采用哪个反馈范围由程序选择而非数据集名称。"""
    prefix = '''from harness_task import load_feedback_training,load_parent_model,evaluate_model,training_access
from harness.models.feedback_training import fit_feedback_parent
access=training_access()
parent=load_parent_model()
'''
    def program(policies):
        return prefix + 'for policy in ' + repr(policies) + ''':
    X,y,weights,provenance=load_feedback_training(policy)
    model=fit_feedback_parent(parent,X,y,weights,access['parent_feature_columns'],provenance['feedback_rows'])
    evaluate_model(model,label=policy)
'''
    return {**{a: program([a]) for a in ['source_only', 'cumulative_refit', 'recent_mix_refit']},
            'feedback-comparison': program(['source_only', 'cumulative_refit', 'recent_mix_refit'])}


def describe_training_access(folder):
    path = Path(folder) / 'feedback-training.json'
    enabled = path.exists() and json.loads(path.read_bytes()).get('enabled') is True
    if not enabled:
        return '当前任务只允许源训练标签；反馈可用于外层评价，不能调用反馈训练工具。'
    record = json.loads(path.read_bytes())
    return ('反馈训练已由任务宿主开放。load_feedback_training(policy)返回X、y、sample_weights、provenance；'
            '可选source_only、cumulative_refit、recent_mix_refit。训练行已排除探索和确认身份；'
            f"可用反馈{record['feedback_rows']}行。load_parent_model()返回实际源父模型，training_access()给出输入列。"
            'cumulative_refit的sample_weights全部为1，对它做统一化或平方根变换不会改变训练。'
            'recent_mix_refit只保留最近两批，反馈总权重与源总权重相同；模型内部class_weight仍独立生效。'
            f"父后端能力：{json.dumps(record.get('parent_capabilities', {}), ensure_ascii=False)}。"
            'fit_feedback_parent返回带predict的可序列化模型；分类后端也提供predict_proba和classes_。'
            '概率列必须通过model.classes_[np.argmax(probability,axis=1)]映射回真实类别，不能直接交付列索引。'
            '新模型通过evaluate_model保存并独立重载；真正评价由外层完成。')
