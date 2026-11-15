"""作者生成程序使用的任务接口：只提供训练标签，评分标签留在父进程。"""
from hashlib import sha256
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


_training_binding = None


def bind_training_access(folder):
    """由执行器在生成程序开始前绑定任务权限和文件身份；缺少声明时关闭。"""
    global _training_binding
    if _training_binding is not None:
        raise RuntimeError('本进程的训练权限已经绑定，生成程序不能重新声明')
    root = Path(folder).resolve()
    path = root / 'feedback-training.json'
    record = json.loads(path.read_bytes()) if path.exists() else {'schema_version': 1, 'enabled': False}
    if record.get('schema_version') != 1 or type(record.get('enabled')) is not bool:
        raise ValueError('任务训练权限声明无效')
    if record['enabled']:
        required = {'train.csv', 'feedback.csv', 'feedback-identities.json', 'source-identities.json', 'parent.pkl'}
        if set(record.get('files', {})) != required:
            raise ValueError('反馈训练输入清单不完整')
        for name, expected in record['files'].items():
            if sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError('反馈训练输入身份不符')
    _training_binding = (root, record)


def training_access():
    """提供驱动可见能力，不允许通过返回值修改执行器已绑定的权限。"""
    if _training_binding is None:
        return {'enabled': False}
    _, record = _training_binding
    return json.loads(json.dumps({k: v for k, v in record.items() if k not in {'files', 'held_out_ids'}}))


def _feedback_input():
    if _training_binding is None or not _training_binding[1]['enabled']:
        raise PermissionError('当前任务未允许反馈标签训练')
    root, record = _training_binding
    for name, expected in record['files'].items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('训练文件在执行前绑定后被改动')
    return root, record


def load_parent_model():
    root, _ = _feedback_input()
    return cloudpickle.loads((root / 'parent.pkl').read_bytes())


def load_feedback_training(policy='cumulative_refit'):
    """只返回宿主已授权训练行；探索和确认标签没有打包进当前进程。"""
    root, record = _feedback_input()
    if policy not in record['policies']:
        raise ValueError('当前任务不支持该反馈训练策略')
    source = pd.read_csv(root / 'train.csv')
    source_y = source.pop('target').to_numpy()
    feedback = pd.read_csv(root / 'feedback.csv', float_precision='round_trip')
    feedback_y = feedback.pop('target').to_numpy()
    feedback_ids = json.loads((root / 'feedback-identities.json').read_bytes())
    source_ids = json.loads((root / 'source-identities.json').read_bytes())['ids']
    ids, batches = feedback_ids['ids'], np.asarray(feedback_ids['batches'])
    if (list(source.columns) != record['feature_columns'] or list(feedback.columns) != list(source.columns)
            or len(ids) != len(feedback) or len(batches) != len(ids) or len(source_ids) != len(source)
            or len(set(source_ids + ids)) != len(source_ids) + len(ids)
            or set(ids).intersection(record['held_out_ids'])):
        raise ValueError('训练输入身份、用途或列不一致')
    mask = np.zeros(len(ids), dtype=bool) if policy == 'source_only' else np.ones(len(ids), dtype=bool)
    if policy == 'recent_mix_refit':
        # 依真实批次序号取最近两批；空批次不会把更老的样本伪装成近期。
        mask &= batches > record['observed_batches'] - 2
    selected = np.flatnonzero(mask)
    n = len(selected)
    weight = len(source)/n if n and policy == 'recent_mix_refit' else 1.
    features = pd.concat((source, feedback.iloc[selected]), ignore_index=True)
    targets = np.r_[source_y, feedback_y[selected]]
    weights = np.r_[np.ones(len(source)), np.full(n, weight)]
    provenance = {'policy': policy, 'source_rows': len(source), 'feedback_rows': n,
                  'source_ids': source_ids, 'feedback_ids': [ids[i] for i in selected]}
    return features, targets, weights, provenance


def load_data():
    """返回合法训练和无标签的探索、确认输入；这些文件不含未来服务输入。"""
    train=pd.read_csv('input/train.csv');target=train.pop('target').to_numpy()
    return train,target,pd.read_csv('input/validation.csv'),pd.read_csv('input/test.csv')


def _predict_rows(model,frame):
    values=[]
    for i in range(len(frame)):
        prediction=np.asarray(model.predict(frame.iloc[[i]]),dtype=float)
        if prediction.shape!=(1,) or not np.isfinite(prediction).all():raise ValueError('模型必须对当前一行返回一个有限预测')
        values.append(max(0.,float(prediction[0])))
    return np.asarray(values)


def evaluate_model(model,label='candidate',feedback_gain=.5,feedback_hours=6.):
    """保存实际模型和逐行预测；真实评分由外层合法反馈执行器追加到运行回执。"""
    folder=Path('artifacts');folder.mkdir(exist_ok=True)
    number=len(list(folder.glob('candidate-*.json')))+1;stem=f'candidate-{number:03d}'
    if Path('input/native-contract.json').exists():
        from harness.integrations.native_author_task import NativeCandidate
        if not isinstance(model, NativeCandidate):
            raise ValueError('比赛候选必须调用train_candidate(program)真实接续父模型；不能另建替代模型')
        (folder/(stem+'-checkpoint.npz')).write_bytes(model.checkpoint)
        (folder/(stem+'-training.receipt')).write_text(json.dumps(model.report,ensure_ascii=False),encoding='utf8')
    _,_,validation,test=load_data()
    raw=cloudpickle.dumps(model);(folder/(stem+'.pkl')).write_bytes(raw)
    # 先保存再重载，确保交付的正是可重新使用的权重，而不是进程内临时对象。
    restored=cloudpickle.loads(raw)
    before=_predict_rows(restored,validation);confirm=_predict_rows(restored,test)
    classes = training_access().get('parent_capabilities', {}).get('classes', [])
    if classes and (not np.isin(before, classes).all() or not np.isin(confirm, classes).all()):
        raise ValueError('候选预测必须使用任务真实类别；概率列索引须先映射到classes_')
    np.savez_compressed(folder/(stem+'.npz'),validation=before,confirmation=confirm)
    record={'label':str(label),'weights_sha256':sha256(raw).hexdigest(),
            'feedback':{'gain':float(feedback_gain),'hours':float(feedback_hours)},
            'validation_rows':len(before),'confirmation_rows':len(confirm)}
    (folder/(stem+'.json')).write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'HARNESS_EVALUATION:{stem}',flush=True)
    return stem


def train_candidate(program):
    from harness.integrations.native_author_task import train_candidate as train
    return train(program)


def submit_model(model,feedback_gain=.5,feedback_hours=6.):
    """完成作者要求的提交文件，同时保留可用于逐小时服务的实际模型。"""
    stem=evaluate_model(model,'final_submission',feedback_gain,feedback_hours)
    with np.load(Path('artifacts')/(stem+'.npz'),allow_pickle=False) as arrays:values=arrays['confirmation']
    Path('final').mkdir(exist_ok=True)
    pd.DataFrame({'prediction':values}).to_csv('final/submission.csv',index=False)
    return stem
