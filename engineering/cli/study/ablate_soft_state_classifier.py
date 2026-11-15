"""固定分类消融与父预测捷径诊断；全部预测封存后统一开分，不调用研究模型。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness.integrations.broad_transfer_task import accepted, metric
from harness.models.soft_state_classifier import SoftStateClassifier
from harness.runtime.storage import atomic_json

VARIANTS = {
    'full': {},
    'uniform_gate': {'learned_gate': False},
    'single_expert': {'n_experts': 1, 'learned_gate': False},
    'without_parent': {'include_parent': False},
    'without_auxiliary_regularization': {'entropy_weight': 0., 'consistency_weight': 0.},
    'without_weight_decay': {'weight_decay': 0.},
}
SEEDS = [29, 17, 41]
TASKS = ['sgp40', 'secom']
PHASE_FILES = {'explore': 'validation.csv', 'confirm': 'test.csv'}


def model_for(spec, columns):
    if spec['family'] == 'soft_state':
        return SoftStateClassifier(seed=spec['seed'], **VARIANTS[spec['variant']])
    names = list(columns) if spec['include_parent'] else [c for c in columns if c != 'parent_value']
    select = ColumnTransformer([('features', 'passthrough', names)], remainder='drop')
    if spec['family'] == 'logistic':
        return make_pipeline(select, StandardScaler(),
                             LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=17))
    if spec['family'] == 'forest':
        return make_pipeline(select, RandomForestClassifier(n_estimators=96, max_depth=6,
                             class_weight='balanced', n_jobs=1, random_state=17))
    raise ValueError('未冻结的模型族')


def worker(folder):
    """子进程只读取合法训练标签和无标签预测输入，90秒上限由父进程负责。"""
    import torch
    torch.set_num_threads(1)
    spec = json.loads((folder/'spec.json').read_bytes())
    task = Path(spec['task_input'])
    train = pd.read_csv(task/'train.csv')
    y = train.pop('target').to_numpy()
    frames = {'train': train, **{phase: pd.read_csv(task/name) for phase, name in PHASE_FILES.items()},
              'future': pd.read_csv(folder.parent/'service.csv')}
    started = time.monotonic()
    if spec['family'] == 'existing':
        # 本地已封存的可信实验权重；无需重新训练或改写其中的类。
        model_bytes = Path(spec['weights']).read_bytes()
        if sha256(model_bytes).hexdigest() != spec['weights_sha256']:
            raise ValueError('原模型身份改变')
        model = cloudpickle.loads(model_bytes)
    else:
        model = model_for(spec, train.columns).fit(train, y)
        model_bytes = cloudpickle.dumps(model)
    (folder/'model.pkl').write_bytes(model_bytes)
    model = cloudpickle.loads(model_bytes)
    predictions = {}
    classes = np.asarray(getattr(model, 'classes_', []))
    probability_available = hasattr(model, 'predict_proba') and 2 in classes
    for phase, frame in frames.items():
        model_input = frame.iloc[:, :-1].to_numpy() if spec.get('parent_input') else frame
        if phase == 'future':
            # 所有服务类别都由重载后的模型逐行产生，与当前完整交付契约相同。
            values = np.concatenate([np.asarray(model.predict(model_input[i:i+1])) for i in range(len(frame))])
        else:
            values = np.asarray(model.predict(model_input))
        if values.shape != (len(frame),) or not np.isin(values, [1, 2]).all():
            raise ValueError('分类输出非法')
        predictions[phase] = values
        if probability_available:
            scores = np.asarray(model.predict_proba(model_input))[:, int(np.flatnonzero(classes == 2)[0])]
            if scores.shape != values.shape or not np.isfinite(scores).all():
                raise ValueError('概率输出非法')
            predictions[phase+'_probability'] = scores
        # 检查批量接口没有利用整批未来输入改变当前行的判断。
        if phase == 'future':
            selected = np.unique(np.linspace(0, len(frame)-1, min(32, len(frame)), dtype=int))
            if not np.array_equal(values[selected], model.predict(model_input[selected] if isinstance(model_input, np.ndarray) else model_input.iloc[selected])):
                raise ValueError('逐行与批量输出不一致')
    np.savez_compressed(folder/'predictions.npz', **predictions)
    meta = {'success': True, 'seconds': time.monotonic()-started, 'probability_available': probability_available,
            'weights_sha256': sha256(model_bytes).hexdigest(),
            'predictions_sha256': sha256((folder/'predictions.npz').read_bytes()).hexdigest(),
            'epochs': getattr(model, 'epochs_', None), 'training_loss': getattr(model, 'training_loss_', None)}
    atomic_json(folder/'complete.json', meta)


def specifications(task, bundle, previous):
    items = []
    for variant in VARIANTS:
        for seed in SEEDS:
            items.append({'name': f'{variant}-s{seed}', 'family': 'soft_state', 'variant': variant, 'seed': seed})
    for family in ['logistic', 'forest']:
        for include in [True, False]:
            items.append({'name': family+('-with-parent' if include else '-without-parent'),
                          'family': family, 'include_parent': include})
    for name, path, parent_input in [('source-parent', bundle/'parent.pkl', True)]+[
            (arm+'-published-candidate', Path(json.loads((previous/arm/'decision.json').read_bytes())['candidate']['path']).with_suffix('.pkl'), False)
            for arm in ['project', 'comind']]:
        items.append({'name': name, 'family': 'existing', 'weights': str(path.resolve()),
                      'weights_sha256': sha256(path.read_bytes()).hexdigest(), 'parent_input': parent_input})
    for item in items:
        item['task_input'] = str((bundle/'task').resolve())
    return items


def audit_scores(task, truth, predicted, groups, probability=None):
    scores = metric(task, truth, predicted, groups)
    positive = np.asarray(truth) == 2
    scores.update(positive_rows=int(positive.sum()), predicted_positive=int((predicted == 2).sum()),
                  detected_positive=int((positive & (predicted == 2)).sum()),
                  false_alarms=int((~positive & (predicted == 2)).sum()), prevalence=float(positive.mean()))
    if probability is not None:
        scores['average_precision'] = float(average_precision_score(positive, probability)) if positive.any() else None
        scores['roc_auc'] = float(roc_auc_score(positive, probability)) if 0 < positive.sum() < len(positive) else None
    return scores


def summarize(args, specs):
    """所有固定变体均已完成或登记失败，此时才读取目标答案进行开发性诊断。"""
    results = {}
    for task in TASKS:
        bundle, previous, folder = args.bundle/task, args.previous/task, args.output/task
        oracle = json.loads((bundle/'oracle.json').read_bytes())
        train = pd.read_csv(bundle/'task/train.csv')
        with np.load(bundle/'private-answers.npz', allow_pickle=False) as z:
            service_truth = z['y']
        with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
            future, baseline = z['future'], z['baseline']
        groups = np.asarray(oracle['service_groups'])
        truth = {'train': train.target.to_numpy(), 'explore': np.asarray(oracle['explore_y']),
                 'confirm': np.asarray(oracle['confirm_y']), 'future': service_truth[future]}
        by = {'train': np.asarray(json.loads((bundle/'task/task.json').read_bytes())['train_groups']),
              'explore': np.asarray(oracle['explore_groups']), 'confirm': np.asarray(oracle['confirm_groups']), 'future': groups[future]}
        parent = {p: metric(task, truth[p], pd.read_csv(bundle/'task'/n).parent_value, by[p])
                  for p, n in PHASE_FILES.items()}
        rows = []
        for spec in specs[task]:
            out = folder/spec['name']
            meta = json.loads((out/'complete.json').read_bytes())
            row = {'spec': spec, **meta}
            if not meta['success']:
                rows.append(row)
                continue
            with np.load(out/'predictions.npz', allow_pickle=False) as z:
                predictions = {k: z[k] for k in z.files}
            scores = {phase: audit_scores(task, y, predictions[phase], by[phase], predictions.get(phase+'_probability'))
                      for phase, y in truth.items()}
            adopt = all(accepted(task, parent[p], scores[p]) for p in ['explore', 'confirm'])
            direct = baseline.copy(); direct[future] = predictions['future']
            actual = direct if adopt else baseline.copy()
            row.update(phases=scores, adopted_by_original_rule=adopt,
                       raw_complete=metric(task, service_truth, direct, groups),
                       accepted_complete=metric(task, service_truth, actual, groups),
                       source_parent_label_agreement=float(np.mean(train.target.to_numpy() == train.parent_value.to_numpy())),
                       source_prediction_agreement_with_parent=float(np.mean(predictions['train'] == train.parent_value.to_numpy())),
                       future_disagreement_with_parent=int(np.count_nonzero(predictions['future'] != baseline[future])))
            rows.append(row)
        results[task] = rows
    reproduction = json.loads((args.output/'reproduction.json').read_bytes())
    payload = {'scope': '已见数据上的固定组件消融与诊断，不是新完整控制器对照',
               **reproduction, 'results': results, 'default_adopted': False}
    atomic_json(args.output/'scores.json', payload)
    print(json.dumps({'reproduction_all_equal': True, 'state_max_error': reproduction['state_max_error'],
                      'fits': sum(s['family'] != 'existing' for v in specs.values() for s in v),
                      'failures': sum(not r['success'] for v in results.values() for r in v)}, ensure_ascii=False))


def verify_reproduction(args):
    """先核对默认模型等价，再运行其他变体；这里只比较预测和权重，不读取服务答案。"""
    folder = args.output/'sgp40'
    with np.load(folder/'full-s29/predictions.npz') as a, np.load(folder/'comind-published-candidate/predictions.npz') as b:
        reproduction = {p: bool(np.array_equal(a[p], b[p])) for p in ['train', 'explore', 'confirm', 'future']}
        probability_error = {p: float(np.max(np.abs(a[p+'_probability']-b[p+'_probability'])))
                             for p in ['train', 'explore', 'confirm', 'future'] if p+'_probability' in b}
    new = cloudpickle.loads((folder/'full-s29/model.pkl').read_bytes())
    old = cloudpickle.loads((folder/'comind-published-candidate/model.pkl').read_bytes())
    state_error = max(float(np.max(np.abs(value.detach().numpy()-old.net_.state_dict()[key].detach().numpy())))
                      for key, value in new.net_.state_dict().items())
    original = json.loads((args.previous/'sgp40/comind/decision.json').read_bytes())['candidate']
    with np.load(Path(original['path']).with_suffix('.npz')) as z, np.load(folder/'full-s29/predictions.npz') as a:
        reproduction.update(saved_explore=bool(np.array_equal(a['explore'], z['validation'])),
                            saved_confirm=bool(np.array_equal(a['confirm'], z['confirmation'])))
    with np.load(args.previous/'sgp40/comind/issued.npz') as z, np.load(folder/'full-s29/predictions.npz') as a:
        with np.load(args.bundle/'sgp40/outer-inputs.npz') as inputs:
            reproduction['saved_future'] = bool(np.array_equal(a['future'], z['actual'][inputs['future']]))
    atomic_json(args.output/'reproduction.json', {'reproduction': reproduction,
                'probability_max_error': probability_error, 'state_max_error': state_error})
    if not all(reproduction.values()) or state_error > 1e-7:
        raise ValueError('提取版未复现原模型，保留全部结果并先修复等价性')


def run(args):
    if args.output.exists():
        raise ValueError('消融输出目录已存在，不覆盖有效结果')
    args.output.mkdir(parents=True)
    specs = {}
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for task in TASKS:
        folder, bundle = args.output/task, args.bundle/task
        folder.mkdir()
        identity = json.loads((bundle/'task-identity.json').read_bytes())
        for kind in ['task_files', 'private_files']:
            for name, checksum in identity[kind].items():
                p = (bundle/'task' if kind == 'task_files' else bundle)/name
                if sha256(p.read_bytes()).hexdigest() != checksum:
                    raise ValueError('冻结数据身份改变')
        with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
            columns = json.loads((bundle/'task/task.json').read_bytes())['feature_columns']
            pd.DataFrame(z['service_x'][z['future']], columns=columns).to_csv(folder/'service.csv', index=False)
        specs[task] = specifications(task, bundle, args.previous/task)
        for spec in specs[task]:
            out = folder/spec['name']; out.mkdir()
            atomic_json(out/'spec.json', spec)
    atomic_json(args.output/'frozen-variants.json', specs)
    first = [('sgp40', name) for name in ['full-s29', 'comind-published-candidate']]
    schedule = first+[(task, s['name']) for task in TASKS for s in specs[task] if (task, s['name']) not in first]
    for index, (task, name) in enumerate(schedule):
        spec = next(s for s in specs[task] if s['name'] == name)
        folder = args.output/task/spec['name']
        started = time.monotonic()
        with (folder/'stdout.log').open('w', encoding='utf-8') as stdout, (folder/'stderr.log').open('w', encoding='utf-8') as stderr:
            try:
                proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(folder.resolve())],
                                      cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=90)
                failed = proc.returncode != 0
            except subprocess.TimeoutExpired:
                failed = True
        if failed:
            atomic_json(folder/'complete.json', {'success': False, 'seconds': time.monotonic()-started})
        print(json.dumps({'task': task, 'variant': spec['name'], 'success': not failed}, ensure_ascii=False), flush=True)
        if index == 1:
            verify_reproduction(args)
    atomic_json(args.output/'predictions-sealed.json', {'all_fixed_variants_attempted': True,
                'completed': {task: {s['name']: sha256((args.output/task/s['name']/'complete.json').read_bytes()).hexdigest() for s in specs[task]} for task in TASKS}})
    summarize(args, specs)


def main():
    parser = ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        if any(v is None for v in [args.bundle, args.previous, args.output]):
            parser.error('主进程需要bundle、previous和output')
        run(args)


if __name__ == '__main__':
    main()
