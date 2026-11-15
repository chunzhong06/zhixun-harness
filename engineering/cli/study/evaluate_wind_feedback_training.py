"""复用回标训练账本，在稀疏风机标签上比较固定重训和历史择优。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import process_time
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from harness.models.feedback_training import FeedbackTrainingLedger, identity
from harness.runtime.storage import atomic_json

ARMS = ['source_only', 'cumulative_refit', 'recent_mix_refit']
BUNDLE = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs/wind'
PROTOCOL = ROOT / 'engineering/configs/experiments/wind-feedback-training.json'
DEST = ROOT / 'results/experiments/contest-continuation/feedback-training-wind'


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def predict_batch(folder):
    with np.load(folder / 'input.npz', allow_pickle=False) as z:
        frame = pd.DataFrame(z['x'], columns=z['columns'])
    models = cloudpickle.loads((folder / 'models.pkl').read_bytes())
    predictions = {}
    for arm, model in models.items():
        value = np.maximum(0., np.concatenate([model.predict(frame.iloc[i:i+128]) for i in range(0, len(frame), 128)]))
        for index in [0, len(frame)-1]:
            if abs(float(np.maximum(0., model.predict(frame.iloc[[index]]))[0])-value[index]) > 1e-12:
                raise ValueError('批量和单条预测不一致')
        if not np.isfinite(value).all():
            raise ValueError('独立预测含非法数值')
        predictions[arm] = value
    np.savez_compressed(folder / 'predictions.npz', **predictions)


def load_inputs():
    registration = read(BUNDLE / 'task-identity.json')
    for key, base in [('task_files', BUNDLE / 'task'), ('private_files', BUNDLE)]:
        for name, expected in registration[key].items():
            if checksum(base / name) != expected:
                raise ValueError('原任务数据身份改变')
    task = read(BUNDLE / 'task/task.json')
    source = pd.read_csv(BUNDLE / 'task/train.csv')
    y = source.pop('target').to_numpy()
    source.pop('parent_value')
    with np.load(BUNDLE / 'outer-inputs.npz', allow_pickle=False) as z:
        target, baseline = z['service_x'][:, :-1], z['baseline']
    with np.load(BUNDLE / 'private-answers.npz', allow_pickle=False) as z:
        truth, ids = z['y'], z['ids'].tolist()
    parent = cloudpickle.loads((BUNDLE / 'parent.pkl').read_bytes())
    if not np.allclose(np.maximum(0., parent.predict(pd.DataFrame(target, columns=source.columns))), baseline, atol=1e-12, rtol=0):
        raise ValueError('实际父版不对应原始发布')
    timestamps = pd.DatetimeIndex([i.split(':', 1)[1] for i in ids])
    if not timestamps.is_monotonic_increasing:
        raise ValueError('服务时钟未排序')
    blocks = np.asarray((timestamps-pd.Timestamp('2016-09-01')).days // 14)
    cuts = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(ids)]
    eligible = (timestamps.hour % 6 == 5) & (timestamps.minute == 50)
    if len(cuts) != 10:
        raise ValueError('预定九个14天批次与输入不符')
    return task, source, y, target, truth, ids, parent, cuts, eligible


def sequence(out, data, stop_after=9):
    task, source, y, target, truth, ids, parent, cuts, eligible = data
    index = {i: n for n, i in enumerate(ids)}
    ledger = FeedbackTrainingLedger(task['train_ids'], allow_feedback_training=True)
    history = np.zeros(len(ARMS))
    fits, reused = 0, 0
    for i, (start, stop) in enumerate(zip(cuts, cuts[1:])):
        if i >= stop_after:
            break
        folder = out / f'batch-{i+1:02}'
        folder.mkdir(parents=True, exist_ok=True)
        before = {'ledger': ledger.state(), 'squared_errors': history.tolist()}
        receipt = folder / 'completed.json'
        if receipt.exists():
            result = read(receipt)
            if result['before_sha256'] != identity(before):
                raise ValueError('恢复状态断链')
            for name, digest in result['files'].items():
                if checksum(folder / name) != digest:
                    raise ValueError('已完成交付改变')
            ledger = FeedbackTrainingLedger.restore(result['after']['ledger'])
            history = np.asarray(result['after']['squared_errors'])
            reused += 1
            continue
        feedback_rows = np.flatnonzero(eligible[start:stop]) + start
        feedback_ids = [ids[j] for j in feedback_rows]
        chosen = int(np.flatnonzero(history <= history.min()+1e-12)[0])
        if (folder / 'issue.json').exists():
            issue = read(folder / 'issue.json')
            if issue['before_sha256'] != identity(before) or issue['chosen'] != ARMS[chosen] or issue['feedback_ids'] != feedback_ids:
                raise ValueError('待回标交付身份不一致')
            for name, digest in issue['files'].items():
                if checksum(folder / name) != digest:
                    raise ValueError('待回标交付改变')
        else:
            models, training = {}, {}
            for arm in ARMS:
                view = ledger.training_view(arm)
                rows = np.array([index[k] for k in view['feedback_ids']], dtype=int)
                if len(rows) and (np.any(rows >= start) or not np.all(eligible[rows])):
                    raise ValueError('当前、未来或无回标权限的样本进入训练')
                begin = process_time()
                model = parent
                if len(rows):
                    features = pd.concat((source, pd.DataFrame(target[rows], columns=source.columns)), ignore_index=True)
                    model = clone(parent)
                    model.fit(features, np.r_[y, view['feedback_labels']], sample_weight=view['sample_weights'])
                    fits += 1
                models[arm] = model
                training[arm] = {**view, 'training_identity': identity(view), 'fit_called': bool(len(rows)),
                                 'cpu_seconds': process_time()-begin, 'model_iterations': int(model.n_iter_)}
            atomic_json(folder / 'training.json', training)
            (folder / 'models.pkl').write_bytes(cloudpickle.dumps(models))
            np.savez_compressed(folder / 'input.npz', x=target[start:stop], columns=np.asarray(source.columns, dtype=str))
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--predict-batch', str(folder)],
                                    cwd=folder, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            issue = {'batch': i+1, 'ids': ids[start:stop], 'feedback_ids': feedback_ids, 'chosen': ARMS[chosen],
                     'before_sha256': identity(before), 'files': {name: checksum(folder / name) for name in
                        ['training.json', 'models.pkl', 'input.npz', 'predictions.npz']}}
            atomic_json(folder / 'issue.json', issue)
        ledger.issue(ids[start:stop], checksum(folder / 'predictions.npz'), feedback_ids=feedback_ids)
        # 评价器只将预定时钟的标签送入训练/选择；其他行答案留到完整服务后评分。
        labels = truth[feedback_rows]
        ledger.observe(feedback_ids, labels)
        with np.load(folder / 'predictions.npz', allow_pickle=False) as predictions:
            history += [np.sum((predictions[a][feedback_rows-start]-labels)**2) for a in ARMS]
        after = {'ledger': ledger.state(), 'squared_errors': history.tolist()}
        atomic_json(receipt, {'before_sha256': identity(before), 'after': after,
                             'files': {name: checksum(folder / name) for name in [*issue['files'], 'issue.json']}})
        ledger = FeedbackTrainingLedger.restore(read(receipt)['after']['ledger'])
        history = np.asarray(read(receipt)['after']['squared_errors'])
    return {'new_fits': fits, 'reused_batches': reused}


def evaluate(out, data):
    _, _, _, _, truth, ids, _, cuts, eligible = data
    groups = np.array([i.split(':', 1)[0] for i in ids])
    all_predictions = {a: [] for a in [*ARMS, 'pooled_history']}
    costs = {a: {'fit_calls': 0, 'cpu_seconds': 0.} for a in ARMS}
    decisions = []
    for i in range(len(cuts)-1):
        folder = out / f'batch-{i+1:02}'
        issue = read(folder / 'issue.json')
        decisions.append(issue['chosen'])
        with np.load(folder / 'predictions.npz', allow_pickle=False) as predictions:
            for a in all_predictions:
                all_predictions[a].append(predictions[issue['chosen'] if a == 'pooled_history' else a])
        for a, training in read(folder / 'training.json').items():
            costs[a]['fit_calls'] += int(training['fit_called'])
            costs[a]['cpu_seconds'] += training['cpu_seconds']
    costs['pooled_history'] = {'fit_calls': sum(c['fit_calls'] for c in costs.values()),
                              'cpu_seconds': sum(c['cpu_seconds'] for c in costs.values())}
    def rmse(y, pred):
        return float(np.sqrt(np.mean((y-pred)**2)))
    table = {}
    for a, parts in all_predictions.items():
        values = np.concatenate(parts)
        table[a] = {'rmse': rmse(truth, values),
                    'groups': {g: rmse(truth[groups == g], values[groups == g]) for g in sorted(set(groups))},
                    'batches': [rmse(truth[s:e], values[s:e]) for s, e in zip(cuts, cuts[1:])]}
    best = min(ARMS, key=lambda a: table[a]['rmse'])
    return {'rows': len(truth), 'available_feedback_rows': int(eligible.sum()),
            'training_feedback_rows_before_final_batch': int(eligible[:cuts[-2]].sum()),
            'arms': table, 'costs': costs, 'decisions': decisions, 'best_fixed': best,
            'adaptive_rmse_minus_best_fixed': table['pooled_history']['rmse']-table[best]['rmse'],
            'external_model_calls': 0, 'global_default_changed': False}


def main():
    ap = ArgumentParser()
    ap.add_argument('--predict-batch', type=Path)
    args = ap.parse_args()
    if args.predict_batch:
        predict_batch(args.predict_batch)
        return
    DEST.mkdir(parents=True, exist_ok=True)
    code = ['engineering/cli/study/evaluate_wind_feedback_training.py', 'engineering/src/harness/models/feedback_training.py',
            'engineering/src/harness/runtime/storage.py', 'engineering/configs/experiments/wind-feedback-training.json']
    invocation = {'code': {p: checksum(ROOT / p) for p in code}, 'input_identity_sha256': checksum(BUNDLE / 'task-identity.json')}
    if (DEST / 'invocation.json').exists() and read(DEST / 'invocation.json') != invocation:
        raise ValueError('执行源码或输入变更，需要新实验目录')
    atomic_json(DEST / 'invocation.json', invocation)
    for p in code:
        copy = DEST / 'frozen-code' / p
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, copy)
    # 稀疏回标的资格在标签到达前固定，未选中的已服务行也不能混入训练。
    check = FeedbackTrainingLedger(['source'], allow_feedback_training=True)
    check.issue(['a', 'b'], 'a'*64, feedback_ids=['a'])
    try:
        check.observe(['a', 'b'], [1., 2.])
    except ValueError:
        pass
    else:
        raise ValueError('未授权反馈未被拒绝')
    check.observe(['a'], [1.])
    if FeedbackTrainingLedger.restore(check.state()).training_view('cumulative_refit')['feedback_ids'] != ['a']:
        raise ValueError('恢复引入未回标行')
    data = load_inputs()
    first = sequence(DEST, data, 4)
    second = sequence(DEST, data)
    replay = sequence(DEST, data)
    if replay != {'new_fits': 0, 'reused_batches': 9}:
        raise ValueError('完整恢复重复执行')
    summary = evaluate(DEST, data)
    summary.update(recovery={'first': first, 'second': second, 'completed_replay': replay},
                   sparse_feedback_boundary_verified=True, protocol_sha256=checksum(PROTOCOL))
    atomic_json(DEST / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
