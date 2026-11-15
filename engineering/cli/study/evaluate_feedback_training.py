"""在单独允许反馈训练的SECOM协议中逐批更新、发布、回标和恢复。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import process_time
import json
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import cloudpickle
import numpy as np
import pandas as pd
from harness.models.feedback_training import FeedbackTrainingLedger, identity, refit_linear_parent
from harness.models.cumulative_classification_control import CumulativeClassificationControl, confusion, macro_f1
from harness.runtime.storage import atomic_json

ARMS = ['source_only', 'cumulative_refit', 'recent_mix_refit']
PROTOCOL = ROOT / 'engineering/configs/experiments/secom-feedback-training.json'
BUNDLE = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs/secom'
DEST = ROOT / 'results/experiments/contest-continuation/feedback-training-secom'


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('恢复时不能改写已有交付：' + str(path))
    else:
        atomic_json(path, value)


def predict_batch(folder):
    """独立进程仅从发布目录载入权重及无标签输入，按单条请求预测。"""
    with np.load(folder / 'input.npz', allow_pickle=False) as z:
        x, columns = z['x'], z['columns'].tolist()
    models = cloudpickle.loads((folder / 'models.pkl').read_bytes())
    result = {a: [] for a in ARMS}
    for row in x:
        frame = pd.DataFrame([row], columns=columns)
        for arm, model in models.items():
            result[arm].append(int(model.predict(frame)[0]))
    save_json(folder / 'predictions.json', result)


def load_inputs():
    registration = read(BUNDLE / 'task-identity.json')
    for key, base in [('task_files', BUNDLE / 'task'), ('private_files', BUNDLE)]:
        for name, expected in registration[key].items():
            if checksum(base / name) != expected:
                raise ValueError('冻结任务包身份不一致')
    task = read(BUNDLE / 'task/task.json')
    frame = pd.read_csv(BUNDLE / 'task/train.csv')
    source_y = frame.pop('target').to_numpy()
    frame.pop('parent_value')
    with np.load(BUNDLE / 'outer-inputs.npz', allow_pickle=False) as z:
        target, baseline = z['service_x'][:, :-1], z['baseline']
    with np.load(BUNDLE / 'private-answers.npz', allow_pickle=False) as z:
        truth, ids = z['y'], z['ids'].tolist()
    parent = cloudpickle.loads((BUNDLE / 'parent.pkl').read_bytes())
    if not np.array_equal(parent.predict(pd.DataFrame(target, columns=frame.columns)), baseline):
        raise ValueError('原父版未复现已有预测')
    cuts = task['target_batch_cuts']
    if cuts[0] != 0 or cuts[-1] != len(target) or len(set(ids)) != len(ids):
        raise ValueError('目标批次边界或ID无效')
    return task, frame, source_y, target, truth, ids, parent


def run_sequence(out, inputs, *, stop_after=8):
    task, source_x, source_y, target, truth, ids, parent = inputs
    cuts = task['target_batch_cuts']
    features = dict(zip(ids, target, strict=True))
    ledger = FeedbackTrainingLedger(task['train_ids'], allow_feedback_training=True)
    control = CumulativeClassificationControl('pooled_history', len(ARMS), 0, np.zeros((2, 2)), 8)
    fit_calls = 0
    reused = 0
    for i, (start, stop) in enumerate(zip(cuts, cuts[1:])):
        if i >= stop_after:
            break
        folder = out / f'batch-{i+1:02}'
        folder.mkdir(parents=True, exist_ok=True)
        before = {'ledger': ledger.state(), 'control': control.state()}
        completed = folder / 'completed.json'
        if completed.exists():
            record = read(completed)
            if record['before_sha256'] != identity(before):
                raise ValueError('批次恢复状态断链')
            for name, expected in record['files'].items():
                if checksum(folder / name) != expected:
                    raise ValueError('已完成交付发生变化')
            ledger = FeedbackTrainingLedger.restore(record['after']['ledger'])
            control = CumulativeClassificationControl.restore(record['after']['control'])
            reused += 1
            continue
        if (folder / 'issue.json').exists():
            # 对已发放而未完整回标的批次只复用权重和预测，不再进行拟合。
            issue = read(folder / 'issue.json')
            if issue['before_sha256'] != identity(before):
                raise ValueError('待回标批次起点不一致')
            for name, expected in issue['files'].items():
                if checksum(folder / name) != expected:
                    raise ValueError('待回标交付发生变化')
            chosen, _ = control.choose(stop-start)
            if issue['chosen'] != ARMS[chosen] or issue['ids'] != ids[start:stop]:
                raise ValueError('待回标批次选择发生变化')
            outputs = read(folder / 'predictions.json')
        else:
            models, training = {}, {}
            for arm in ARMS:
                view = ledger.training_view(arm)
                if set(view['feedback_ids']) - set(ids[:start]):
                    raise ValueError('当前或未来反馈被用于训练')
                begin = process_time()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    models[arm] = refit_linear_parent(parent, source_x, source_y, features, view)
                fitted = bool(view['feedback_ids'])
                fit_calls += int(fitted)
                training[arm] = {**view, 'fit_called': fitted, 'cpu_seconds': process_time()-begin,
                                 'training_identity': identity(view), 'warnings': [str(w.message) for w in captured]}
            atomic_json(folder / 'training.json', training)
            (folder / 'models.pkl').write_bytes(cloudpickle.dumps(models))
            np.savez_compressed(folder / 'input.npz', x=target[start:stop], columns=np.asarray(source_x.columns, dtype=str))
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--predict-batch', str(folder)]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            outputs = read(folder / 'predictions.json')
            chosen, _ = control.choose(stop-start)
            issue = {'batch': i+1, 'ids': ids[start:stop], 'before_sha256': identity(before),
                     'feedback_rows_available': start, 'chosen': ARMS[chosen],
                     'adaptive_predictions': outputs[ARMS[chosen]],
                     'files': {p: checksum(folder / p) for p in ['training.json', 'models.pkl', 'input.npz', 'predictions.json']}}
            save_json(folder / 'issue.json', issue)
        ledger.issue(issue['ids'], checksum(folder / 'predictions.json'))
        # 此处才由评价器提供本批标签。学习器与独立预测进程均未收到truth全集。
        labels = truth[start:stop]
        ledger.observe(issue['ids'], labels)
        control.observe(labels, np.asarray([outputs[a] for a in ARMS]).T)
        after = {'ledger': ledger.state(), 'control': control.state()}
        save_json(completed, {'before_sha256': identity(before), 'after': after,
                             'files': {p: checksum(folder / p) for p in [*issue['files'], 'issue.json']}})
        # 用序列化后的状态继续下一批，让本次实测覆盖实际恢复路径。
        ledger = FeedbackTrainingLedger.restore(read(completed)['after']['ledger'])
        control = CumulativeClassificationControl.restore(read(completed)['after']['control'])
    return {'new_fits': fit_calls, 'reused_batches': reused, 'completed_batches': stop_after}


def boundary_checks():
    """验证默认权限、错序/提前回标拒绝与未到达标签不影响训练视图。"""
    checks = {}
    blocked = FeedbackTrainingLedger(['source'])
    try:
        blocked.training_view('cumulative_refit')
    except PermissionError:
        checks['score_only_task_rejects_feedback_training'] = True
    live = FeedbackTrainingLedger(['source'], allow_feedback_training=True)
    try:
        live.observe(['future'], [2])
    except ValueError:
        checks['feedback_before_issue_rejected'] = True
    live.issue(['now'], 'a'*64)
    try:
        live.observe(['future'], [2])
    except ValueError:
        checks['wrong_batch_feedback_rejected'] = True
    try:
        live.training_view('cumulative_refit')
    except ValueError:
        checks['pending_delivery_cannot_train'] = True
    live.observe(['now'], [1])
    a = FeedbackTrainingLedger.restore(live.state())
    b = FeedbackTrainingLedger.restore(live.state())
    checks['restored_training_view_equal'] = a.training_view('cumulative_refit') == live.training_view('cumulative_refit')
    a.issue(['future'], 'b'*64)
    b.issue(['future'], 'b'*64)
    a.observe(['future'], [1])
    b.observe(['future'], [2])
    checks['new_feedback_changes_only_subsequent_view'] = a.training_view('cumulative_refit') != b.training_view('cumulative_refit') and live.training_view('cumulative_refit')['feedback_labels'] == [1.]
    if len(checks) != 6 or not all(checks.values()):
        raise ValueError('反馈训练边界核验失败')
    return checks


def evaluate(out, inputs):
    _, _, _, _, truth, _, _ = inputs
    records = [read(out / f'batch-{i+1:02}/issue.json') for i in range(8)]
    outputs = [read(out / f'batch-{i+1:02}/predictions.json') for i in range(8)]
    table = {}
    for arm in [*ARMS, 'pooled_history']:
        parts = [r['adaptive_predictions'] if arm == 'pooled_history' else p[arm] for r, p in zip(records, outputs)]
        prediction = np.concatenate(parts)
        matrix = confusion(truth, prediction)
        tn, fp, fn, tp = matrix.ravel()
        splits = np.split(truth, np.cumsum([len(p) for p in parts])[:-1])
        table[arm] = {'macro_f1': float(macro_f1(matrix)), 'failed_recall': float(tp/(tp+fn)),
                      'detected': int(tp), 'missed': int(fn), 'false_alarms': int(fp),
                      'batches': [float(macro_f1(confusion(y, p))) for y, p in zip(splits, parts)]}
    costs = {a: {'fit_calls': 0, 'cpu_seconds': 0.} for a in ARMS}
    all_warnings = []
    for i in range(8):
        trained = read(out / f'batch-{i+1:02}/training.json')
        for arm, row in trained.items():
            costs[arm]['fit_calls'] += int(row['fit_called'])
            costs[arm]['cpu_seconds'] += row['cpu_seconds']
            all_warnings.extend(row['warnings'])
    costs['pooled_history'] = {'fit_calls': sum(c['fit_calls'] for c in costs.values()),
                              'cpu_seconds': sum(c['cpu_seconds'] for c in costs.values())}
    best = max(ARMS, key=lambda a: table[a]['macro_f1'])
    worthwhile = [a for a in ARMS[1:] if table[a]['macro_f1'] >= table['source_only']['macro_f1']+.01
                  and table[a]['failed_recall'] >= table['source_only']['failed_recall']]
    return {'rows': len(truth), 'arms': table, 'costs': costs, 'decisions': [r['chosen'] for r in records],
            'best_fixed': best, 'worth_expanding': worthwhile,
            'selection_gain_over_best_fixed': table['pooled_history']['macro_f1']-table[best]['macro_f1'],
            'warnings': all_warnings, 'external_model_calls': 0, 'global_default_changed': False}


def main():
    ap = ArgumentParser()
    ap.add_argument('--predict-batch', type=Path)
    ap.add_argument('--output', type=Path, default=DEST)
    args = ap.parse_args()
    if args.predict_batch:
        predict_batch(args.predict_batch)
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    code = ['engineering/src/harness/models/feedback_training.py',
            'engineering/src/harness/models/cumulative_classification_control.py',
            'engineering/cli/study/evaluate_feedback_training.py']
    save_json(out / 'invocation.json', {'protocol_sha256': checksum(PROTOCOL),
              'input_identity_sha256': checksum(BUNDLE / 'task-identity.json'),
              'code': {p: checksum(ROOT / p) for p in code}})
    checks = boundary_checks()
    inputs = load_inputs()
    # 中途完成四批后退出序列执行，再从磁盘恢复；完整重开应全部复用。
    first = run_sequence(out, inputs, stop_after=4)
    second = run_sequence(out, inputs)
    replay = run_sequence(out, inputs)
    if replay != {'new_fits': 0, 'reused_batches': 8, 'completed_batches': 8}:
        raise ValueError('完成后恢复产生了新工作')
    result = evaluate(out, inputs)
    result.update(boundary_checks=checks, recovery={'first': first, 'second': second, 'completed_replay': replay},
                  protocol_sha256=checksum(PROTOCOL))
    atomic_json(out / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
