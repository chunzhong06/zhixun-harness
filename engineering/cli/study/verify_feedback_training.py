"""从封存交付重算评分并独立审计训练标签来历，不重新训练或改变策略。"""
from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from harness.runtime.storage import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(task, folder):
    bundle = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs' / task
    spec = read(bundle / 'task/task.json')
    with np.load(bundle / 'private-answers.npz', allow_pickle=False) as z:
        all_ids, truth = z['ids'].tolist(), z['y']
    answer = dict(zip(all_ids, truth, strict=True))
    available, issued, returned, predictions = {}, [], [], {}
    summary = read(folder / 'summary.json')
    checked_models = 0
    for batch in sorted(folder.glob('batch-*')):
        issue = read(batch / 'issue.json')
        completed = read(batch / 'completed.json')
        for name, expected in completed['files'].items():
            if sha256((batch / name).read_bytes()).hexdigest() != expected:
                raise ValueError('实际交付文件摘要改变')
        training = read(batch / 'training.json')
        for arm, view in training.items():
            if view['source_ids'] != spec['train_ids'] or len(view['feedback_ids']) != len(view['feedback_labels']):
                raise ValueError('源或反馈训练身份不匹配')
            if len(set(view['feedback_ids'])) != len(view['feedback_ids']):
                raise ValueError('反馈训练ID重复')
            if set(view['feedback_ids']) - available.keys():
                raise ValueError('未到达标签进入训练')
            if any(y != available[k] for k, y in zip(view['feedback_ids'], view['feedback_labels'], strict=True)):
                raise ValueError('训练标签与当时返回的真实标签不同')
            expected_ids = [] if arm == 'source_only' else [k for b in (returned[-2:] if arm == 'recent_mix_refit' else returned) for k in b]
            if view['feedback_ids'] != expected_ids:
                raise ValueError('固定更新策略未沿冻结规则选样')
            n_source = len(spec['train_ids'])
            weights = np.asarray(view['sample_weights'])
            if weights.shape != (n_source+len(expected_ids),) or not np.array_equal(weights[:n_source], np.ones(n_source)):
                raise ValueError('源训练权重不一致')
            if expected_ids:
                expected_weight = n_source/len(expected_ids) if arm == 'recent_mix_refit' else 1.
                if not np.allclose(weights[n_source:], expected_weight, atol=1e-12, rtol=0):
                    raise ValueError('反馈训练权重不一致')
            checked_models += 1
        if task == 'secom':
            outputs = read(batch / 'predictions.json')
            if issue['adaptive_predictions'] != outputs[issue['chosen']]:
                raise ValueError('实际自适应交付不对应选中权重')
            feedback_ids = issue['ids']
        else:
            with np.load(batch / 'predictions.npz', allow_pickle=False) as z:
                outputs = {k: z[k] for k in z.files}
            timestamps = pd.DatetimeIndex([i.split(':', 1)[1] for i in issue['ids']])
            feedback_ids = [i for i, t in zip(issue['ids'], timestamps, strict=True) if t.hour % 6 == 5 and t.minute == 50]
            if issue['feedback_ids'] != feedback_ids:
                raise ValueError('风机回标超出预定时钟')
        for arm, values in {**outputs, 'pooled_history': outputs[issue['chosen']]}.items():
            if len(values) != len(issue['ids']):
                raise ValueError('预测与实际交付ID未逐行对应')
            predictions.setdefault(arm, []).extend(values)
        issued.extend(issue['ids'])
        returned.append(feedback_ids)
        available.update({k: answer[k] for k in feedback_ids})
    if issued != all_ids:
        raise ValueError('完整交付缺失、重排或重复')
    scores = {}
    for arm, values in predictions.items():
        value = float(f1_score(truth, values, average='macro', labels=[1, 2], zero_division=0)) if task == 'secom' else float(np.sqrt(np.mean((truth-values)**2)))
        expected = summary['arms'][arm]['macro_f1' if task == 'secom' else 'rmse']
        if abs(value-expected) > 1e-12:
            raise ValueError('独立评分与报告不一致')
        scores[arm] = value
    invocation = read(folder / 'invocation.json')
    for name, expected in invocation['code'].items():
        if sha256((folder / 'frozen-code' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('本次执行的封存源码不对应记录')
    return {'task': task, 'rows': len(issued), 'audited_model_records': checked_models,
            'feedback_received': len(available), 'future_or_unavailable_training_rows': 0,
            'training_labels_match_actual_answers': True, 'source_and_feedback_weights_verified': True,
            'actual_deliveries_and_source_snapshots_verified': True, 'scores': scores}


def main():
    root = ROOT / 'results/experiments/contest-continuation'
    results = [verify('secom', root / 'feedback-training-secom-run415'), verify('wind', root / 'feedback-training-wind')]
    atomic_json(root / 'feedback-training-verification.json', {'passed': True, 'tasks': results, 'new_training_calls': 0})
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
