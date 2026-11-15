"""独立核对真实研究的合法训练、确认、持续发布和完整成绩，不新增训练。"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from harness.runtime.storage import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    run = ROOT / 'results/experiments/contest-continuation/feedback-research-secom'
    bundle = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs/secom'
    spec = read(bundle / 'task/task.json')
    with np.load(bundle / 'outer-inputs.npz', allow_pickle=False) as z:
        x, baseline = z['service_x'], z['baseline']
    with np.load(bundle / 'private-answers.npz', allow_pickle=False) as z:
        ids, truth = z['ids'].tolist(), z['y']
    answers = dict(zip(ids, truth, strict=True))
    cuts = spec['target_batch_cuts']
    search_ids = read(run / 'search-task/feedback-identities.json')['ids']
    if search_ids != ids[:cuts[2]]:
        raise ValueError('探索和确认反馈提前用于研究训练')
    decision, summary = read(run / 'decision.json'), read(run / 'summary.json')
    candidate = decision['candidate']
    with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
        scores = [float(f1_score(truth[a:b], z[key], labels=[1, 2], average='macro', zero_division=0))
                  for a, b, key in [(cuts[2], cuts[3], 'validation'), (cuts[3], cuts[4], 'confirmation')]]
    if scores != decision['candidate_scores'] or not all(a > b for a, b in zip(scores, decision['parent_scores'])):
        raise ValueError('实际采用未通过冻结探索和确认规则')
    complete, deltas = [], []
    for i, (start, stop) in enumerate(zip(cuts, cuts[1:]), 1):
        folder = run / f'batch-{i:02}'
        issued = read(folder / 'issued.json')
        if issued['ids'] != ids[start:stop]:
            raise ValueError('交付ID缺失或重排')
        if i <= 4:
            actual = baseline[start:stop]
        else:
            meta = read(folder / 'input/feedback-training.json')
            for name, expected in meta['files'].items():
                if sha256((folder / 'input' / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('实际训练资料被改动')
            feedback_ids = read(folder / 'input/feedback-identities.json')['ids']
            labels = pd.read_csv(folder / 'input/feedback.csv', float_precision='round_trip')['target'].to_numpy()
            if feedback_ids != ids[:start] or not np.array_equal(labels, [answers[k] for k in feedback_ids]):
                raise ValueError('发布重训读取了当前/未来或不对应的标签')
            if (folder / 'program.py').read_bytes() != Path(candidate['path']).parents[1].joinpath('candidate.py').read_bytes():
                raise ValueError('发布重训程序与研究实际选中程序不同')
            path = folder / 'artifacts' / Path(candidate['path']).name
            record = read(path)
            if record['label'] != candidate['label'] or record['feedback'] != candidate['feedback']:
                raise ValueError('持续发布更换了候选身份')
            weights = path.with_suffix('.pkl')
            if sha256(weights.read_bytes()).hexdigest() != issued['model']:
                raise ValueError('实际交付模型身份不对应')
            model = cloudpickle.loads(weights.read_bytes())
            actual = model.predict(pd.DataFrame(x[start:stop], columns=spec['feature_columns']))
        delta = float(np.max(np.abs(actual-issued['prediction'])))
        if delta > 1e-12:
            raise ValueError('独立重载预测不对应实际发放')
        deltas.append(delta)
        complete.extend(issued['prediction'])
        feedback = read(folder / 'ledger.json')['feedback'][-1]
        if feedback['prediction_sha256'] != sha256((folder / 'issued.json').read_bytes()).hexdigest():
            raise ValueError('标签回标没有绑定实际交付')
    with np.load(run / 'issued.npz', allow_pickle=False) as z:
        if z['ids'].tolist() != ids or not np.array_equal(z['harness'], complete):
            raise ValueError('完整评分文件与逐批交付不一致')
        for arm in summary['arms']:
            for part, start in [('complete', 0), ('future', cuts[4])]:
                score = f1_score(truth[start:], z[arm][start:], labels=[1, 2], average='macro', zero_division=0)
                if abs(score-summary['arms'][arm][part]['macro_f1']) > 1e-12:
                    raise ValueError('完整或后续分数不符')
        equal = bool(np.array_equal(z['harness'], z['fixed_same_start_cumulative_refit']))
    calls = sum((run / 'queue/responses/mle-model' / p.name).is_file() for p in (run / 'queue/requests/mle-model').glob('*.json'))
    if calls != summary['model_calls']:
        raise ValueError('真实响应数不符')
    result = {'passed': True, 'real_model_responses': calls, 'source_and_feedback_roles_verified': True,
              'selection_confirmation_verified': True, 'publication_program_and_candidate_identity_verified': True,
              'independent_prediction_max_difference': max(deltas), 'rows': len(ids),
              'exactly_equal_to_same_start_cumulative_refit': equal, 'new_training_calls': 0}
    atomic_json(run / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
