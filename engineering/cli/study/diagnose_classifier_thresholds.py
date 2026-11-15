"""固定基模型和阈值网格，仅探索选阈值；独立确认后逐条交付并记录故障功能。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import metric, accepted
from harness.models.probability_threshold_classifier import ProbabilityThresholdClassifier
from harness.runtime.storage import atomic_json

BASES = ['forest-with-parent', 'forest-without-parent', 'project-published-candidate']
THRESHOLDS = [.01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5]


def main():
    parser = ArgumentParser()
    parser.add_argument('--ablation', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有阈值诊断')
    source = args.bundle/'secom'
    identity = json.loads((source/'task-identity.json').read_bytes())
    for key, prefix in [('task_files', source/'task'), ('private_files', source)]:
        for name, checksum in identity[key].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('任务身份改变')
    args.output.mkdir(parents=True)
    oracle = json.loads((source/'oracle.json').read_bytes())
    frames = {p: pd.read_csv(source/'task'/f) for p, f in [('explore', 'validation.csv'), ('confirm', 'test.csv')]}
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline, service_x = z['future'], z['baseline'], z['service_x']
    columns = json.loads((source/'task/task.json').read_bytes())['feature_columns']
    service = pd.DataFrame(service_x[future], columns=columns)
    parent_metrics = {p: metric('secom', oracle[p+'_y'], frame.parent_value, oracle[p+'_groups']) for p, frame in frames.items()}
    decisions = {}
    for name in BASES:
        old = args.ablation/'secom'/name
        meta = json.loads((old/'complete.json').read_bytes())
        if not meta['success'] or sha256((old/'model.pkl').read_bytes()).hexdigest() != meta['weights_sha256']:
            raise ValueError('冻结基模型无效')
        if sha256((old/'predictions.npz').read_bytes()).hexdigest() != meta['predictions_sha256']:
            raise ValueError('原预测变化')
        model = cloudpickle.loads((old/'model.pkl').read_bytes())
        probabilities = {phase: np.asarray(model.predict_proba(frame))[:, 1] for phase, frame in frames.items()}
        with np.load(old/'predictions.npz', allow_pickle=False) as z:
            for phase in frames:
                if not np.allclose(probabilities[phase], z[phase+'_probability'], atol=1e-12, rtol=0):
                    raise ValueError('原概率无法复现')
        candidates = [{'threshold': t, 'explore': metric('secom', oracle['explore_y'],
                        np.where(probabilities['explore'] > t, 2, 1), oracle['explore_groups'])} for t in THRESHOLDS]
        # 一次选定后才看确认；平手优先原0.5，再优先较高阈值，不按服务答案挑选。
        winner = min(candidates, key=lambda c: (-c['explore']['score'], abs(c['threshold']-.5), -c['threshold']))
        out = args.output/name; out.mkdir()
        atomic_json(out/'exploration-selection.json', {'candidates': candidates, 'selected_threshold': winner['threshold'],
                    'base_weights_sha256': meta['weights_sha256']})
        confirmation = metric('secom', oracle['confirm_y'], np.where(probabilities['confirm'] > winner['threshold'], 2, 1), oracle['confirm_groups'])
        adopt = accepted('secom', parent_metrics['explore'], winner['explore']) and accepted('secom', parent_metrics['confirm'], confirmation)
        decisions[name] = {'threshold': winner['threshold'], 'explore': winner['explore'], 'confirm': confirmation,
                            'adopted': adopt, 'base_weights_sha256': meta['weights_sha256']}
        atomic_json(out/'decision.json', decisions[name])
        wrapped = ProbabilityThresholdClassifier(model, winner['threshold'])
        (out/'model.pkl').write_bytes(cloudpickle.dumps(wrapped))
        wrapped = cloudpickle.loads((out/'model.pkl').read_bytes())
        with np.load(old/'predictions.npz', allow_pickle=False) as z:
            if not np.allclose(wrapped.predict_proba(service)[:, 1], z['future_probability'], atol=1e-12, rtol=0):
                raise ValueError('服务输入或包装改变了冻结基模型的概率')
        predictions = np.asarray([wrapped.predict(service.iloc[[i]])[0] for i in range(len(service))])
        if not np.array_equal(predictions, wrapped.predict(service)):
            raise ValueError('逐行与批量预测不一致')
        direct = baseline.copy(); direct[future] = predictions
        actual = direct if adopt else baseline.copy()
        np.savez_compressed(out/'issued.npz', ids=np.asarray(oracle['service_ids']), actual=actual, direct=direct, baseline=baseline)
    atomic_json(args.output/'all-predictions-sealed.json', {name: sha256((args.output/name/'issued.npz').read_bytes()).hexdigest() for name in BASES})
    with np.load(source/'private-answers.npz', allow_pickle=False) as z:
        truth, ids = z['y'], z['ids']
    groups = np.asarray(oracle['service_groups'])
    scores = {}
    for name in BASES:
        out = args.output/name
        with np.load(out/'issued.npz', allow_pickle=False) as z:
            if not np.array_equal(z['ids'], ids) or not np.array_equal(z['actual'][~future], baseline[~future]):
                raise ValueError('交付身份或维护前预测变化')
            row = {'decision': decisions[name], 'weights_sha256': sha256((out/'model.pkl').read_bytes()).hexdigest()}
            for kind in ['actual', 'direct']:
                values = z[kind]
                positive = truth[future] == 2
                row[kind] = {'complete': metric('secom', truth, values, groups),
                             'future': metric('secom', truth[future], values[future], groups[future]),
                             'future_failures': int(positive.sum()),
                             'future_detected': int(np.sum(positive & (values[future] == 2))),
                             'future_false_alarms': int(np.sum(~positive & (values[future] == 2)))}
            scores[name] = row
    atomic_json(args.output/'scores.json', {'scope': '已见SECOM固定组件阈值诊断，0训练和0研究模型调用', 'models': scores})
    print(json.dumps(scores, ensure_ascii=False))


if __name__ == '__main__':
    main()
