"""真实发布层执行：各方只使用自己的发布版及同样公共工具，逐行推理后按批回标。"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.models.binary_publication_bank import BinaryPublicationBank, prepare_bank, FAMILIES
from harness.models.cumulative_classification_control import CumulativeClassificationControl, confusion
from harness.runtime.storage import atomic_json
from engineering.cli.study.replay_classification_feedback import batches_for, score

MODES = ('fixed', 'pooled_history', 'agile_batch')


def run_case(task, args):
    source = args.bundle/task
    previous = args.base/('shared-classifier-full-pair' if task == 'sgp40' else 'broad-transfer-full-pair')/task
    identity = json.loads((source/'task-identity.json').read_bytes())
    for key, prefix in [('task_files', source/'task'), ('private_files', source)]:
        for name, checksum in identity[key].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('冻结任务资料变化')
    meta = json.loads((source/'task/task.json').read_bytes())
    oracle = json.loads((source/'oracle.json').read_bytes())
    train = pd.read_csv(source/'task/train.csv'); target = train.pop('target').to_numpy()
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline, features = z['future'], z['baseline'], z['service_x']
    cuts = batches_for(task, meta, len(baseline), future)
    output = args.output/task; output.mkdir()
    # 与原实际服务使用相同CSV交换格式；预测模型只获得当前行，标签由外层释放。
    pd.DataFrame(features, columns=meta['feature_columns']).to_csv(output/'service.csv', index=False)
    frame = pd.read_csv(output/'service.csv')
    originals, banks, trajectories, choices, controls = {}, {}, {}, {}, {}
    for arm in ['project', 'comind']:
        old = previous/arm
        decision = json.loads((old/'decision.json').read_bytes())
        delivered = json.loads((old/'predictions-complete.json').read_bytes())
        if sha256((old/'issued.npz').read_bytes()).hexdigest() != delivered['issued_sha256']:
            raise ValueError('原交付身份改变')
        with np.load(old/'issued.npz', allow_pickle=False) as z:
            if not np.array_equal(z['ids'], oracle['service_ids']):
                raise ValueError('原交付ID顺序改变')
            originals[arm] = z['actual'].copy()
        weights = Path(decision['candidate']['path']).with_suffix('.pkl') if delivered['adopted'] else source/'parent.pkl'
        if delivered['adopted'] and sha256(weights.read_bytes()).hexdigest() != decision['candidate']['weights_sha256']:
            raise ValueError('原实际发布权重变化')
        bank_folder = output/arm/'models'
        prepared = prepare_bank(bank_folder, train, target, source/'parent.pkl', weights, published_is_parent=not delivered['adopted'])
        atomic_json(output/arm/'input-identity.json', {'own_original_issued_sha256': delivered['issued_sha256'],
                    'own_published_weights_sha256': sha256(weights.read_bytes()).hexdigest(),
                    'source_task_identity_sha256': sha256((source/'task-identity.json').read_bytes()).hexdigest(),
                    'new_reference_fits': prepared['new_fits'], 'uses_opponent_model': False})
        for mode in MODES:
            key = arm+'/'+mode
            banks[key] = BinaryPublicationBank(bank_folder)
            trajectories[key] = baseline.copy()
            choices[key] = []
    # 预测模型不接收这些标签；前4批到期标签初始化控制器，后续只在整批交付后释放。
    with np.load(source/'private-answers.npz', allow_pickle=False) as z:
        y, ids = z['y'], z['ids']
    prefix = confusion(y[:cuts[4]], baseline[:cuts[4]])
    for key, bank in banks.items():
        controls[key] = CumulativeClassificationControl(key.split('/')[1], len(bank.identity['arm_names']), 0, prefix, 4)
    for batch, (start, end) in enumerate(zip(cuts[4:-1], cuts[5:]), start=5):
        shadows = {}
        for key, control in controls.items():
            if int(control.actual.sum()) != start:
                raise ValueError('选择器接触了不属于过去的标签')
            selected, weights = control.choose(end-start)
            bank = banks[key]
            matrix, raw_scores = [], []
            for row in range(start, end):
                predictions, probabilities = bank.predict_row(frame.iloc[[row]])
                matrix.append(predictions)
                raw_scores.append([probabilities[name] for name in FAMILIES])
            matrix = np.asarray(matrix)
            shadows[key] = matrix
            actual = matrix[:, selected]
            trajectories[key][start:end] = actual
            chosen = bank.identity['arm_names'][selected]; choices[key].append(chosen)
            stem = output/(f'batch-{batch}-'+key.replace('/', '-'))
            np.savez_compressed(stem.with_suffix('.npz'), ids=ids[start:end], actual=actual, shadows=matrix,
                                model_scores=np.asarray(raw_scores))
            atomic_json(stem.with_suffix('.json'), {'feedback_rows_before': start, 'prediction_rows': [start, end],
                'selected': chosen, 'selected_definition': bank.identity['definitions'][selected],
                'weights': weights, 'source_models': bank.identity['models'], 'model_score_kinds': bank.score_kind,
                'issued_sha256': sha256(stem.with_suffix('.npz').read_bytes()).hexdigest()})
        # 两组和全部模式都完成当前批真实预测，才共同进入该批的反馈阶段。
        atomic_json(output/f'batch-{batch}-sealed.json', {'all_modes_issued': True, 'last_feedback_row_before': start})
        for key, control in controls.items():
            observation = control.observe(y[start:end], shadows[key])
            state = control.state()
            atomic_json(output/(f'batch-{batch}-'+key.replace('/', '-')+'-state.json'), {'observation': observation, 'state': state})
            controls[key] = CumulativeClassificationControl.restore(state)
    for arm in ['project', 'comind']:
        if not np.array_equal(trajectories[arm+'/fixed'], originals[arm]):
            raise ValueError('真实逐行推理不能复现原固定交付')
    for values in trajectories.values():
        if not np.array_equal(values[~future], baseline[~future]):
            raise ValueError('过去交付被改写')
    np.savez_compressed(output/'issued.npz', ids=ids, **trajectories)
    groups = np.asarray(oracle['service_groups'])
    metrics = {key: {**score(task, y, values, groups, future), 'choices': choices[key], 'updates': controls[key].updates}
               for key, values in trajectories.items()}
    result = {'task': task, 'scope': '冻结真实研究决定后的发布层完整实跑；不是新的全程LLM搜索',
              'policies': metrics, 'new_reference_fits': 4, 'new_research_model_calls': 0,
              'uses_opponent_model': False, 'original_fixed_predictions_exact': True,
              'actual_models_called_one_row_at_a_time': True,
              'issued_sha256': sha256((output/'issued.npz').read_bytes()).hexdigest()}
    atomic_json(output/'scores.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError('不覆盖已有真实发布试验')
    args.output.mkdir(parents=True)
    results = {task: run_case(task, args) for task in ['sgp40', 'secom']}
    atomic_json(args.output/'scores.json', results)


if __name__ == '__main__':
    main()
