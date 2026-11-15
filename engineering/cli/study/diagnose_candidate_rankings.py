"""只用已有探索证据检查候选提名及排序敏感性，不打开服务答案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import metric
from harness.models.cumulative_classification_control import confusion, macro_f1
from harness.runtime.storage import atomic_json


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for task in ['sgp40', 'secom', 'wind', 'ncmapss_ds06']:
        source = args.base/'broad-transfer-inputs'/task
        oracle = json.loads((source/'oracle.json').read_bytes())
        y, groups = np.asarray(oracle['explore_y']), np.asarray(oracle['explore_groups'])
        parent = pd.read_csv(source/'task/validation.csv').iloc[:, -1].to_numpy()
        classification = task in ['sgp40', 'secom']
        previous = args.base/('shared-classifier-full-pair' if task == 'sgp40' else 'broad-transfer-full-pair')/task
        for arm in ['project', 'comind']:
            records = json.loads((previous/arm/'executions/index.json').read_bytes())
            decision = json.loads((previous/arm/'decision.json').read_bytes())
            candidates = [{'id': 'parent', 'label': '源父版', 'prediction': parent}]
            for execution in records:
                if execution['returncode']:
                    continue
                for candidate in execution['candidates']:
                    path = Path(candidate['path'])
                    if sha256(path.with_suffix('.pkl').read_bytes()).hexdigest() != candidate['weights_sha256']:
                        raise ValueError('候选实际权重改变')
                    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
                        prediction = archive['validation']
                    score = metric(task, y, prediction, groups)['score']
                    if abs((1-score if classification else score)-candidate['score']) > 1e-10:
                        raise ValueError('原探索评分不能复算')
                    candidates.append({'id': f"{execution['sequence']:04d}/{path.stem}",
                                       'label': candidate['label'], 'prediction': prediction,
                                       'weights_sha256': candidate['weights_sha256']})
            matrix = np.column_stack([c['prediction'] for c in candidates])
            names = [c['id'] for c in candidates]
            measured = [metric(task, y, c['prediction'], groups) for c in candidates]
            original = np.asarray([v['score'] for v in measured])
            group_mean = np.asarray([v['group_mean'] for v in measured])
            worst = np.asarray([v['worst'] for v in measured])
            if classification:
                prefix = confusion(y, parent)
                complete_proxy = np.asarray([macro_f1(prefix+confusion(y, c['prediction'])) for c in candidates])
            else:
                complete_proxy = np.sqrt((((matrix-y[:, None])**2).mean(axis=0)+np.mean((parent-y)**2))/2)
            best = np.argmax if classification else np.argmin
            chosen = {name: names[int(best(values))] for name, values in [('pooled', original), ('group_mean', group_mean), ('worst_group', worst), ('same_explore_complete_proxy', complete_proxy)]}
            actual = f"{decision['selected_execution']:04d}/{Path(decision['candidate']['path']).stem}" if decision['candidate'] else None
            sensitivity = {}
            if len(np.unique(groups)) > 1:
                for g in np.unique(groups):
                    keep = groups != g
                    values = [metric(task, y[keep], c['prediction'][keep], groups[keep])['score'] for c in candidates]
                    sensitivity['leave_group_'+str(g)] = names[int(best(values))]
            if classification:
                for k, rows in enumerate(np.array_split(np.arange(len(y)), 3), 1):
                    values = [metric(task, y[rows], c['prediction'][rows], groups[rows])['score'] for c in candidates]
                    sensitivity['explore_order_third_'+str(k)] = names[int(best(values))]
            serialized = [{**{k:v for k,v in c.items() if k != 'prediction'}, 'metrics': measured[i],
                           'complete_proxy': float(complete_proxy[i])} for i,c in enumerate(candidates)]
            original_best = float(original[int(best(original))])
            actual_score = None if actual not in names else float(original[names.index(actual)])
            result = {'task': task, 'arm': arm, 'rows': len(y), 'groups': len(np.unique(groups)),
                      'candidate_count_with_parent': len(candidates), 'original_nomination': actual,
                      'original_nomination_explore': actual_score, 'pooled_best_explore': original_best,
                      'nomination_equals_best_score': actual_score is not None and abs(actual_score-original_best) <= 1e-10,
                      'rankings': chosen, 'selection_sensitivity': sensitivity, 'candidates': serialized,
                      'uses_service_answers': False}
            results[task+'/'+arm] = result
            print(json.dumps({k:v for k,v in result.items() if k != 'candidates'}, ensure_ascii=False), flush=True)
    atomic_json(args.output/'diagnosis.json', results)


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--base', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
