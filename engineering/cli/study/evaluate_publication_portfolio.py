"""冻结已有选案，对父版、确认版和补训版的发布后组合做开发消融。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from engineering.cli.study.audit_publication_effect import cases
from engineering.cli.workflow import london_field_driver as london
from harness.models.publication_portfolio import stream
from harness.models.sparse_reference import choose_from_feedback
from harness.runtime.storage import atomic_json


def run(baseline, audit, output):
    if output.exists():
        raise ValueError('保留原试验，只能写入新目录')
    output.mkdir(parents=True)
    london.configure('2026-07-01')
    results = {}
    for site, arm, work, before, after in cases(baseline):
        destination = output / site / arm
        destination.mkdir(parents=True)
        with np.load(audit / site / arm / 'issued.npz', allow_pickle=False) as z:
            ids = z['ids']
            columns = np.column_stack([z['baseline'], z['accepted_weights'], z['refitted_weights']])
        with np.load(work / 'issued.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            returned = z['returned']
        future = np.arange(len(ids)) >= 28 * 24
        x, ref, clock = columns[future], returned[future], ids[future]
        predicted, weights, state = stream(x, ref, clock, initial=2)
        greedy, choices = choose_from_feedback(x, ref, clock, clock[0], 2)
        # 因果与恢复回归：当前标签改动不能影响当前及此前输出；中段保存不改后续。
        changed = ref.copy()
        pivot = int(np.flatnonzero(np.isfinite(ref))[min(10, np.isfinite(ref).sum()-1)])
        changed[pivot] += 1000.
        perturbed, _, _ = stream(x, changed, clock, 2)
        np.testing.assert_array_equal(predicted[:pivot+1], perturbed[:pivot+1])
        mid = len(clock) // 2
        first, _, saved = stream(x[:mid], ref[:mid], clock[:mid], 2)
        last, _, _ = stream(x[mid:], ref[mid:], clock[mid:], 2, state=json.loads(json.dumps(saved)))
        np.testing.assert_array_equal(predicted, np.r_[first, last])
        output_columns = {'parent': columns[:, 0], 'accepted': columns[:, 1], 'current_refit': columns[:, 2]}
        for name, values in [('equal_mix', x.mean(axis=1)), ('greedy_28d', greedy), ('online_mix', predicted)]:
            output_columns[name] = np.r_[columns[~future, 0], values]
        # 组合与状态已封存后才读取完整真值评分。
        np.savez_compressed(destination / 'issued.npz', ids=ids, weights=weights, choices=choices, **output_columns)
        atomic_json(destination / 'state.json', state)
        truth = london.readings(ROOT / 'data/public/london-air-2026/stations.zip', site).to_numpy()[ids]
        valid = np.isfinite(truth)
        metrics = {name: {'complete_rmse': float(np.sqrt(np.mean((values[valid]-truth[valid])**2))),
                          'after_maintenance_rmse': float(np.sqrt(np.mean((values[valid & future]-truth[valid & future])**2)))}
                   for name, values in output_columns.items()}
        results[site+'/'+arm] = {'metrics': metrics, 'feedback_count': int(np.isfinite(ref).sum()),
                               'strict_causality_verified': True, 'restart_verified': True,
                               'final_weights': weights[-1].tolist()}
    report = {'scope': '已见案例的发布层开发消融；保持原搜索、选案和权重，非新系统独立胜负',
              'new_training': 0, 'new_model_calls': 0, 'results': results}
    atomic_json(output / 'scores.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    for name in ('baseline', 'audit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    run(args.baseline.resolve(), args.audit.resolve(), args.output.resolve())
