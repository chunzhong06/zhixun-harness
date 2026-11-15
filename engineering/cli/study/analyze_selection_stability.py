"""按设备检验已生成候选的选择稳定性及冻结迁移模型的成对差异。"""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.data.cmapss_input import read_cmapss, test_answers
from harness.runtime.storage import atomic_json


def resample_scores(losses, groups, strata, draws=5000):
    """设备成组抽样、FD子集内分层；保留每台设备的全部窗口和配对预测。"""
    groups, strata = np.asarray(groups), np.asarray(strata)
    if losses.ndim != 2 or len(losses) != len(groups) or not np.isfinite(losses).all():
        raise ValueError('成对损失与设备身份不完整')
    units = np.unique(groups)
    sums = np.asarray([losses[groups == unit].sum(axis=0) for unit in units])
    counts = np.asarray([np.count_nonzero(groups == unit) for unit in units])
    levels = []
    for unit in units:
        values = np.unique(strata[groups == unit])
        assert len(values) == 1
        levels.append(values[0])
    levels = np.asarray(levels)
    rng = np.random.default_rng(20261115)
    sampled_sums = np.zeros((draws, losses.shape[1]))
    sampled_counts = np.zeros(draws)
    for level in np.unique(levels):
        members = np.flatnonzero(levels == level)
        chosen = members[rng.integers(0, len(members), size=(draws, len(members)))]
        sampled_sums += sums[chosen].sum(axis=1)
        sampled_counts += counts[chosen].sum(axis=1)
    return np.sqrt(sampled_sums / sampled_counts[:, None]), sums, counts


def pair_summary(losses, groups, strata):
    sampled, sums, counts = resample_scores(losses, groups, strata)
    point = np.sqrt(losses.mean(axis=0))
    differences = sampled[:, 0]-sampled[:, 1]
    relative = 100*(sampled[:, 0]/sampled[:, 1]-1)
    device_delta = sums[:, 0]/counts-sums[:, 1]/counts
    return {'rows': len(losses), 'devices': len(counts), 'rmse': point.tolist(),
            'rmse_difference': float(point[0]-point[1]),
            'rmse_difference_resampling_interval_95': np.quantile(differences, [.025, .975]).tolist(),
            'relative_error_change_percent': float(100*(point[0]/point[1]-1)),
            'relative_error_resampling_interval_95': np.quantile(relative, [.025, .975]).tolist(),
            'device_mse_difference_quantiles': np.quantile(device_delta, [0, .25, .5, .75, 1]).tolist(),
            'first_arm_better_devices': int(np.count_nonzero(device_delta < 0))}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = json.loads((args.bundle/'oracle.json').read_bytes())
    y = np.asarray(oracle['explore_y'])
    groups = np.asarray(oracle['explore_groups'])
    strata = np.asarray([g.split(':')[0] for g in groups])
    records = json.loads((args.project/'executions/index.json').read_bytes())
    losses, sequences = [], []
    for record in records:
        if record['returncode'] or not record['candidates']:
            continue
        candidate = record['candidates'][-1]
        with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
            loss = (z['validation']-y)**2
        assert abs(float(np.sqrt(loss.mean()))-candidate['score']) < 1e-10
        losses.append(loss)
        sequences.append(record['sequence'])
    losses = np.column_stack(losses)
    sampled, sums, counts = resample_scores(losses, groups, strata)
    chosen = np.asarray(sequences)[sampled.argmin(axis=1)]
    leave_one = (sums.sum(axis=0)[None, :]-sums)/(counts.sum()-counts)[:, None]
    selected_leave_one = np.asarray(sequences)[leave_one.argmin(axis=1)]
    selection = {'exploration_rows': len(y), 'devices': len(counts), 'successful_candidates': len(sequences),
        'observed_best_execution': sequences[int(losses.mean(axis=0).argmin())],
        'bootstrap_selected_counts': dict(Counter(str(i) for i in chosen)),
        'leave_one_device_out_selected_counts': dict(Counter(str(i) for i in selected_leave_one)),
        'execution22_vs_execution5': pair_summary(losses[:, [sequences.index(22), sequences.index(5)]], groups, strata),
        'scope': '条件于已经由这些探索反馈生成的候选池；仅诊断选择稳定性，不给整个自适应搜索的显著性结论'}
    with np.load(args.transfer/'issued.npz', allow_pickle=False) as z:
        ids, subset = z['ids'], z['subsets']
        predicted = np.column_stack((z['project'], z['comind']))
    archive = ROOT/'data/public/cmapss/CMAPSSData.zip'
    answers = {}
    for name in np.unique(subset):
        tracks = read_cmapss(archive, subset=str(name), partition='test')
        answers.update(test_answers(archive, str(name), tracks))
    truth = np.asarray([answers[k] for k in ids])
    target_loss = (predicted-truth[:, None])**2
    transfer = {'project_vs_comind': pair_summary(target_loss, ids, subset),
                'by_subset': {str(name): pair_summary(target_loss[subset == name], ids[subset == name], subset[subset == name])
                              for name in np.unique(subset)},
                'scope': '条件于两组已冻结发布模型和四个固定子集；设备重采样不包括重新训练或再次LLM搜索的不确定性。子集区间为描述性点区间，无多重显著性检验。'}
    report = {'method': '同设备配对、FD子集内分层的5000次设备重采样；保留所有值，不删离群点',
              'selection': selection, 'transfer': transfer, 'new_training': 0, 'new_model_calls': 0,
              'random_seed': 20261115, 'resampling_draws': 5000,
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__}}
    atomic_json(args.output/'analysis.json', report)
    print(json.dumps({'selection': selection, 'transfer_primary': transfer['project_vs_comind']}, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    for name in ('project', 'bundle', 'transfer', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
