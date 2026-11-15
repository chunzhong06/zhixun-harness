"""核对冻结发布版的反馈可见退步；未来小时只用于事后诊断信号，不产生选案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.workflow import london_field_driver as scene
from harness.runtime.storage import atomic_json


def metrics(truth, candidate, parent, mask):
    mask = np.asarray(mask) & np.isfinite(truth)
    if not mask.any():
        return {'count': 0, 'candidate_rmse': None, 'parent_rmse': None, 'mse_difference': None,
                'candidate_bias': None, 'parent_bias': None}
    a, b = candidate[mask]-truth[mask], parent[mask]-truth[mask]
    return {'count': int(mask.sum()), 'candidate_rmse': float(np.sqrt(np.mean(a*a))),
            'parent_rmse': float(np.sqrt(np.mean(b*b))), 'mse_difference': float(np.mean(a*a-b*b)),
            'candidate_bias': float(a.mean()), 'parent_bias': float(b.mean())}


def signals(ids, truth, candidate, parent, returned, start):
    """本时刻参考到达后形成信号；下一段真实损失在独立结果列，不能进入信号。"""
    events = np.flatnonzero((ids >= start) & np.isfinite(returned))
    differences = (candidate[events]-returned[events])**2-(parent[events]-returned[events])**2
    rows = []
    for event, at in enumerate(events):
        now = ids[at]
        future = (ids > now) & (ids <= now+6) & np.isfinite(truth)
        if not future.any():
            continue
        future_delta = float(np.mean((candidate[future]-truth[future])**2-(parent[future]-truth[future])**2))
        row = {'hour': int(now), 'future_scored': int(future.sum()), 'future_last_hour': int(ids[future][-1]),
               'following_six_hours_mse_difference': future_delta}
        for days in (1, 7, 28):
            past = (ids[events[:event+1]] > now-days*24)
            visible = events[:event+1][past]
            assert len(visible) and ids[visible].max() <= now and ids[future].min() > now
            row[f'past_{days}d_count'] = len(visible)
            row[f'past_{days}d_mse_difference'] = float(differences[:event+1][past].mean()) if len(visible) >= 2 else None
        rows.append(row)
    return rows


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sealed = json.loads((source/'predictions-complete.json').read_bytes())
    identity = json.loads((source/'raw-predictions-complete.json').read_bytes())
    expected = json.loads((source/'scores.json').read_bytes())
    assert sealed['all_service_predictions_sealed'] and sealed['feedback_continued_without_reset']
    archive = ROOT/'data/public/london-air-2026/stations.zip'
    assert sha256(archive.read_bytes()).hexdigest() == identity['archive_sha256']
    scene.configure('2026-07-01')
    scene.CLOCK = pd.date_range(scene.START-pd.Timedelta(days=180), scene.START+pd.Timedelta(days=62, hours=-1), freq='h')
    results, fingerprints = {}, {}
    for site in ('BX2', 'CD1', 'CE2'):
        truth_all = scene.readings(archive, site).to_numpy()
        for arm in ('project', 'comind'):
            path = source/site/arm/'issued.npz'
            fingerprints[site+'/'+arm] = sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as z:
                ids, candidate, parent, returned = (z[k].copy() for k in ('ids', 'prediction', 'baseline', 'returned'))
                tail = z['tail'].copy()
            assert len(ids) == 62*24 and np.all(np.diff(ids) == 1) and np.isfinite(candidate).all() and np.isfinite(parent).all()
            truth = truth_all[ids]
            reference = np.isfinite(returned)
            np.testing.assert_allclose(returned[reference], truth[reference], rtol=0, atol=0)
            assert np.all(scene.CLOCK[ids[reference]].hour % 6 == 0)
            day = (ids-ids[0])//24
            np.testing.assert_array_equal(tail, day >= 56)
            adopted = day >= 28
            full = metrics(truth, candidate, parent, np.ones(len(ids), dtype=bool))
            last = metrics(truth, candidate, parent, tail)
            for measured, name in [(arm, 'candidate_rmse'), ('parent', 'parent_rmse')]:
                for observed, key in [(full[name], 'complete_62_days_rmse'), (last[name], 'tail_rmse')]:
                    target = expected['rows'][site]['scores'][measured][key]
                    assert observed is None and target is None or observed is not None and target is not None and abs(observed-target) < 1e-10
            periods = {}
            for period, mask in [('original_postpublication_28d', adopted & ~tail), ('tail_6d', tail), ('all_postpublication_34d', adopted)]:
                periods[period] = {name: metrics(truth, candidate, parent, mask & subset)
                                   for name, subset in [('all', np.ones(len(ids), bool)), ('reference', reference), ('nonreference', ~reference)]}
            daily = []
            for number in range(28, 62):
                record = {'day': number, 'date': scene.CLOCK[ids[day == number][0]].date().isoformat()}
                for name, subset in [('all', np.ones(len(ids), bool)), ('reference', reference), ('nonreference', ~reference)]:
                    record.update({name+'_'+k: v for k, v in metrics(truth, candidate, parent, (day == number) & subset).items()})
                daily.append(record)
            lag = {period: {str(k): metrics(truth, candidate, parent, mask & (ids % 6 == k)) for k in range(6)}
                   for period, mask in [('original_postpublication_28d', adopted & ~tail), ('tail_6d', tail)]}
            arrivals = signals(ids, truth, candidate, parent, returned, int(ids[0]+28*24))
            agreement = {}
            for period, minimum, maximum in [('original_postpublication_28d', ids[0]+28*24, ids[0]+56*24), ('tail_6d', ids[0]+56*24, ids[-1]+1)]:
                agreement[period] = {}
                for days in (1, 7, 28):
                    rows = [r for r in arrivals if minimum <= r['hour'] < maximum and r['future_last_hour'] < maximum
                            and r[f'past_{days}d_mse_difference'] is not None]
                    exact = [r for r in rows if r[f'past_{days}d_mse_difference'] != 0 and r['following_six_hours_mse_difference'] != 0]
                    agreement[period][str(days)] = {'signals': len(rows), 'nontied_signals': len(exact),
                        'same_direction_fraction': float(np.mean([np.sign(r[f'past_{days}d_mse_difference']) == np.sign(r['following_six_hours_mse_difference']) for r in exact])) if exact else None,
                        'last_visible_mse_difference': rows[-1][f'past_{days}d_mse_difference'] if rows else None}
            destination = output/site/arm
            destination.mkdir(parents=True)
            pd.DataFrame(daily).to_csv(destination/'daily.csv', index=False)
            atomic_json(destination/'arrival-signals.json', arrivals)
            result = {'periods': periods, 'clock_offset': lag, 'signals': agreement}
            atomic_json(destination/'diagnosis.json', result)
            results[site+'/'+arm] = result
    report = {'scope': '已评分部署轨迹的开发诊断；未生成切换策略或重新选择模型', 'results': results,
              'prediction_sha256': fingerprints, 'source_score_reproduced': True, 'new_training': 0, 'new_model_calls': 0,
              'signals_only_use_arrived_references': True, 'future_losses_only_for_diagnosis': True}
    atomic_json(output/'diagnosis.json', report)
    print(json.dumps({'scope': report['scope'], 'periods': {key: value['periods'] for key, value in results.items()},
                      'new_training': 0, 'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
