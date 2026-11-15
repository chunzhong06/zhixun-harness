"""在已封存的真实部署模型上做发布层消融；全部新预测先封存，再统一评分。"""
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
from harness.integrations.recipe_regressor import code_inputs
from harness.models.publication_control import MODES, SourceSupport, stream
from harness.models.sparse_reference import correct, setting_of
from harness.runtime.storage import atomic_json


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((source/'raw-predictions-complete.json').read_bytes())
    assert json.loads((source/'predictions-complete.json').read_bytes())['all_service_predictions_sealed']
    archive = ROOT/'data/public/london-air-2026/stations.zip'
    assert sha256(archive.read_bytes()).hexdigest() == identity['archive_sha256']
    scene.configure('2026-07-01')
    scene.CLOCK = pd.date_range(scene.START-pd.Timedelta(days=180), scene.START+pd.Timedelta(days=62, hours=-1), freq='h')
    features, names = code_inputs(scene.prepare(archive, True, True))
    issued, identities = {}, {}
    for site in ('BX2', 'CD1', 'CE2'):
        for arm in ('project', 'comind'):
            folder = source/site/arm
            with np.load(folder/'issued.npz', allow_pickle=False) as z:
                ids, current, parent, returned = (z[k].copy() for k in ('ids', 'prediction', 'baseline', 'returned'))
            assert len(ids) == 1488 and np.all(np.diff(ids) == 1)
            at = int(ids[0]+28*24)
            future = ids >= at
            frozen = identity['frozen'][site+'/'+arm]
            if frozen['adopted']:
                with np.load(folder/'raw-predictions.npz', allow_pickle=False) as z:
                    history = z['candidate_clock'][z['candidate_clock'] < at]
                    raw = z['candidate_raw'][:len(history)]
                assert ids[0] <= history[0] <= history[-1] < at
                known = returned[history-ids[0]]
                calibrated = correct(raw[:, None], known, history, setting_of(frozen['feedback']))[:, 0]
                original_inputs = pd.read_csv(folder/'candidate-service.csv')
                assert list(original_inputs.columns) == names
                np.testing.assert_allclose(features[ids[future]], original_inputs.to_numpy(), rtol=0, atol=1e-8)
            else:
                history = np.arange(at-21*24-6, at)
                calibrated = parent[history-ids[0]].copy()
            support = SourceSupport(features[history, :4], calibrated, parent[history-ids[0]])
            destination = output/site/arm;destination.mkdir(parents=True)
            arrays = {'ids': ids, 'returned': returned}
            counters = {}
            for mode in MODES:
                values, weights, limited, state = stream(ids[future], np.column_stack((parent[future], current[future])),
                    returned[future], features[ids[future], :4], mode=mode, support=support)
                actual = current.copy();actual[future] = values
                np.testing.assert_array_equal(actual[~future], current[~future])
                if mode == 'current':np.testing.assert_array_equal(actual, current)
                if mode == 'parent':np.testing.assert_array_equal(actual, parent)
                arrays[mode] = actual;arrays[mode+'_weights'] = weights;arrays[mode+'_limited'] = limited
                counters[mode] = {'limited_hours': int(limited.sum()), 'mean_candidate_weight': float(weights[:, 1].mean()),
                                  'tail_mean_candidate_weight': float(weights[-144:, 1].mean()),
                                  'updates': state['selector']['updates'] if state['selector'] else 0,
                                  'loss_scale_restarts': state['selector']['restarts'] if state['selector'] else 0}
                atomic_json(destination/(mode+'-state.json'), state)
            np.savez_compressed(destination/'issued.npz', **arrays)
            atomic_json(destination/'control.json', {'support': support.state(), 'calibration_first_hour': int(history[0]),
                'calibration_last_hour': int(history[-1]), 'publication_hour': at, 'features': names[:4], 'modes': counters})
            issued[site, arm] = (ids, {mode: arrays[mode] for mode in MODES}, counters)
            identities[site+'/'+arm] = {'original_issued_sha256': sha256((folder/'issued.npz').read_bytes()).hexdigest(),
                'original_raw_sha256': sha256((folder/'raw-predictions.npz').read_bytes()).hexdigest(), 'original_model': frozen,
                'current_prediction_exactly_reproduced': True, 'parent_prediction_exactly_reproduced': True}
    atomic_json(output/'predictions-complete.json', {'all_modes_sealed': True, 'identities': identities,
        'calibration_strictly_before_publication': True, 'target_nonreference_answers_not_used': True})
    results, pooled = {}, {}
    for site in ('BX2', 'CD1', 'CE2'):
        truth_all = scene.readings(archive, site).to_numpy()
        for arm in ('project', 'comind'):
            ids, modes, counters = issued[site, arm]
            truth = truth_all[ids];day = (ids-ids[0])//24
            masks = {'original_56d': day < 56, 'complete_62d': np.ones(len(ids), bool),
                     'tail_6d': day >= 56, 'postpublication_34d': day >= 28}
            metrics = {}
            for mode, values in modes.items():
                metrics[mode] = {}
                for period, scope in masks.items():
                    valid = scope & np.isfinite(truth)
                    residual = values[valid]-truth[valid]
                    metrics[mode][period] = {'count': int(valid.sum()),
                        'rmse': float(np.sqrt(np.mean(residual**2))) if len(residual) else None}
                    pooled.setdefault(arm, {}).setdefault(mode, {}).setdefault(period, []).extend(residual.tolist())
            results[site+'/'+arm] = {'metrics': metrics, 'control': counters}
    totals = {arm: {mode: {period: float(np.sqrt(np.mean(np.square(values)))) if values else None
                           for period, values in periods.items()} for mode, periods in modes.items()}
              for arm, modes in pooled.items()}
    report = {'scope': '已见资料上的冻结发布模型控制消融，不是新独立完整搜索胜负', 'results': results,
              'pooled_rmse': totals, 'new_model_calls': 0, 'new_base_model_training': 0,
              'all_predictions_sealed_before_scoring': True}
    atomic_json(output/'scores.json', report)
    print(json.dumps({'pooled_rmse': totals, 'new_model_calls': 0, 'new_base_model_training': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
