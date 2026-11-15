"""延续伦敦原实际发布版：先封存基础预测，再按原时序回标，最后评分新尾段。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.study.audit_publication_effect import replay, selected_models
from engineering.cli.workflow import london_field_driver as scene
from harness.integrations.recipe_regressor import code_inputs
from harness.models.sparse_reference import correct, setting_of
from harness.models.stream_plan import StreamPlanBank
from harness.runtime.storage import atomic_json

SITES = ('BX2', 'CD1', 'CE2')


def bindings(comparison):
    rows = {r['task']: r for r in json.loads(comparison.read_bytes())['rows'] if r['task'] in SITES}
    assert set(rows) == set(SITES)
    for site in SITES:
        for arm in ('project', 'comind'):
            published = rows[site][arm]
            work = ROOT/published['path']
            decision = json.loads((work/'decision.json').read_bytes())
            assert decision['adopted'] == published['adopted']
            candidate = None
            if decision['adopted']:
                _, before, after = selected_models(work, arm)
                candidate = before if arm == 'comind' else after
                if arm == 'comind':
                    assert candidate['weights_sha256'] == published['weights_sha256']
            yield site, arm, work, published, candidate


def input_files(comparison):
    """冻结前仅收集已发布证据和模型身份，不读取新增时段目标值。"""
    paths = {comparison}
    for _, arm, work, published, candidate in bindings(comparison):
        paths.update(work/name for name in ('decision.json', 'issued.npz', 'initial/plan.json', 'initial/training.json'))
        training = json.loads((work/'initial/training.json').read_bytes())
        for component in training['components']:
            paths.update(Path(component['folder'])/'fit'/name for name in ('source.py', 'model.joblib'))
        if candidate:
            paths.update(work/name for name in ('oracle.npz', 'executions/index.json', 'serving/input/model.pkl',
                                                'serving/input/service.csv', 'task-identity.json'))
            records = json.loads((work/'executions/index.json').read_bytes())
            # selected_models核对原选案与补训绑定；两套身份均属于这条冻结证明链。
            for record in records:
                if record['sequence'] == json.loads((work/'decision.json').read_bytes())['selected_execution'] or record['phase'] == 'publication':
                    for item in record['candidates']:
                        path = Path(item['path'])
                        paths.update([path, path.with_suffix('.pkl'), path.with_suffix('.npz')])
            if arm == 'comind':
                paths.add(work/'author-workspace/author-selection.json')
        if published.get('prediction_evidence'):
            paths.add(ROOT/published['prediction_evidence'])
    return sorted(paths)


def sparse_reference(archive, site):
    """只构造六小时回标序列；非回标小时的目标数值留到最终评分阶段。"""
    with ZipFile(archive) as z:
        raw = json.loads(z.read(site+'.json'))['RawAQData']
    assert raw['@SiteCode'] == site and raw['@SpeciesCode'] == 'PM25'
    rows = raw.get('Data') or []
    rows = [rows] if isinstance(rows, dict) else rows
    clock = pd.DatetimeIndex(pd.to_datetime([r['@MeasurementDateGMT'] for r in rows]))
    assert clock.is_unique and clock.is_monotonic_increasing
    selected = np.flatnonzero(clock.hour % 6 == 0)
    values = pd.to_numeric(pd.Series([rows[i].get('@Value') for i in selected]), errors='coerce').to_numpy()
    result = pd.Series(values, index=clock[selected]).reindex(scene.CLOCK).to_numpy()
    assert not np.isfinite(result[scene.CLOCK.hour % 6 != 0]).any()
    return result


def feedback(raw, clock, labels, setting, tail_start):
    reference = labels[clock]
    values = correct(raw[:, None], reference, clock, setting)[:, 0]
    arrivals = np.flatnonzero((clock >= tail_start) & np.isfinite(reference))
    probes = arrivals[np.linspace(0, len(arrivals)-1, min(3, len(arrivals))).astype(int)] if len(arrivals) else []
    for at in probes:
        changed = reference.copy()
        changed[at] += 123.
        altered = correct(raw[:, None], changed, clock, setting)[:, 0]
        np.testing.assert_array_equal(altered[:at+1], values[:at+1])
    return values, len(probes)


def run(args):
    comparison, archive, output = (p.resolve() for p in (args.comparison, args.archive, args.output))
    output.mkdir(parents=True, exist_ok=False)
    scene.configure('2026-07-01')
    old_clock = scene.CLOCK.copy()
    old_features, old_names = code_inputs(scene.prepare(archive, True, True))
    scene.CLOCK = pd.date_range(scene.START-pd.Timedelta(days=180), scene.START+pd.Timedelta(days=62, hours=-1), freq='h')
    features, names = code_inputs(scene.prepare(archive, True, True))
    assert names == old_names
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(scene.CLOCK[:len(old_clock)], old_clock)
    np.testing.assert_array_equal(features[:len(old_features)], old_features)
    start = int(np.flatnonzero(scene.CLOCK >= scene.START)[0])
    service = np.arange(start, len(scene.CLOCK))
    tail_start = start+56*24
    assert len(service) == 62*24
    initial_history = np.arange(start-21*24-6, start)
    initial_clock = np.r_[initial_history, service]
    states, frozen = {}, {}
    for site, arm, work, published, candidate in bindings(comparison):
        folder = output/site/arm
        folder.mkdir(parents=True)
        plan = json.loads((work/'initial/plan.json').read_bytes())
        training = json.loads((work/'initial/training.json').read_bytes())
        bank = StreamPlanBank(folder/'parent-components', features, names)
        columns, identities = [], []
        for i, component in enumerate(training['components']):
            origin = Path(component['folder'])/'fit'
            target = folder/'parent-components'/str(i)/'fit'
            target.mkdir(parents=True)
            for name in ('source.py', 'model.joblib'):
                shutil.copy2(origin/name, target/name)
            assert sha256((target/'source.py').read_text(encoding='utf-8').encode()).hexdigest() == component['source_sha256']
            copied = {**component, 'folder': str(target.parent)}
            past = bank.base(copied, initial_history, np.zeros(len(initial_history)), strict=True)
            future = bank.base(copied, service, np.maximum(0., features[service, 4]))
            columns.append(np.r_[past, future])
            identities.append({'id': component['id'], 'weights_sha256': component['weights_sha256'],
                               'source_sha256': component['source_sha256']})
        state = {'work': work, 'published': published, 'plan': plan,
                 'parent_raw': np.column_stack(columns), 'candidate': candidate}
        arrays = {'parent_clock': initial_clock, 'parent_raw': state['parent_raw']}
        if candidate:
            identity = json.loads((work/'task-identity.json').read_bytes())
            assert sha256(archive.read_bytes()).hexdigest() == identity['archive_sha256']
            with np.load(work/'oracle.npz', allow_pickle=False) as z:
                history = np.r_[z['validation_hours'], z['confirmation_hours']]
            at = start+28*24
            assert history[-1]+1 == at
            future = service[service >= at]
            previous_inputs = pd.read_csv(work/'serving/input/service.csv')
            assert list(previous_inputs.columns) == names
            np.testing.assert_allclose(features[future[:len(previous_inputs)]], previous_inputs.to_numpy(), rtol=0, atol=1e-8)
            inputs = folder/'candidate-service.csv'
            pd.DataFrame(features[future], columns=names).to_csv(inputs, index=False)
            path = Path(candidate['path'])
            predicted = replay(path.with_suffix('.pkl'), inputs, folder/'candidate-serving', timeout=240)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                past = np.r_[z['validation'], z['confirmation']]
            state.update(candidate_clock=np.r_[history, future], candidate_raw=np.r_[past, predicted],
                         candidate_history=len(history), at=at)
            arrays.update(candidate_clock=state['candidate_clock'], candidate_raw=state['candidate_raw'])
        np.savez_compressed(folder/'raw-predictions.npz', **arrays)
        states[site, arm] = state
        frozen[site+'/'+arm] = {'original_run': str(work), 'adopted': candidate is not None, 'parent_components': identities,
            'candidate_weights_sha256': candidate['weights_sha256'] if candidate else None,
            'feedback': candidate['feedback'] if candidate else None, 'publication_rule': published['publication_rule']}
    atomic_json(output/'raw-predictions-complete.json', {'all_base_predictions_sealed': True, 'new_training': 0,
        'new_model_calls': 0, 'old_feature_prefix_reproduced': True, 'frozen': frozen,
        'archive_sha256': sha256(archive.read_bytes()).hexdigest(),
        'comparison_sha256': sha256(comparison.read_bytes()).hexdigest()})
    issued, checks = {}, {}
    for site in SITES:
        labels = sparse_reference(archive, site)
        for arm in ('project', 'comind'):
            state = states[site, arm]
            baseline = np.zeros(len(service))
            count = 0
            for i, component in enumerate(state['plan']['components']):
                values, probes = feedback(state['parent_raw'][:, i], initial_clock, labels,
                                          setting_of(component['feedback']), tail_start)
                baseline += component['weight']*values[len(initial_history):]
                count += probes
            with np.load(state['work']/'issued.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], service[:56*24])
                np.testing.assert_allclose(baseline[:56*24], z['baseline'], rtol=0, atol=1e-8)
                expected = z[arm].copy()
            actual = baseline.copy()
            if state['candidate']:
                values, probes = feedback(state['candidate_raw'], state['candidate_clock'], labels,
                                          setting_of(state['candidate']['feedback']), tail_start)
                actual[service >= state['at']] = values[state['candidate_history']:]
                count += probes
            published = state['published']
            if published.get('prediction_evidence'):
                path = ROOT/published['prediction_evidence']
                assert sha256(path.read_bytes()).hexdigest() == published['prediction_sha256']
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], service[:56*24])
                    expected = z['accepted_weights']
            np.testing.assert_allclose(actual[:56*24], expected, rtol=0, atol=1e-8)
            issued[site, arm] = actual
            issued[site, arm+'_parent'] = baseline
            checks[site+'/'+arm] = {'old_1344_predictions_reproduced': True, 'future_feedback_prefix_checks': count}
            np.savez_compressed(output/site/arm/'issued.npz', ids=service, prediction=actual, baseline=baseline,
                                returned=labels[service], tail=service >= tail_start)
        np.testing.assert_allclose(issued[site, 'project_parent'], issued[site, 'comind_parent'], rtol=0, atol=1e-8)
    atomic_json(output/'predictions-complete.json', {'all_service_predictions_sealed': True, 'checks': checks,
        'tail_hours_per_site': 144, 'full_hours_per_site': 1488, 'timezone': 'GMT/UTC',
        'target_nonfeedback_answers_not_used': True, 'feedback_continued_without_reset': True})
    rows, totals = {}, {arm: [] for arm in ('project', 'comind', 'parent')}
    for site in SITES:
        # 到这里才读取非回标小时的目标值；此后不再生成或修改任何预测。
        truth = scene.readings(archive, site).to_numpy()[service]
        valid, tail = np.isfinite(truth), service >= tail_start
        result = {'tail_issued': int(tail.sum()), 'tail_scored': int((valid & tail).sum()), 'scores': {}}
        for arm in ('project', 'comind', 'parent'):
            values = issued[site, arm if arm != 'parent' else 'project_parent']
            residual = values[valid & tail]-truth[valid & tail]
            result['scores'][arm] = {'tail_rmse': float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
                                     'complete_62_days_rmse': float(np.sqrt(np.mean((values[valid]-truth[valid])**2))) if valid.any() else None}
            totals[arm].extend(residual.tolist())
        rows[site] = result
    pooled = {arm: float(np.sqrt(np.mean(np.square(values)))) if values else None for arm, values in totals.items()}
    macro = {arm: float(np.mean([r['scores'][arm]['tail_rmse'] for r in rows.values() if r['tail_scored']]))
             if any(r['tail_scored'] for r in rows.values()) else None for arm in totals}
    report = {'scope': '既有站点未评分六日尾段的冻结部署续服；同地区同站，不是新增独立设备',
              'period': ['2026-08-26T00:00:00', '2026-08-31T23:00:00'], 'rows': rows,
              'pooled_tail_rmse': pooled, 'macro_site_tail_rmse': macro, 'new_training': 0, 'new_model_calls': 0,
              'original_prefix_reproduced': True, 'all_predictions_sealed_before_final_scoring': True}
    atomic_json(output/'scores.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
