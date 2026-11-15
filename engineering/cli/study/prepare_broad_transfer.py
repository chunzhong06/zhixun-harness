"""从四类已有原件构造合法研究包；只训练源父版，服务答案单独封存且不评分。"""
from argparse import ArgumentParser
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
import warnings
from zipfile import ZipFile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from harness.integrations.broad_transfer_task import CLASSIFICATION, verify_core, sources
from harness.runtime.storage import atomic_json


def block(x, y, ids, groups):
    return {'x': np.asarray(x, float), 'y': np.asarray(y, float), 'ids': np.asarray(ids, str), 'groups': np.asarray(groups, str)}


def take(data, mask):
    return {k: v[mask] for k, v in data.items()}


def join(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def sgp40(path):
    with np.load(path, allow_pickle=False) as z:
        threshold = float(np.median(z['A_y']))
        train = block(z['A_x'], 1+(z['A_y'] > threshold), ['A:'+str(i) for i in z['A_ids']], ['A']*700)
        portions = []
        assert np.array_equal(z['B_ids'], z['C_ids'])
        assert not set(z['A_ids']) & set(z['B_ids'])
        for batch in np.array_split(np.argsort(z['B_ids'], kind='stable'), 8):
            portions.append(join([block(z[s+'_x'][batch], 1+(z[s+'_y'][batch] > threshold),
                                 [s+':'+str(i) for i in z[s+'_ids'][batch]], [s]*len(batch)) for s in ['B', 'C']]))
    service = join(portions)
    return train, join(portions[:3]), portions[3], service, np.arange(400) >= 200, [f'signal_{i}' for i in range(48)], {
        'description': 'SGP40 VOC binary classes split by source A median; fit 700 source mixtures. B/C share 200 paired target mixtures. Eight ordered batches: first three exploration, fourth hidden confirmation, final four new-model service. Earlier deliveries stay at parent. All past batch labels arrive after each complete batch; no label refitting.',
        'source_threshold': threshold, 'correlation_unit': 'B/C same mixture paired; 400 rows are 200 mixture units'}


def chronological_cuts(times, fractions):
    cuts = []
    for fraction in fractions:
        n = int(len(times)*fraction)
        while 0 < n < len(times) and times[n] == times[n-1]:
            n += 1
        cuts.append(n)
    if len(set(cuts)) != len(cuts):
        raise ValueError('时间平手造成空批次')
    return cuts


def secom(path):
    with ZipFile(path) as z:
        x = np.loadtxt(StringIO(z.read('secom.data').decode()))
        labels = pd.read_csv(StringIO(z.read('secom_labels.data').decode()), sep=r'\s+', header=None)
    times = pd.to_datetime(labels[1], dayfirst=True).to_numpy()
    order = np.argsort(times, kind='stable'); times = times[order]
    all_rows = block(x[order], np.where(labels[0].to_numpy()[order] == 1, 2, 1),
                     ['row:'+str(i) for i in order], ['source']*len(order))
    cut = chronological_cuts(times, [.4])[0]
    train = take(all_rows, slice(0, cut)); service = take(all_rows, slice(cut, None))
    cuts = [0] + chronological_cuts(times[cut:], np.arange(1, 8)/8) + [len(service['y'])]
    portions = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        part = take(service, slice(a, b)); part['groups'][:] = 'batch'+str(i+1); portions.append(part)
    service = join(portions)
    return train, join(portions[:3]), portions[3], service, np.arange(len(service['y'])) >= cuts[4], [f'measurement_{i}' for i in range(x.shape[1])], {
        'description': 'SECOM chronological quality control: class1=pass, class2=fail. First40% (timestamp ties kept) source fit. Remaining rows in eight contiguous batches: batches1-3 exploration, batch4 hidden confirmation, batches5-8 future service. All quality labels arrive after each complete batch; no label refitting.',
        'source_end': str(times[cut-1]), 'target_start': str(times[cut]), 'target_batch_cuts': cuts}


def wind(path):
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values(['timestamp', 'turbine_id'], kind='stable')
    train_df = df[(df.turbine_id.isin(['Kelmarsh_1', 'Kelmarsh_2', 'Kelmarsh_3'])) & (df.timestamp < '2016-09-01')]
    target = df[(df.turbine_id.isin(['Kelmarsh_4', 'Kelmarsh_5', 'Kelmarsh_6'])) & (df.timestamp >= '2016-09-01')]
    names = ['wind_speed', 'rotor_speed', 'ambient_temp']
    def convert(frame):
        return block(frame[names], frame.power_pu, frame.turbine_id.astype(str)+':'+frame.timestamp.astype(str), frame.turbine_id)
    train, service = convert(train_df), convert(target)
    reference = (target.timestamp.dt.hour % 6 == 5) & (target.timestamp.dt.minute == 50)
    explore = take(service, (reference & (target.timestamp < '2016-09-22')).to_numpy())
    confirm = take(service, (reference & (target.timestamp >= '2016-09-22') & (target.timestamp < '2016-09-29')).to_numpy())
    return train, explore, confirm, service, (target.timestamp >= '2016-09-29').to_numpy(), names, {
        'description': 'Kelmarsh cleaned normal-operation rows, simultaneous power_pu regression, not forecasting. Source turbines1-3 before2016-09-01; target turbines4-6 Sep-Dec. Current measured wind speed, rotor speed and temperature only. Legal six-hour feedback is exact05:50/11:50/17:50/23:50, missing slots not replaced. Sep1-21 feedback exploration, Sep22-28 feedback hidden confirmation; fixed publication Sep29. All earlier target rows stay parent; all later source-file rows delivered. Later feedback is not used by this frozen one-maintenance controller.',
        'train_rows': len(train_df), 'target_rows': len(target), 'feedback_clock': 'exact h%6=5, minute=50',
        'scope': 'author cleaned normal operation; original cleaning used power'}


def flight_rows(file, split):
    """逐完整航班汇总，内存只保留身份列和单航班物理信号；不读虚拟/健康参数。"""
    identity = file['A_'+split][:, :2]
    boundaries = np.r_[0, np.flatnonzero(np.any(identity[1:] != identity[:-1], axis=1))+1, len(identity)]
    rows, targets, ids, groups = [], [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        unit, cycle = identity[start].astype(int)
        measured = np.column_stack([file['X_s_'+split][start:end], file['W_'+split][start:end]])
        label = file['Y_'+split][start:end, 0]
        if not np.all(label == label[0]):
            raise ValueError('航班内目标不一致')
        rows.append(np.r_[measured.mean(0), measured.std(0), cycle])
        targets.append(label[0]); ids.append(f'{unit}:{cycle}'); groups.append(str(unit))
    if len(set(ids)) != len(ids):
        raise ValueError('航班身份出现不连续重现')
    return block(rows, targets, ids, groups)


def ncmapss_ds06(path):
    import h5py
    with h5py.File(path, 'r') as f:
        dev = flight_rows(f, 'dev'); service = flight_rows(f, 'test')
        variables = [v.decode() for key in ['X_s_var', 'W_var'] for v in f[key][:].ravel()]
    assert set(dev['groups']) == set('123456') and set(service['groups']) == set(['7', '8', '9', '10'])
    train = take(dev, np.isin(dev['groups'], ['1', '2', '3', '4']))
    names = [stat+'_'+v for stat in ['mean', 'std'] for v in variables] + ['elapsed_flight']
    return train, take(dev, dev['groups'] == '5'), take(dev, dev['groups'] == '6'), service, np.ones(len(service['y']), bool), names, {
        'description': 'N-CMAPSS DS06 remaining flights at the END of the currently completed flight. Author dev units1-4 source fit; devunit5 past exploration and devunit6 hidden confirmation; author testunits7-10 all flights service, no target feedback. Inputs: mean/std of14 measured sensors+4 operating conditions and elapsed flight index. Exclude virtual sensors, health flags, health parameters, future flights and device IDs from predictor input. Different devices are disjoint; source unit identities only available as train_groups.',
        'measurement_window': 'current complete flight only', 'train_units': ['1', '2', '3', '4'],
        'explore_units': ['5'], 'confirm_units': ['6'], 'service_units': ['7', '8', '9', '10']}


def prepare(task, path, folder):
    train, explore, confirm, service, future, names, metadata = globals()[task](path)
    if not all(len(v['y']) and np.isfinite(v['y']).all() for v in [train, explore, confirm, service]):
        raise ValueError('空数据或非有限目标')
    if set(train['ids']) & (set(explore['ids']) | set(confirm['ids']) | set(service['ids'])):
        raise ValueError('源训练与目标身份重合')
    if set(explore['ids']) & set(confirm['ids']):
        raise ValueError('探索与确认身份重合')
    if (set(explore['ids']) | set(confirm['ids'])) & set(service['ids'][future]):
        raise ValueError('合法反馈与未来服务身份重合')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(np.where(np.isfinite(train['x']), train['x'], np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.)
    def features(part):
        raw = part['x']; present = np.isfinite(raw)
        return np.column_stack([np.where(present, raw, medians), present.astype(float)])
    columns = names + ['present_'+n for n in names]
    frames = [pd.DataFrame(features(p), columns=columns) for p in [train, explore, confirm, service]]
    parent = (make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=17))
              if task in CLASSIFICATION else HistGradientBoostingRegressor(max_iter=96, max_depth=6, max_leaf_nodes=15, l2_regularization=1., random_state=17))
    parent.fit(frames[0], train['y'])
    # 先落盘再重载同一父版；预测裁剪与现有任务runtime一致，对负功率值不修改答案。
    folder.mkdir(parents=True); (folder/'parent.pkl').write_bytes(cloudpickle.dumps(parent))
    parent = cloudpickle.loads((folder/'parent.pkl').read_bytes())
    for frame in frames:
        prediction = np.maximum(0., parent.predict(frame))
        indices = np.linspace(0, len(frame)-1, min(32, len(frame))).astype(int)
        single = np.asarray([max(0., float(parent.predict(frame.iloc[[i]])[0])) for i in indices])
        if not np.allclose(single, prediction[indices], rtol=0, atol=1e-8):
            raise ValueError('父版逐条和批量预测不等价')
        frame['parent_value'] = prediction
    task_folder = folder/'task'; task_folder.mkdir()
    train_frame = frames[0].copy(); train_frame['target'] = train['y']
    train_frame.to_csv(task_folder/'train.csv', index=False)
    frames[1].to_csv(task_folder/'validation.csv', index=False); frames[2].to_csv(task_folder/'test.csv', index=False)
    metadata.update(feature_columns=list(frames[0]), train_groups=train['groups'].tolist(),
                    train_ids=train['ids'].tolist(), source_only_imputation=True, prediction_floor=0.)
    atomic_json(task_folder/'task.json', metadata)
    atomic_json(folder/'oracle.json', {**{name+'_'+key: part[key].tolist() for name, part in [('explore', explore), ('confirm', confirm)] for key in ['y', 'ids', 'groups']},
                                     'service_ids': service['ids'].tolist(), 'service_groups': service['groups'].tolist()})
    np.savez_compressed(folder/'outer-inputs.npz', service_x=frames[3].to_numpy(), baseline=frames[3].parent_value.to_numpy(), future=future)
    np.savez_compressed(folder/'private-answers.npz', y=service['y'], ids=service['ids'])
    np.savez_compressed(folder/'source-preprocessing.npz', medians=medians)
    identity = {'task': task, 'source_file': path.relative_to(ROOT).as_posix(), 'task_files': {p.name: sha256(p.read_bytes()).hexdigest() for p in task_folder.iterdir()},
                'train_rows': len(train['y']), 'explore_rows': len(explore['y']), 'confirm_rows': len(confirm['y']),
                'service_rows': len(service['y']), 'future_rows': int(future.sum()), 'features': len(frames[0].columns),
                'reference_sources': {k: sha256(v.encode()).hexdigest() for k, v in sources(task).items()},
                'private_files': {name: sha256((folder/name).read_bytes()).hexdigest() for name in ['oracle.json', 'outer-inputs.npz', 'private-answers.npz', 'parent.pkl', 'source-preprocessing.npz']},
                'no_service_scores_computed': True, 'parent_one_row_equivalence_checked': True}
    atomic_json(folder/'task-identity.json', identity)
    print(json.dumps({task: {k: v for k, v in identity.items() if k.endswith('rows') or k == 'features'}}, ensure_ascii=False), flush=True)


def main():
    parser = ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    verify_core()
    if args.output.exists():
        raise ValueError('不覆盖已生成任务包')
    plan = json.loads((ROOT/'engineering/configs/experiments/broad-transfer-plan.json').read_bytes())
    for task, spec in plan['cases'].items():
        path = ROOT/spec['file']
        h = sha256()
        with path.open('rb') as fp:
            for chunk in iter(lambda: fp.read(8*1024*1024), b''):
                h.update(chunk)
        if h.hexdigest() != spec['sha256']:
            raise ValueError('原件SHA变化：'+task)
        prepare(task, path, args.output/task)


if __name__ == '__main__':
    main()
