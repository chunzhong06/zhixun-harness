"""作者TabICLv2的CPU实际发布试验；源训练标签与外层评价答案分开。"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from argparse import ArgumentParser
from hashlib import sha256
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'engineering/.environments/tabicl-runtime'), str(ROOT), str(ROOT/'engineering/src')]
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from harness.runtime.storage import atomic_json
from harness.integrations.official_research_task import metric
from engineering.cli.workflow.official_components import accepted


def worker(folder):
    """工作进程只读源标签及无标签输入；作者权重固定，禁止自动下载。"""
    import torch
    from tabicl import TabICLRegressor
    torch.set_num_threads(1)
    spec = json.loads((folder/'spec.json').read_bytes())
    checkpoint = Path(spec['checkpoint'])
    if sha256(checkpoint.read_bytes()).hexdigest() != spec['checkpoint_sha256']:
        raise ValueError('作者预训练权重改变')
    train = pd.read_csv(folder/'train.csv'); y = train.pop('target').to_numpy()
    model = TabICLRegressor(n_estimators=8, batch_size=1, kv_cache=True, device='cpu',
        n_jobs=1, model_path=checkpoint, allow_auto_download=False, random_state=42, use_amp=False)
    started = time.monotonic(); model.fit(train, y); fit_seconds = time.monotonic()-started
    print(json.dumps({'fit_seconds': fit_seconds, 'source_rows': len(y)}, ensure_ascii=False), flush=True)
    model.save(folder/'model.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=True)
    del model; gc.collect()
    model = TabICLRegressor.load(folder/'model.pkl', device='cpu')
    predictions = {}; serving_started = time.monotonic()
    for phase in ['explore', 'confirm', 'future']:
        frame = pd.read_csv(folder/(phase+'.csv'))
        values = np.asarray([float(model.predict(frame.iloc[[i]])[0]) for i in range(len(frame))])
        if not np.isfinite(values).all():
            raise ValueError('作者模型产生非有限输出')
        predictions[phase] = np.maximum(0., values)
        if phase == 'explore':
            # 同一无标签探索输入的批量/单行和调用顺序一致性，不读取服务答案。
            grouped = np.maximum(0., model.predict(frame))
            repeated = max(0., float(model.predict(frame.iloc[[0]])[0]))
            if not np.allclose(grouped, predictions[phase], rtol=1e-5, atol=1e-4) or not np.isclose(repeated, predictions[phase][0], rtol=1e-5, atol=1e-4):
                raise ValueError('缓存或批量接口改变单行结果')
    np.savez_compressed(folder/'predictions.npz', **predictions)
    result = {'success': True, 'fit_seconds': fit_seconds, 'inference_seconds': time.monotonic()-serving_started,
              'weights_sha256': sha256((folder/'model.pkl').read_bytes()).hexdigest(),
              'checkpoint_sha256': spec['checkpoint_sha256'], 'state_bytes': (folder/'model.pkl').stat().st_size,
              'checkpoint_bytes': checkpoint.stat().st_size,
              'predictions_sha256': sha256((folder/'predictions.npz').read_bytes()).hexdigest(),
              'single_row_and_batch_consistent': True, 'source_rows': len(y), 'features': train.shape[1],
              'n_estimators': 8, 'cpu_threads': 1}
    atomic_json(folder/'complete.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run(args):
    source, out = args.bundle, args.output
    out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((source/'task-identity.json').read_bytes())
    for name, checksum in identity['task_files'].items():
        if sha256((source/'task'/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('原官方任务输入改变')
    train = pd.read_csv(source/'task/train.csv'); y = train.pop('target').to_numpy()
    meta = json.loads((source/'task/task.json').read_bytes()); groups = np.asarray(meta['train_groups'])
    columns = list(train.columns[:-1])
    frames = {'explore': pd.read_csv(source/'task/validation.csv'), 'confirm': pd.read_csv(source/'task/test.csv')}
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline = z['future'], z['baseline']
        frames['future'] = pd.DataFrame(z['service_x'][future], columns=meta['feature_columns'])
    folder = out/'tabicl'; folder.mkdir()
    fitting = train[columns].copy(); fitting['target'] = y; fitting.to_csv(folder/'train.csv', index=False)
    for phase, frame in frames.items():frame[columns].to_csv(folder/(phase+'.csv'), index=False)
    checkpoint = ROOT/'references/models/tabicl-v2/tabicl-regressor-v2-20260212.ckpt'
    manifest = json.loads((checkpoint.parent/'source-manifest.json').read_bytes())
    expected = next(r['lfs']['oid'] for r in manifest['files'] if r['path'] == checkpoint.name)
    atomic_json(folder/'spec.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': expected,
                'author_default_ensemble': 8, 'batch_size_for_memory': 1, 'kv_cache': True,
                'full_source_rows': len(y), 'target_feedback_used_for_training': False})
    started = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, '-B', __file__, '--worker', str(folder.resolve())],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=900)
        (folder/'stdout.log').write_text(proc.stdout, encoding='utf-8')
        (folder/'stderr.log').write_text(proc.stderr, encoding='utf-8')
        if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
        foundation = json.loads((folder/'complete.json').read_bytes())
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        foundation = {'success': False, 'reason': str(exc), 'elapsed_seconds': time.monotonic()-started}
        atomic_json(folder/'failure.json', foundation)
    # 同一原始输入上的源训练树及设备等权中位数；没有根据新模型输出调参。
    model = HistGradientBoostingRegressor(max_iter=96, max_depth=6, max_leaf_nodes=15, l2_regularization=1., random_state=17)
    started = time.monotonic();model.fit(train[columns], y); tree_seconds = time.monotonic()-started
    (out/'hist-gradient.pkl').write_bytes(cloudpickle.dumps(model));model = cloudpickle.loads((out/'hist-gradient.pkl').read_bytes())
    row_weights = np.asarray([1./np.sum(groups == g) for g in groups]); order = np.argsort(y, kind='stable')
    value = float(y[order][np.searchsorted(np.cumsum(row_weights[order]), row_weights.sum()/2)])
    predictions = {'hist_gradient': {}, 'balanced_source_median': {}}
    for phase, frame in frames.items():
        predictions['hist_gradient'][phase] = np.maximum(0., [float(model.predict(frame.iloc[[i]][columns])[0]) for i in range(len(frame))])
        predictions['balanced_source_median'][phase] = np.full(len(frame), value)
    atomic_json(out/'source-median.json', {'value': value, 'device_weights': '每台源设备总权重相同'})
    if foundation['success']:
        with np.load(folder/'predictions.npz', allow_pickle=False) as z: predictions['tabicl'] = {phase:z[phase] for phase in frames}
    oracle = json.loads((source/'oracle.json').read_bytes())
    parent = {phase:metric('bearing', oracle[phase+'_y'], frames[phase].iloc[:, -1], oracle[phase+'_groups']) for phase in ['explore','confirm']}
    decisions = {}
    for name, pp in predictions.items():
        measured = {phase:metric('bearing', oracle[phase+'_y'], pp[phase], oracle[phase+'_groups']) for phase in ['explore','confirm']}
        decisions[name] = {**measured, 'adopted': all(accepted('bearing', parent[p], measured[p]) for p in ['explore','confirm'])}
    selected = min(decisions, key=lambda k: decisions[k]['explore']['score'])
    atomic_json(out/'decisions-sealed.json', {'parent': parent, 'candidates': decisions, 'selected_by_explore': selected})
    issued = {}
    for name, pp in predictions.items():
        direct = baseline.copy();direct[future] = pp['future']
        issued[name+'/direct'] = direct
        issued[name+'/actual'] = direct if decisions[name]['adopted'] else baseline.copy()
    np.savez_compressed(out/'issued.npz', ids=np.asarray(oracle['service_ids']), **issued)
    # 所有预测已封存后才由官方原包按ID取得服务答案。
    from harness.data.official_bearing import read_official_bearing
    archive = Path(identity['archive'])
    if sha256(archive.read_bytes()).hexdigest() != identity['archive_sha256']:
        raise ValueError('官方包改变')
    package = read_official_bearing(archive);answers = {r.sample_id:r.label for r in package.validation}
    truth = np.asarray([answers[k] for k in oracle['service_ids']]); service_groups = np.asarray(oracle['service_groups'])
    for name, values in issued.items():
        if not np.array_equal(values[~future], baseline[~future]):raise ValueError('前五轮交付改变')
    scores = {name:{'complete':metric('bearing', truth, values, service_groups),
                    'future':metric('bearing', truth[future], values[future], service_groups[future])} for name, values in issued.items()}
    result = {'foundation': foundation, 'decisions': decisions, 'selected_by_explore': selected,
              'scores':scores,'baseline':metric('bearing',truth,baseline,service_groups), 'tree_fit_seconds':tree_seconds,
              'new_research_model_calls':0,'issued_sha256':sha256((out/'issued.npz').read_bytes()).hexdigest(),
              'scope':'ICML2026作者预测器的默认8成员CPU组件试验，完整交付包含固定前五轮；不是新完整研究驱动对照'}
    atomic_json(out/'scores.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    worker(args.worker) if args.worker else run(args)
