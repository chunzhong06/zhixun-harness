"""复用轴承选定机制检查其他任务；分别报告固定模型与运行器实际采用。"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.audit_native_run import audit
from engineering.cli.study.evaluate_native_transfer import dataset, input_object
from engineering.cli.study.verify_native_balanced_reference import Broker
from engineering.cli.workflow.native_research import run
from harness.integrations.native_research import export_release, measure
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'native-balanced-cross-task'
NAME = 'width64-e12-balanced1'


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def scores(task, y, predictions, groups):
    owners = np.asarray(groups)
    return {name: {'overall': measure(task, y, p), 'by_group': {
        str(u): measure(task, y[owners == u], p[owners == u]) for u in sorted(set(groups))}}
        for name, p in predictions.items()}


def predict(task, release, x, metadata):
    """每组预测在独立进程中仅加载自己的发布目录，不重新训练。"""
    paths = [json.loads(p) for p in metadata['paths']]
    requests = '\n'.join(json.dumps({'task': metadata['task'], 'inputs': input_object(paths, row)}) for row in x)
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    proc = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'],
        cwd=release, env=env, input=requests, capture_output=True, text=True, encoding='utf-8', timeout=90)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    rows = [json.loads(line)['outputs'] for line in proc.stdout.splitlines()]
    p = np.asarray([[v['score'] for v in row] for row in rows]) if task == 'gas' else np.asarray([r['value'] for r in rows])
    if len(p) != len(x) or not np.isfinite(p).all():
        raise ValueError('独立预测数量或数值不符')
    return p


async def one(task, chosen):
    folder = OUT / task
    counter = {'calls': 0}
    factory = lambda directory, calls: Broker(counter, calls)
    options = {'driver': 'fixture', 'calls': 2, 'stages': 1, 'broker_factory': factory}
    actual = await run(task, folder / 'runtime', **options)
    before = counter['calls']
    restored = await run(task, folder / 'runtime', resume=True, **options)
    if restored != actual or before != counter['calls'] or before != 2:
        raise ValueError('完成恢复改变运行或重复动作')
    check = audit(folder / 'runtime')
    if check['launched_training_workers'] != 1 or actual['driver_completed_stages'] != 1:
        raise ValueError('新参考程序未实际完成一次训练')
    old = json.loads((BASE / 'width-search' / task / 'selection.json').read_bytes())
    archived = next(c for c in old['candidates'] if c['name'] == NAME)
    candidate = Path(actual['stages'][0]['selected']['checkpoint'])
    with np.load(candidate, allow_pickle=False) as a, np.load(archived['checkpoint'], allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('实际通用程序与已登记候选状态不等价')
    spec = chosen['tasks'][task]
    for path, expected in spec['inputs_sha256'].items():
        if checksum(path) != expected:
            raise ValueError('当前交付来源改变')
    models = {'initial': Path(spec['initial']), 'current': Path(spec['selected']), 'extended': candidate}
    atomic_json(folder / 'comparison.json', {'fixed_before_external_scoring': True,
        'models': {k: {'checkpoint': str(v), 'sha256': checksum(v)} for k, v in models.items()}})
    task_dir = folder / 'runtime/task'
    task_info = json.loads((task_dir / 'task.json').read_bytes())
    with np.load(spec['initial'], allow_pickle=False) as checkpoint:
        meta = json.loads(str(checkpoint['metadata']))
    releases = {}
    for name, path in models.items():
        releases[name] = folder / 'models' / name
        export_release(task_dir, path, releases[name])
    # 全量公开验证和外部资料的评价独立于运行器的探索/确认/服务决定。
    with ZipFile(task_info['archive']) as z, TemporaryDirectory(dir=folder, prefix='temporary-') as temp:
        with _trainer(z, Path(temp), task_info['profile']) as trainer:
            raw = [json.loads(row) for row in z.read('public/validation.jsonl').splitlines()]
            from importlib import import_module
            data = import_module(trainer.__package__ + '.data')
            vx = np.asarray([data.flatten(row['inputs'])[1] for row in raw])
    from engineering.cli.study.evaluate_rul_baseline import PACKAGES
    from harness.data.official_gas import read_official_gas
    reader = read_official_gas if task == 'gas' else PACKAGES[task][2]
    package = reader(Path(task_info['archive']))
    labels = np.asarray([r.label for r in package.validation])
    if not np.array_equal(vx, np.asarray([r.values for r in package.validation])):
        raise ValueError('全量验证输入行序变化')
    vp = {name: predict(task, release, vx, meta) for name, release in releases.items()}
    full = scores(task, labels, vp, [str(r.unit_id or r.group) for r in package.validation])
    for name, expected in (('initial', old['baseline']['score']), ('current', spec['official_validation_score']), ('extended', archived['score'])):
        if abs(full[name]['overall']['score'] - expected['score']) > 1e-9:
            raise ValueError('独立全量预测未复现已有评分')
    np.savez_compressed(folder / 'official-issued.npz', ids=np.asarray([r.sample_id for r in package.validation]), **vp)
    x, ids, groups, truth = dataset(task)
    predictions = {name: predict(task, release, x, meta) for name, release in releases.items()}
    np.savez_compressed(folder / 'external-issued.npz', ids=np.asarray(ids), groups=np.asarray(groups), **predictions)
    previous = BASE / 'frozen-transfer' / task
    if checksum(previous / 'issued.npz') != json.loads((previous / 'scores.json').read_bytes())['issued_sha256']:
        raise ValueError('旧发放身份改变')
    with np.load(previous / 'issued.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], ids):
            raise ValueError('外部行序不符')
        for now, then in (('initial', 'initial'), ('current', 'selected')):
            if not np.allclose(saved[then], predictions[now], atol=1e-10, rtol=0):
                raise ValueError('旧外部实际预测未复现')
    result = {'task': task, 'runtime': check, 'accepted': actual['stages'][0]['accepted'],
        'completed_resume_new_calls': 0, 'first_candidate_arrays_identical': True,
        'official_rows': len(labels), 'official': full, 'external_rows': len(ids),
        'external': scores(task, truth(), predictions, groups), 'old_external_predictions_reproduced': True,
        'issued_sha256': {p: checksum(folder / p) for p in ('official-issued.npz', 'external-issued.npz')}}
    atomic_json(folder / 'summary.json', result)
    print(json.dumps({'task': task, 'accepted': result['accepted'], 'runtime_scores': check['scores'],
        'official': {k: v['overall'] for k, v in full.items()},
        'external': result['external'], 'actual_model_calls': 0, 'training_workers': 1}, ensure_ascii=False), flush=True)
    return result


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    chosen = json.loads((BASE / 'bearing-adoption-audit/delivery-selection.json').read_bytes())
    results = [await one(task, chosen) for task in ('gas', 'engine')]
    atomic_json(OUT / 'summary.json', {'scope': '固定通用程序的开发复用与已有外部压力检验；不是新盲测或语言模型决策收益',
        'actual_model_calls': 0, 'training_workers': 2, 'tasks': results})


if __name__ == '__main__':
    asyncio.run(main())
