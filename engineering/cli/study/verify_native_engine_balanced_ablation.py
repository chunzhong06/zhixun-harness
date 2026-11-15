"""固定既有四份权重解释组件作用，再检查同一程序的连续实际发放。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.audit_native_run import audit
from engineering.cli.study.evaluate_native_transfer import dataset
from engineering.cli.study.verify_native_balanced_cross_task import predict, scores
from engineering.cli.study.verify_native_balanced_reference import Broker
from engineering.cli.workflow.native_research import run
from harness.integrations.native_research import export_release
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'native-engine-balanced-ablation'
ARMS = {'plain': ('full-validation-search', 'plain-lr0.001-e12-all'),
        'balanced': ('full-validation-search', 'balanced-e12-all'),
        'expanded': ('width-search', 'width64-e12-balanced0'),
        'expanded_balanced': ('width-search', 'width64-e12-balanced1')}


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def existing():
    from engineering.cli.study.evaluate_rul_baseline import PACKAGES
    filename, expected, reader = PACKAGES['engine']
    archive = ROOT / 'data/official/engine' / filename
    if checksum(archive) != expected:
        raise ValueError('官方原包变化')
    package = reader(archive)
    vx = np.asarray([r.values for r in package.validation]); y = np.asarray([r.label for r in package.validation])
    paths = {}; predictions = {}; identities = {}
    task_dir = BASE / 'full-validation-search/engine/inputs'
    with np.load(task_dir / 'initial.npz', allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
    with ZipFile(archive) as z, TemporaryDirectory(dir=OUT, prefix='temporary-') as temp:
        with _trainer(z, Path(temp), 'C') as trainer:
            paths['initial'] = task_dir / 'initial.npz'
            for name, (scope, candidate) in ARMS.items():
                report = json.loads((BASE / scope / 'engine/selection.json').read_bytes())
                item = next(c for c in report['candidates'] if c['name'] == candidate)
                paths[name] = Path(item['checkpoint'])
            for name, path in paths.items():
                model = trainer.Model.load(path)
                predictions[name] = model.predict(vx)
                if name != 'initial':
                    receipt = json.loads((path.parent / 'training.json').read_bytes())
                    if receipt['checkpoint_sha256'] != checksum(path) or receipt['parent_checkpoint_sha256'] != checksum(paths['initial']):
                        raise ValueError('四组候选未从相同官方父版接续')
                    with np.load(path.parent / 'predictions.npz', allow_pickle=False) as saved:
                        if not np.allclose(saved['validation'], predictions[name], atol=1e-10, rtol=0):
                            raise ValueError('候选原分数未复现')
                identities[name] = {'checkpoint': str(path), 'sha256': checksum(path)}
    atomic_json(OUT / 'fixed-models.json', identities)
    full = scores('engine', y, predictions, [r.unit_id.split(':')[0] for r in package.validation])
    np.savez_compressed(OUT / 'official-issued.npz', ids=np.asarray([r.sample_id for r in package.validation]), **predictions)
    x, ids, groups, truth = dataset('engine')
    issued = {}
    for name, path in paths.items():
        folder = OUT / 'models' / name
        export_release(task_dir, path, folder)
        issued[name] = predict('engine', folder, x, metadata)
    np.savez_compressed(OUT / 'external-issued.npz', ids=np.asarray(ids), groups=np.asarray(groups), **issued)
    old = BASE / 'native-balanced-cross-task/engine'
    previous = json.loads((old / 'summary.json').read_bytes())
    if checksum(old / 'external-issued.npz') != previous['issued_sha256']['external-issued.npz']:
        raise ValueError('前次实际发放身份改变')
    with np.load(old / 'external-issued.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], ids):
            raise ValueError('外部行序变化')
        for name, original in (('initial', 'initial'), ('plain', 'current'), ('expanded_balanced', 'extended')):
            if not np.array_equal(issued[name], saved[original]):
                raise ValueError('前次发放未逐值复现')
    result = {'official': full, 'external': scores('engine', truth(), issued, groups),
        'new_training_runs': 0, 'ids_sha256': sha256(json.dumps(ids).encode()).hexdigest(),
        'issued_sha256': {name: checksum(OUT / name) for name in ('official-issued.npz', 'external-issued.npz')}}
    atomic_json(OUT / 'ablation.json', result)
    return result, (x, ids, groups, truth, metadata)


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ablation, inputs = existing()
    counter = {'calls': 0}
    factory = lambda directory, calls: Broker(counter, calls)
    options = {'driver': 'fixture', 'calls': 2, 'stages': 3, 'broker_factory': factory}
    actual = await run('engine', OUT / 'runtime', **options)
    before = counter['calls']
    restored = await run('engine', OUT / 'runtime', resume=True, **options)
    if before != 6 or counter['calls'] != before or restored != actual:
        raise ValueError('三轮调用或完成恢复异常')
    checked = audit(OUT / 'runtime')
    if actual['driver_completed_stages'] != 3:
        raise ValueError('三轮运行未全部完成')
    final = Path(actual['final_checkpoint'])
    atomic_json(OUT / 'final-selection.json', {'checkpoint': str(final), 'sha256': checksum(final),
        'accepted_stages': [r['stage'] for r in actual['stages'] if r['accepted']]})
    x, ids, groups, truth, metadata = inputs
    issued = predict('engine', OUT / 'runtime/release', x, metadata)
    np.savez_compressed(OUT / 'final-external-issued.npz', ids=np.asarray(ids), groups=np.asarray(groups), actual=issued)
    final_scores = scores('engine', truth(), {'actual': issued}, groups)['actual']
    result = {'scope': '既有候选消融与固定工程驱动三轮实际接续，已有外部资料压力检验',
        'ablation': ablation, 'runtime': checked, 'final_external': final_scores,
        'actual_model_calls': 0, 'fixture_calls': counter['calls'], 'completed_resume_new_calls': 0,
        'final_external_issued_sha256': checksum(OUT / 'final-external-issued.npz')}
    atomic_json(OUT / 'verification.json', result)
    print(json.dumps({'ablation': {scope: {k: v['overall']['score'] for k, v in ablation[scope].items()} for scope in ('official', 'external')},
        'runtime': checked, 'final_external': final_scores, 'actual_model_calls': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
