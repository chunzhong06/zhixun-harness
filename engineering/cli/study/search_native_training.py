"""只在官方全量公开验证集选续训方案，逐设备报告；不是独立迁移结论。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.workflow.native_continuation import execute
from harness.data.official_gas import read_official_gas
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.models.official_rul_cpu import _trainer
from harness.integrations.native_research import measure, export_release, exploration_diagnostics
from harness.runtime.storage import atomic_json


def source(*, lr, epochs, head, balanced=False, robust=False, classification=False, hidden=None):
    """同一训练逻辑按任务类型适配损失，不按设备名称或评价答案制定权重。"""
    lines = ['import numpy as np']
    if robust and classification:
        lines += ['def objective(logits, target):',
                  '    z = logits - logits.max(axis=1, keepdims=True)',
                  '    probability = np.exp(z)', '    probability /= probability.sum(axis=1, keepdims=True)',
                  '    target_probability = np.full_like(probability, 0.05 / probability.shape[1])',
                  '    target_probability[np.arange(len(target)), target] += 0.95',
                  '    loss = -np.sum(target_probability * np.log(np.maximum(probability, 1e-300)), axis=1)',
                  '    return loss, probability - target_probability']
    elif robust:
        lines += ['def objective(logits, target):', '    error = logits[:, 0] - target',
                  '    absolute = np.abs(error)',
                  '    loss = np.where(absolute <= 0.5, 0.5 * error ** 2, 0.5 * (absolute - 0.25))',
                  '    return loss, np.clip(error, -0.5, 0.5)[:, None]']
    lines += ['def train(session):']
    if hidden is not None:
        lines += [f"    if session.metadata['hidden'] < {hidden}:", f'        session.expand_hidden({hidden})']
    if balanced:
        lines += ['    identity = session.targets' if classification else '    identity = session.groups',
                  '    names, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)',
                  '    weights = 1.0 / counts[inverse]']
    args = [f'learning_rate={lr}']
    if head: args += ["trainable=['w2', 'b2']"]
    if balanced: args += ['sample_weights=weights']
    if robust: args += ['loss_gradient=objective']
    lines += [f'    for epoch in range({epochs}):', '        session.train_epoch(' + ', '.join(args) + ')']
    return '\n'.join(lines) + '\n'


def programs(classification, space='training'):
    yield 'official-one-epoch', 'def train(session):\n    session.train_epoch()\n'
    if space == 'width':
        for hidden in (64, 128):
            for epochs in (4, 12):
                for balanced in (False, True):
                    name = f'width{hidden}-e{epochs}-balanced{int(balanced)}'
                    yield name, source(lr=0.001, epochs=epochs, head=False, balanced=balanced,
                                       classification=classification, hidden=hidden)
        return
    for lr in (0.0001, 0.001):
        for epochs in (4, 12):
            for head in (False, True):
                name = f'plain-lr{lr}-e{epochs}-' + ('head' if head else 'all')
                yield name, source(lr=lr, epochs=epochs, head=head)
    for epochs in (4, 12):
        for head in (False, True):
            name = f'balanced-e{epochs}-' + ('head' if head else 'all')
            yield name, source(lr=0.001, epochs=epochs, head=head, balanced=True, classification=classification)
    for balanced in (False, True):
        for head in (False, True):
            name = f'robust-balanced{int(balanced)}-' + ('head' if head else 'all')
            yield name, source(lr=0.001, epochs=12, head=head, balanced=balanced, robust=True, classification=classification)


def run_task(task, folder, space='training'):
    folder.mkdir(parents=True)
    name, identity, reader = ('cpu-a-guide-20261110-v1.zip', PACKAGE_SHA, read_official_gas) if task == 'gas' else PACKAGES[task]
    archive = ROOT / 'data/official' / task / name
    assert sha256(archive.read_bytes()).hexdigest() == identity
    package = reader(archive)
    inputs = folder / 'inputs'
    inputs.mkdir()
    parent = inputs / 'initial.npz'
    parent.write_bytes(package.checkpoint)
    x = np.asarray([r.values for r in package.train], dtype=float)
    y = np.asarray([r.label - 1 if task == 'gas' else r.label for r in package.train])
    groups = np.asarray([str(r.unit_id or r.group) for r in package.train])
    vx = np.asarray([r.values for r in package.validation], dtype=float)
    vy = np.asarray([r.label for r in package.validation])
    vg = [str(r.unit_id or r.group) for r in package.validation]
    np.savez_compressed(inputs / 'train.npz', x=x, y=y, groups=groups)
    np.savez_compressed(inputs / 'evaluation-inputs.npz', validation=vx)
    profile = {'gas': 'A', 'bearing': 'B', 'engine': 'C'}[task]
    with ZipFile(archive) as z, TemporaryDirectory(dir=folder, prefix='temporary-') as temp:
        with _trainer(z, Path(temp), profile) as trainer:
            original = trainer.Model.load(parent)
            prediction = original.predict(vx)
            paths = original.meta['paths']
        sample_requests = [{'task': original.meta['task'], 'inputs': json.loads(row)['inputs']}
                           for row in z.read('public/validation.jsonl').splitlines()[:12]]
    atomic_json(inputs / 'task.json', {'task': task, 'profile': profile, 'stage_id': 'full-public-validation',
        'archive': str(archive.resolve()), 'archive_sha256': identity, 'feature_paths': paths,
        'inputs_sha256': {name: sha256((inputs / name).read_bytes()).hexdigest()
                          for name in ('train.npz', 'evaluation-inputs.npz')}})
    np.savez_compressed(folder / 'initial-predictions.npz', validation=prediction)
    rows = [{'name': 'initial', 'checkpoint': str(parent.resolve()), 'score': measure(task, vy, prediction),
             'diagnostics': exploration_diagnostics(task, vy, prediction, vg)}]
    predictions = [prediction]
    for number, (name, program) in enumerate(programs(task == 'gas', space), 1):
        target = folder / f'{number:02d}-{name}'
        path = folder / f'{number:02d}-{name}.py'
        path.write_text(program, encoding='utf-8')
        report = execute(inputs, parent, path, target, expected_parent_sha256=sha256(package.checkpoint).hexdigest(), max_epochs=12)
        with np.load(target / 'predictions.npz', allow_pickle=False) as stored:
            prediction = stored['validation'].copy()
        rows.append({'name': name, 'checkpoint': str((target / 'checkpoint.npz').resolve()),
                     'source': str(path.resolve()), 'score': measure(task, vy, prediction),
                     'diagnostics': exploration_diagnostics(task, vy, prediction, vg),
                     'training_seconds': report['training_seconds'], 'parent_sha256': report['parent_checkpoint_sha256']})
        predictions.append(prediction)
        atomic_json(folder / 'candidates.json', rows)
    best = min(range(len(rows)), key=lambda i: rows[i]['score']['loss'])
    release = export_release(inputs, Path(rows[best]['checkpoint']), folder / 'release')
    env = os.environ.copy();env.pop('PYTHONPATH', None);env['PYTHONDONTWRITEBYTECODE'] = '1'
    test = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'],
                          cwd=folder / 'release', env=env, capture_output=True, text=True, encoding='utf-8',
                          input='\n'.join(json.dumps(r) for r in sample_requests), timeout=30)
    if test.returncode: raise RuntimeError(test.stderr)
    outputs = [json.loads(line)['outputs'] for line in test.stdout.splitlines()]
    actual = np.asarray([[r['score'] for r in output] for output in outputs]) if task == 'gas' else np.asarray([output['value'] for output in outputs])
    assert np.allclose(actual, predictions[best][:12], atol=1e-10, rtol=0)
    report = {'task': task, 'scope': '全量官方公开验证集开发选案；不是独立迁移或现场驱动增量',
              'train_rows': len(y), 'validation_rows': len(vy), 'training_groups': len(set(groups)),
              'baseline': rows[0], 'selected': rows[best], 'release': release,
              'independent_prediction_rows': len(actual), 'candidates': rows}
    atomic_json(folder / 'selection.json', report)
    return report


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--space', choices=['training', 'width'], default='training')
    args = parser.parse_args()
    reports = [run_task(task, (args.output / task).resolve(), args.space) for task in ('gas', 'bearing', 'engine')]
    atomic_json(args.output / 'summary.json', {'tasks': reports})
    print(json.dumps([{'task': r['task'], 'baseline': r['baseline']['score'], 'selected': r['selected']['name'],
                       'score': r['selected']['score'], 'release_bytes': r['release']['artifact_bytes']} for r in reports], ensure_ascii=False))


if __name__ == '__main__': main()
