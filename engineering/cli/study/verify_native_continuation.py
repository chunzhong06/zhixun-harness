"""以三份官方检查点验证续训源码接口；不把子集工程检查报告为性能实验。"""
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
from engineering.cli.workflow.native_continuation import execute
from harness.models.native_continuation import tensor_identity, validate_program
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

DEFAULT = '''def train(session):
    for epoch in range(2):
        session.train_epoch()
'''
HEAD = '''def train(session):
    session.train_epoch(learning_rate=0.001, trainable=['w2', 'b2'])
'''
HUBER = '''import numpy as np
def robust(logits, target):
    error = logits[:, 0] - target
    absolute = np.abs(error)
    loss = np.where(absolute <= 1.0, 0.5 * error ** 2, absolute - 0.5)
    return loss, np.clip(error, -1.0, 1.0)[:, None]
def train(session):
    session.train_epoch(loss_gradient=robust)
'''
WIDER = '''def train(session):
    session.expand_hidden(128)
    session.train_epoch(learning_rate=0.0001)
'''


def case(task, output):
    profile = {'gas': 'A', 'bearing': 'B', 'engine': 'C'}[task]
    archive = next((ROOT / 'data/official' / task).glob('*.zip'))
    output.mkdir(parents=True)
    inputs = output / 'inputs'
    inputs.mkdir()
    with ZipFile(archive) as z, TemporaryDirectory(dir=output, prefix='temporary-') as temp:
        temp = Path(temp)
        with _trainer(z, temp, profile) as trainer:
            parent = inputs / 'parent.npz'
            parent.write_bytes(z.read('checkpoint.npz'))
            original = trainer.Model.load(parent)
            for part, count in [('train', 512), ('validation', 12)]:
                raw = b'\n'.join(z.read('public/' + part + '.jsonl').splitlines()[:count]) + b'\n'
                (temp / (part + '.jsonl')).write_bytes(raw)
            train = trainer.read_dataset(temp / 'train.jsonl', original.meta['task'], original.meta['classes'], original.meta['paths'])
            validation = trainer.read_dataset(temp / 'validation.jsonl', original.meta['task'], original.meta['classes'], original.meta['paths'])
            np.savez_compressed(inputs / 'train.npz', x=train['x'], y=train['y'])
            np.savez_compressed(inputs / 'evaluation-inputs.npz', explore=validation['x'][:6], confirm=validation['x'][6:])
            atomic_json(inputs / 'task.json', {'task': task, 'profile': profile, 'stage_id': 'engineering-parent-chain',
                'archive': str(archive.resolve()), 'archive_sha256': sha256(archive.read_bytes()).hexdigest(),
                'feature_paths': original.meta['paths'], 'inputs_sha256': {
                    name: sha256((inputs / name).read_bytes()).hexdigest() for name in ('train.npz', 'evaluation-inputs.npz')}})
            source = output / 'default.py'
            source.write_text(DEFAULT, encoding='utf-8')
            first = execute(inputs, parent, source, output / 'first', expected_parent_sha256=sha256(parent.read_bytes()).hexdigest())
            original.fit(train['x'], train['y'], 2)
            child = trainer.Model.load(output / 'first/checkpoint.npz')
            differences = [float(np.max(np.abs(a[k] - b[k]))) for a, b in
                [(original.params, child.params), (original.first, child.first), (original.second, child.second)] for k in a]
            assert max(differences) <= 1e-12, differences
            assert original.meta == child.meta
            assert first['loaded_tensors'] == first['before_first_gradient']
            source = output / 'head.py'
            source.write_text(HEAD, encoding='utf-8')
            second = execute(inputs, output / 'first/checkpoint.npz', source, output / 'second',
                             expected_parent_sha256=first['checkpoint_sha256'])
            assert second['loaded_tensors'] == first['final_tensors']
            assert set(second['parameter_changes']) == {'w2', 'b2'}
            for key in ('w1', 'b1', 'adam_first_w1', 'adam_first_b1', 'adam_second_w1', 'adam_second_b1'):
                assert second['loaded_tensors'][key] == second['final_tensors'][key]
            # 模拟拒绝第二候选：下一次请求仍以实际选中的first为父版。
            third = execute(inputs, output / 'first/checkpoint.npz', source, output / 'after_rejection',
                            expected_parent_sha256=first['checkpoint_sha256'])
            assert third['checkpoint_sha256'] == second['checkpoint_sha256']
            assert third['initial_steps'] == first['final_steps']
            source = output / 'wider.py'
            source.write_text(WIDER, encoding='utf-8')
            wider = execute(inputs, parent, source, output / 'wider',
                            expected_parent_sha256=sha256(parent.read_bytes()).hexdigest())
            assert wider['migration']['retained_parent_tensors'] == wider['loaded_tensors']
            assert wider['migration']['initial_prediction_max_difference'] <= 1e-9
            expanded = trainer.Model.load(output / 'wider/checkpoint.npz')
            assert expanded.meta['hidden'] == 128
            if task != 'gas':
                source = output / 'huber.py'
                source.write_text(HUBER, encoding='utf-8')
                robust = execute(inputs, parent, source, output / 'huber', expected_parent_sha256=sha256(parent.read_bytes()).hexdigest())
                assert robust['history'][0]['custom_gradient'] and robust['parameter_changes']
            try:
                execute(inputs, parent, source, output / 'wrong-parent', expected_parent_sha256='0' * 64)
            except ValueError:
                assert not (output / 'wrong-parent').exists()
            else:
                raise AssertionError('错误父版身份未被拒绝')
            for bad in ['def train(session):\n    session._model.params.clear()',
                        'import os\ndef train(session):\n    session.train_epoch()']:
                try:
                    validate_program(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError('越过续训接口的源码未被拒绝')
            requests = [{'task': original.meta['task'], 'inputs': json.loads(line)['inputs']}
                        for line in (temp / 'validation.jsonl').read_text(encoding='utf-8').splitlines()]
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            process = subprocess.run([sys.executable, '-B', 'baselines/predict.py', '--checkpoint', 'checkpoint.npz'],
                cwd=output / 'wider', env=environment, input='\n'.join(json.dumps(r) for r in requests),
                capture_output=True, text=True, encoding='utf-8', timeout=30)
            if process.returncode:
                raise RuntimeError(process.stderr)
            issued = [json.loads(line)['outputs'] for line in process.stdout.splitlines()]
            actual = np.asarray([[v['score'] for v in row] for row in issued]) if task == 'gas' else np.asarray([row['value'] for row in issued])
            model = trainer.Model.load(output / 'wider/checkpoint.npz')
            expected = np.asarray([model.predict(x[None, :])[0] for x in validation['x']])
            assert np.allclose(actual, expected, rtol=0, atol=1e-12)
            result = {'task': task, 'scope': '官方原件上的工程子集检查，不是完整性能对照',
                'train_rows': len(train['x']), 'independent_prediction_rows': len(actual),
                'official_training_max_tensor_difference': max(differences),
                'parent_chain': [first['parent_checkpoint_sha256'], first['checkpoint_sha256'], second['checkpoint_sha256']],
                'optimizer_and_scalers_preserved': True, 'frozen_layer_preserved': True,
                'rejected_candidate_not_used_as_parent': True, 'wrong_parent_rejected': True,
                'independent_prediction_equal': True, 'artifact_bytes': second['artifact_bytes']}
            result.update(expanded_hidden=128, expansion_parent_state_preserved=True,
                          expansion_initial_prediction_max_difference=wider['migration']['initial_prediction_max_difference'],
                          expansion_independent_prediction_equal=True, expansion_artifact_bytes=wider['artifact_bytes'])
            atomic_json(output / 'verification.json', result)
            return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [case(task, (args.output / task).resolve()) for task in ('gas', 'bearing', 'engine')]
    atomic_json(args.output / 'verification.json', {'passed': True, 'cases': results})
    print(json.dumps({'passed': True, 'cases': results}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
