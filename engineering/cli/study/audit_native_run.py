"""核对真实阶段父版、逐条服务预测与独立发布目录，不重新训练或挑选候选。"""
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
from harness.models.native_continuation import tensor_identity
from harness.models.complete_checkpoint import load_checkpoint
from harness.models.official_rul_cpu import _trainer
from harness.integrations.native_research import measure
from harness.runtime.storage import atomic_json


def audit(folder):
    folder = Path(folder).resolve()
    run = json.loads((folder / 'run.json').read_bytes())
    task = json.loads((folder / 'task/task.json').read_bytes())
    oracle = json.loads((folder / 'oracle.json').read_bytes())
    assert run['status'] == 'completed'
    parent = folder / 'task/initial.npz'
    trials = 0
    training_workers = 0
    reused = 0
    chain = []
    issued = []
    with np.load(folder / 'service-inputs.npz', allow_pickle=False) as data:
        service_x = data['x'].copy()
    with ZipFile(task['archive']) as package, TemporaryDirectory(dir=folder, prefix='temporary-audit-') as temp:
        with _trainer(package, Path(temp), task['profile']) as trainer:
            for stage in run['stages']:
                stage_dir = folder / f"stage-{stage['stage']}"
                before = sha256(parent.read_bytes()).hexdigest()
                loaded = tensor_identity(load_checkpoint(parent, trainer.Model))
                for execution in sorted((stage_dir / 'executions').glob('*/execution.json')):
                    record = json.loads(execution.read_bytes())
                    trials += 1
                    training_workers += int(record.get('launched_training_worker', True))
                    reused += int(record.get('execution_reused', False))
                    if record['returncode']:
                        continue
                    for candidate in record['candidates']:
                        report = json.loads(Path(candidate['report']).read_bytes())
                        assert report['parent_checkpoint_sha256'] == before
                        assert report['loaded_tensors'] == loaded
                        if report.get('migration') is None:
                            assert loaded == report['before_first_gradient']
                        else:
                            assert report['migration']['retained_parent_tensors'] == loaded
                            if report['migration']['kind'] == 'source-trained-input-components':
                                assert report['migration']['source_only'] is True
                            else:
                                assert report['migration']['initial_prediction_max_difference'] <= 1e-9
                        assert report['final_steps'] > report['initial_steps']
                        assert sha256(Path(candidate['checkpoint']).read_bytes()).hexdigest() == candidate['weights_sha256']
                if stage['accepted']:
                    assert stage['confirmation']['candidate']['loss'] < stage['confirmation']['parent']['loss']
                    parent = Path(stage['selected']['checkpoint'])
                assert Path(stage['published_checkpoint']) == parent
                assert sha256(parent.read_bytes()).hexdigest() == stage['published_sha256']
                model = load_checkpoint(parent, trainer.Model)
                with np.load(stage_dir / 'issued.npz', allow_pickle=False) as data:
                    indices, prediction = data['indices'].copy(), data['prediction'].copy()
                expected = np.asarray([model.predict(x[None, :])[0] for x in service_x[indices]])
                assert np.allclose(prediction, expected, atol=1e-12, rtol=0)
                issued.append(prediction)
                chain.append({'stage': stage['stage'], 'parent_sha256': before,
                              'published_sha256': stage['published_sha256'], 'accepted': stage['accepted']})
            release = folder / 'release'
            manifest = json.loads((release / 'manifest.json').read_bytes())
            assert sha256((release / 'checkpoint.npz').read_bytes()).hexdigest() == sha256(parent.read_bytes()).hexdigest()
            size = sum(p.stat().st_size for p in release.rglob('*') if p.is_file())
            assert size == manifest['artifact_bytes'] and size <= 64 * 1024**2
            raw = [json.loads(line) for line in package.read('public/validation.jsonl').splitlines()[:12]]
            model = load_checkpoint(parent, trainer.Model)
            from importlib import import_module
            data_module = import_module(trainer.__package__ + '.data')
            expected = np.asarray([model.predict(np.asarray(data_module.flatten(row['inputs'])[1])[None, :])[0] for row in raw])
            requests = [{'task': model.meta['task'], 'inputs': row['inputs']} for row in raw]
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            process = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'],
                cwd=release, env=environment, input='\n'.join(json.dumps(row) for row in requests),
                capture_output=True, text=True, encoding='utf-8', timeout=30)
            if process.returncode:
                raise RuntimeError(process.stderr)
            rows = [json.loads(line)['outputs'] for line in process.stdout.splitlines()]
            prediction = np.asarray([[v['score'] for v in row] for row in rows]) if run['task'] == 'gas' else np.asarray([row['value'] for row in rows])
            assert np.allclose(prediction, expected, atol=1e-12, rtol=0)
    actual = np.concatenate(issued)
    with np.load(folder / 'issued.npz', allow_pickle=False) as saved:
        assert np.array_equal(actual, saved['actual'])
        assert measure(run['task'], oracle['serve']['y'], actual) == run['scores']['actual']
        assert measure(run['task'], oracle['serve']['y'], saved['baseline']) == run['scores']['baseline']
    report = {'passed': True, 'task': run['task'], 'trials': trials, 'model_calls': sum(s['calls'] for s in run['stages']),
              'actual_model_calls': sum(s['calls'] for s in run['stages'] if s['driver'] != 'fixture'),
              'launched_training_workers': training_workers, 'reused_candidates': reused,
              'driver_completed_stages': run['driver_completed_stages'], 'parent_chain': chain,
              'issued_rows': len(actual), 'release_independent_rows': len(prediction),
              'release_bytes': size, 'scores': run['scores']}
    atomic_json(folder / 'audit.json', report)
    return report


def main():
    parser = ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), ensure_ascii=False))


if __name__ == '__main__':
    main()
