"""对固定官方任务做真实训练、重载、采用判断和完整交付消融。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from harness.models.anchored_residual import AnchoredResidualRegressor
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json
from engineering.cli.workflow.official_components import accepted

VARIANTS = {'hybrid64': {}, 'local64': {'network_weight': 0.},
            'network64': {'local_weight': 0.}, 'hybrid16': {'anchors': 16}}


def worker(args):
    if args.worker == 'train':
        frame = pd.read_csv(args.bundle / 'task/train.csv')
        groups = json.loads((args.bundle / 'task/task.json').read_bytes())['train_groups']
        x, y = frame.iloc[:, :-1], frame.iloc[:, -1]
        start = time.perf_counter()
        model = AnchoredResidualRegressor(**VARIANTS[args.variant]).fit(x, y, groups)
        weights = cloudpickle.dumps(model)
        (args.output / 'model.pkl').write_bytes(weights)
        reloaded = cloudpickle.loads(weights)
        values = {}
        for kind, filename in [('validation', 'validation.csv'), ('confirmation', 'test.csv')]:
            data = pd.read_csv(args.bundle / 'task' / filename)
            before = model.predict(data)
            prediction = reloaded.predict(data)
            np.testing.assert_array_equal(before, prediction)
            values[kind] = prediction
        np.savez_compressed(args.output / 'checks.npz', **values)
        atomic_json(args.output / 'training.json', {'weights_sha256': sha256(weights).hexdigest(),
            'fit_and_checks_seconds': time.perf_counter()-start, 'anchors': model.anchors_,
            'reload_verified': True, 'parameters': model.get_params(),
            'train_sha256': sha256((args.bundle / 'task/train.csv').read_bytes()).hexdigest(),
            'groups_sha256': sha256((args.bundle / 'task/task.json').read_bytes()).hexdigest()})
    else:
        model = cloudpickle.loads((args.output / 'model.pkl').read_bytes())
        data = pd.read_csv(args.output / 'service.csv')
        values = []
        for i in range(len(data)):
            prediction = np.asarray(model.predict(data.iloc[[i]]))
            if prediction.shape != (1,) or not np.isfinite(prediction).all():
                raise ValueError('逐条服务输出无效')
            values.append(float(prediction[0]))
        np.save(args.output / 'service.npy', values)


def execute(args, work, task, variant, kind, timeout):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [str(args.python.resolve()), '-B', str(Path(__file__).resolve()), '--worker', kind,
               '--bundle', str(args.bundle.resolve() / task), '--output', str(work), '--variant', variant]
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    (work / (kind+'-stdout.log')).write_text(result.stdout, encoding='utf-8')
    (work / (kind+'-stderr.log')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(task+'/'+variant+'/'+kind+': '+result.stderr[-1800:])


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for task in ('engine', 'bearing'):
        bundle = args.bundle / task
        oracle = json.loads((bundle / 'oracle.json').read_bytes())
        identity = json.loads((bundle / 'task-identity.json').read_bytes())
        parent_checks = [metric(task, oracle[key+'_y'], pd.read_csv(bundle/'task'/name).iloc[:, -1], oracle[key+'_groups'])
                         for key, name in [('explore', 'validation.csv'), ('confirm', 'test.csv')]]
        variants = list(VARIANTS) if task == 'engine' else ['hybrid64', 'local64']
        for variant in variants:
            work = output / task / variant
            work.mkdir(parents=True)
            execute(args, work, task, variant, 'train', 90)
            with np.load(work / 'checks.npz', allow_pickle=False) as z:
                checks = [metric(task, oracle[key+'_y'], z[kind], oracle[key+'_groups'])
                          for key, kind in [('explore', 'validation'), ('confirm', 'confirmation')]]
                if task == 'engine' and variant == 'hybrid64':
                    original = args.baseline/'engine/comind-recovered'
                    candidate = json.loads((original/'decision.json').read_bytes())['candidate']
                    with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as old:
                        for kind in ('validation', 'confirmation'):
                            np.testing.assert_allclose(z[kind], old[kind], rtol=0, atol=1e-5)
            adopted = all(accepted(task, a, b) for a, b in zip(parent_checks, checks, strict=True))
            atomic_json(work / 'decision.json', {'adopted': adopted, 'parent_checks': parent_checks,
                                                'candidate_checks': checks})
            with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
                future, baseline, x = z['future'], z['baseline'], z['service_x']
            columns = json.loads((bundle/'task/task.json').read_bytes())['feature_columns']
            pd.DataFrame(x[future], columns=columns).to_csv(work/'service.csv', index=False)
            execute(args, work, task, variant, 'serve', 240)
            predicted = np.load(work/'service.npy', allow_pickle=False)
            if task == 'engine' and variant == 'hybrid64':
                np.testing.assert_allclose(predicted, np.load(args.baseline/'engine/comind-recovered/serving/prediction.npy'),
                                           rtol=0, atol=1e-5)
            direct = baseline.copy()
            direct[future] = predicted
            actual = direct if adopted else baseline.copy()
            np.savez_compressed(work/'issued.npz', ids=np.asarray(oracle['service_ids']), baseline=baseline, direct=direct, actual=actual)
            atomic_json(work/'predictions-complete.json', {'done': True, 'rows': len(actual)})
            from engineering.cli.workflow import public_maintenance as public
            package = public.PACKAGES[task][2](Path(identity['archive']))
            answers = {r.sample_id: r.label for r in package.validation}
            y = np.asarray([answers[k] for k in oracle['service_ids']])
            groups = np.asarray(oracle['service_groups'])
            scores = {name: {'complete': metric(task, y, values, groups),
                             'after_maintenance': metric(task, y[future], values[future], groups[future])}
                      for name, values in [('parent', baseline), ('direct', direct), ('actual', actual)]}
            report = {'adopted': adopted, 'checks': checks, 'scores': scores,
                      'training': json.loads((work/'training.json').read_bytes()),
                      'author_prediction_reproduced': task == 'engine' and variant == 'hybrid64'}
            atomic_json(work/'scores.json', report)
            results[task+'/'+variant] = report
            print(json.dumps({'case': task+'/'+variant, 'adopted': adopted,
                              'actual_rmse': scores['actual']['complete']['score'],
                              'direct_rmse': scores['direct']['complete']['score']}, ensure_ascii=False), flush=True)
    atomic_json(output/'scores.json', {'scope': '官方已见开发任务的真实组件消融及迁移', 'results': results,
                                      'new_training': 6, 'new_model_calls': 0})


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--worker', choices=('train', 'serve'))
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--python', type=Path, default=ROOT/'engineering/.environments/mle-star/Scripts/python.exe')
    args = p.parse_args()
    worker(args) if args.worker else run(args)
