"""复用既有权重诊断采用风险；全部验证选案与跨设备压力检验明确分开。"""
from hashlib import sha256
from itertools import permutations
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
from engineering.cli.study.evaluate_native_transfer import dataset, input_object
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.study.run_rul_maintenance import partition, PROTOCOL as ROLE_PROTOCOL
from harness.integrations.native_research import export_release
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'bearing-adoption-audit'
AGGREGATE = 'width128-e4-balanced1'


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(p) - y))))


def metrics(y, predictions, groups):
    owners = np.asarray(groups)
    return {name: {'overall': rmse(y, p), 'by_device': {
        u: {'rows': int(np.sum(owners == u)), 'rmse': rmse(y[owners == u], p[owners == u])}
        for u in sorted(set(groups))}} for name, p in predictions.items()}


def load_existing(package, archive):
    """重新预测并校对所有旧候选，避免用未绑定权重的汇总表作结论。"""
    x = np.asarray([r.values for r in package.validation])
    y = np.asarray([r.label for r in package.validation])
    groups = [r.unit_id for r in package.validation]
    candidates, predictions, seen = {}, {}, {}
    with ZipFile(archive) as z, TemporaryDirectory(dir=OUT, prefix='temporary-') as temp:
        with _trainer(z, Path(temp), 'B') as trainer:
            for scope in ('full-validation-search', 'width-search'):
                folder = BASE / scope / 'bearing'
                saved = json.loads((folder / 'selection.json').read_bytes())
                task = json.loads((folder / 'inputs/task.json').read_bytes())
                if task['archive_sha256'] != checksum(archive):
                    raise ValueError('原候选官方原包身份不一致')
                with np.load(folder / 'inputs/evaluation-inputs.npz', allow_pickle=False) as stored:
                    if not np.array_equal(stored['validation'], x):
                        raise ValueError('原候选评价行序不一致')
                for row in saved['candidates']:
                    path = Path(row['checkpoint'])
                    identity = checksum(path)
                    predicted_file = (folder / 'initial-predictions.npz' if row['name'] == 'initial'
                                      else path.parent / 'predictions.npz')
                    with np.load(predicted_file, allow_pickle=False) as stored:
                        predicted = stored['validation'].copy()
                    actual = trainer.Model.load(path).predict(x)
                    if not np.allclose(actual, predicted, rtol=0, atol=1e-10):
                        raise ValueError('旧预测未复现：' + row['name'])
                    score = rmse(y, predicted)
                    if abs(score - row['score']['score']) > 1e-9:
                        raise ValueError('旧整体评分未复现')
                    for group in row['diagnostics']['by_group']:
                        ix = np.asarray(groups) == group['group']
                        if abs(rmse(y[ix], predicted[ix]) - group['rmse']) > 1e-9:
                            raise ValueError('旧设备评分未复现')
                    if row['name'] == 'initial':
                        if identity != sha256(package.checkpoint).hexdigest():
                            raise ValueError('初始模型身份不符')
                    else:
                        training = json.loads((path.parent / 'training.json').read_bytes())
                        if training['checkpoint_sha256'] != identity or training['parent_checkpoint_sha256'] != sha256(package.checkpoint).hexdigest():
                            raise ValueError('候选父子权重身份不符')
                    if identity in seen:
                        if not np.array_equal(predictions[seen[identity]], predicted):
                            raise ValueError('同权重不同预测')
                        continue
                    seen[identity] = row['name']
                    candidates[row['name']] = {'checkpoint': str(path), 'sha256': identity,
                        'task_dir': str(folder / 'inputs'), 'predictions_sha256': checksum(predicted_file),
                        'reloaded_prediction_max_difference': float(np.max(np.abs(actual - predicted)))}
                    predictions[row['name']] = predicted
    return candidates, predictions, y, groups


def audit_roles(package, predictions, y):
    """探索选案、确认接受后才评价服务；全轨迹只是同设备上增加确认观测。"""
    spec = json.loads(ROLE_PROTOCOL.read_bytes())
    original, assignment = partition(package, spec)
    groups = np.asarray([r.unit_id for r in package.validation])
    sparse = np.asarray([(r.time_end - 1) % spec['development_observation_stride'] == 0 for r in package.validation])
    units = sorted(set(groups))
    results = []
    for full in (False, True):
        for explore, confirm, serve in permutations(units):
            ex = (groups == explore) & (np.ones(len(y), dtype=bool) if full else sparse)
            co = (groups == confirm) & (np.ones(len(y), dtype=bool) if full else sparse)
            se = groups == serve
            scores = {name: rmse(y[ex], p[ex]) for name, p in predictions.items()}
            proposal = min(scores, key=scores.get)
            for policy, choice in (('fixed_aggregate_candidate', AGGREGATE), ('exploration_selected', proposal)):
                accepted = (scores[choice] < scores['initial'] and
                            rmse(y[co], predictions[choice][co]) < rmse(y[co], predictions['initial'][co]))
                selected = choice if accepted else 'initial'
                results.append({'observations': 'full' if full else 'stride10', 'policy': policy,
                    'roles': {'explore': explore, 'adopt': confirm, 'serve': serve},
                    'original_assignment': assignment == {explore: 'explore', confirm: 'adopt', serve: 'serve'},
                    'explore_rows': int(ex.sum()), 'confirm_rows': int(co.sum()), 'service_rows': int(se.sum()),
                    'proposal': choice, 'accepted': bool(accepted), 'selected': selected,
                    'explore_parent': scores['initial'], 'explore_candidate': scores[choice],
                    'confirm_parent': rmse(y[co], predictions['initial'][co]),
                    'confirm_candidate': rmse(y[co], predictions[choice][co]),
                    'service_parent': rmse(y[se], predictions['initial'][se]),
                    'service_selected': rmse(y[se], predictions[selected][se])})
    return results


def external(candidates, selected):
    x, ids, groups, answers = dataset('bearing')
    with np.load(candidates['initial']['checkpoint'], allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
    paths = [json.loads(p) for p in metadata['paths']]
    requests = '\n'.join(json.dumps({'task': metadata['task'], 'inputs': input_object(paths, row)}) for row in x)
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
    predictions = {}
    for name in dict.fromkeys(('initial', AGGREGATE, selected)):
        info = candidates[name]
        release = OUT / 'external' / name
        export_release(info['task_dir'], info['checkpoint'], release)
        process = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'],
            cwd=release, env=env, input=requests, capture_output=True, text=True, encoding='utf-8', timeout=60)
        if process.returncode:
            raise RuntimeError(process.stderr)
        prediction = np.asarray([json.loads(line)['outputs']['value'] for line in process.stdout.splitlines()])
        if prediction.shape != (len(ids),) or not np.isfinite(prediction).all():
            raise ValueError('独立预测数量或数值无效')
        predictions[name] = prediction
    issued = OUT / 'external/issued.npz'
    np.savez_compressed(issued, ids=np.asarray(ids), groups=np.asarray(groups), **predictions)
    old = BASE / 'frozen-transfer/bearing'
    if checksum(old / 'issued.npz') != json.loads((old / 'scores.json').read_bytes())['issued_sha256']:
        raise ValueError('既有迁移预测身份变更')
    with np.load(old / 'issued.npz', allow_pickle=False) as original:
        if not np.array_equal(original['ids'], ids):
            raise ValueError('已有迁移样本行序不符')
        for current, previous in (('initial', 'initial'), (AGGREGATE, 'selected')):
            if not np.allclose(predictions[current], original[previous], rtol=0, atol=1e-10):
                raise ValueError('已有迁移发放未复现')
    return {'scope': '已有研发暴露的外部设备压力检验，未追加训练或按迁移结果改选',
        'rows': len(ids), 'scores': metrics(answers(), predictions, groups),
        'old_predictions_reproduced': True, 'issued_sha256': checksum(issued)}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    filename, expected, reader = PACKAGES['bearing']
    archive = ROOT / 'data/official/bearing' / filename
    if checksum(archive) != expected:
        raise ValueError('官方原包变化')
    package = reader(archive)
    candidates, predictions, y, groups = load_existing(package, archive)
    scores = metrics(y, predictions, groups)
    baseline = scores['initial']['by_device']
    eligible = [name for name, score in scores.items() if all(
        score['by_device'][unit]['rmse'] <= baseline[unit]['rmse'] for unit in baseline)]
    selected = min(eligible, key=lambda name: scores[name]['overall'])
    # 写明开发选择后，外部入口才开始执行；没有根据外部结果改换候选的分支。
    selection = {'criterion': '公开验证每设备不退步，再取总体RMSE最小；含父版',
                 'eligible': eligible, 'selected': selected, 'candidate': candidates[selected]}
    atomic_json(OUT / 'selection.json', selection)
    roles = audit_roles(package, predictions, y)
    report = {'scope': '复用run407/409的开发诊断，不改变既有实验或当前交付包',
        'new_training_runs': 0, 'model_service_calls': 0, 'unique_candidates': len(candidates),
        'rows': len(y), 'devices': len(set(groups)), 'candidates': candidates,
        'official_scores': scores, 'selection': selection, 'role_rotations': roles,
        'external': external(candidates, selected)}
    atomic_json(OUT / 'summary.json', report)
    print(json.dumps({'new_training_runs': 0, 'unique_candidates': len(candidates), 'selected': selected,
        'official': {k: scores[k] for k in ('initial', AGGREGATE, selected)},
        'original_roles': [r for r in roles if r['original_assignment']],
        'external': report['external']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
