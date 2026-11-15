"""四类冻结任务检验源父预测捷径；不改变目标反馈和隐藏确认用途。"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import warnings

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import metric, accepted
from harness.models.binary_publication_bank import THRESHOLDS
from harness.models.crossfit_parent_feature import crossfit_parent
from harness.runtime.storage import atomic_json

TASKS = ['sgp40', 'secom', 'wind', 'ncmapss_ds06']
VARIANTS = ['naive_parent', 'raw', 'crossfit_parent']


def run_case(task, args):
    source, out = args.bundle/task, args.output/task
    out.mkdir()
    identity = json.loads((source/'task-identity.json').read_bytes())
    for section, prefix in [('task_files', source/'task'), ('private_files', source)]:
        for name, checksum in identity[section].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('原任务资料身份改变')
    meta = json.loads((source/'task/task.json').read_bytes())
    oracle = json.loads((source/'oracle.json').read_bytes())
    train = pd.read_csv(source/'task/train.csv'); y = train.pop('target').to_numpy()
    if train.columns[-1] != 'parent_value':
        raise ValueError('父预测列不在约定位置')
    x = train.iloc[:, :-1].to_numpy()
    parent = cloudpickle.loads((source/'parent.pkl').read_bytes())
    # 原源父版身份与训练时实际输入都核对，随后克隆超参重新做折外拟合。
    # 原任务对父版输出已有非负下限；折外特征也执行相同变换。
    original = np.maximum(0., np.asarray(parent.predict(x)))
    if not np.allclose(original, train['parent_value'], rtol=1e-10, atol=1e-8):
        raise ValueError('原父预测训练特征不能重现')
    classification = task in ['sgp40', 'secom']
    oof, folds, parents, fit_record = crossfit_parent(parent, x, y, meta['train_groups'], classification=classification)
    oof = np.maximum(0., oof)
    (out/'fold-parents.pkl').write_bytes(cloudpickle.dumps(parents))
    np.savez_compressed(out/'source-crossfit.npz', target=y, original=original, oof=oof, fold=folds)
    fit_record.update(parent_weights_sha256=sha256((source/'parent.pkl').read_bytes()).hexdigest(),
                      fold_weights_sha256=sha256((out/'fold-parents.pkl').read_bytes()).hexdigest(),
                      source_predictions_sha256=sha256((out/'source-crossfit.npz').read_bytes()).hexdigest(),
                      original_metric=metric(task, y, original, meta['train_groups']),
                      crossfit_metric=metric(task, y, oof, meta['train_groups']))
    atomic_json(out/'crossfit.json', fit_record)
    explore = pd.read_csv(source/'task/validation.csv')
    confirm = pd.read_csv(source/'task/test.csv')
    eval_y, eval_g = np.asarray(oracle['explore_y']), np.asarray(oracle['explore_groups'])
    con_y, con_g = np.asarray(oracle['confirm_y']), np.asarray(oracle['confirm_groups'])
    baseline_explore = metric(task, eval_y, explore.iloc[:, -1], eval_g)
    baseline_confirm = metric(task, con_y, confirm.iloc[:, -1], con_g)
    variants, frozen = {}, {}
    for variant in VARIANTS:
        folder = out/variant; folder.mkdir()
        columns = list(train.columns) if variant != 'raw' else list(train.columns[:-1])
        estimator = (RandomForestClassifier(n_estimators=96, max_depth=6, class_weight='balanced', n_jobs=1, random_state=17)
                     if classification else ExtraTreesRegressor(n_estimators=96, max_depth=6, n_jobs=1, random_state=17))
        model = make_pipeline(ColumnTransformer([('features', 'passthrough', columns)], remainder='drop'), estimator)
        fitting = train.copy()
        if variant == 'crossfit_parent':
            fitting['parent_value'] = oof
        started = time.monotonic(); model.fit(fitting, y); seconds = time.monotonic()-started
        payload = cloudpickle.dumps(model); (folder/'model.pkl').write_bytes(payload)
        model = cloudpickle.loads(payload)
        # 阈值只在探索窗口选择一次；隐藏确认不参与改阈值或改模型。
        if classification:
            ep = np.asarray(model.predict_proba(explore))[:, 1]
            options = [(t, metric(task, eval_y, np.where(ep > t, 2, 1), eval_g)) for t in THRESHOLDS]
            threshold, scored = max(options, key=lambda p: (p[1]['score'], p[0]))
            cp = np.asarray(model.predict_proba(confirm))[:, 1]
            con = metric(task, con_y, np.where(cp > threshold, 2, 1), con_g)
            np.savez_compressed(folder/'maintenance.npz', explore_probability=ep, confirm_probability=cp)
        else:
            threshold = None
            ep, cp = np.maximum(model.predict(explore), 0), np.maximum(model.predict(confirm), 0)
            scored, con = metric(task, eval_y, ep, eval_g), metric(task, con_y, cp, con_g)
            np.savez_compressed(folder/'maintenance.npz', explore=ep, confirm=cp)
        variants[variant] = {'explore': scored, 'confirm': con, 'threshold': threshold,
                             'adopted': accepted(task, baseline_explore, scored) and accepted(task, baseline_confirm, con),
                             'weights_sha256': sha256(payload).hexdigest(), 'fit_seconds': seconds}
        frozen[variant] = (model, threshold)
        atomic_json(folder/'decision.json', variants[variant])
    pools = {}
    for name, names in [('base_pool', VARIANTS[:2]), ('augmented_pool', VARIANTS)]:
        selected = min(names, key=lambda n: ((-1 if classification else 1)*variants[n]['explore']['score'], names.index(n)))
        pools[name] = {'selected': selected, 'adopted': variants[selected]['adopted']}
    atomic_json(out/'decisions-sealed.json', {'variants': variants, 'pools': pools, 'service_labels_opened': False})
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        baseline, future = z['baseline'], z['future']
        pd.DataFrame(z['service_x'][future], columns=meta['feature_columns']).to_csv(out/'service.csv', index=False)
    service = pd.read_csv(out/'service.csv')
    predictions = {}
    for variant, (model, threshold) in frozen.items():
        # 仅真正通过确认的模型进入真实服务；被拒绝者保持父版，不浪费整段服务计算。
        full = baseline.copy(); started = time.monotonic()
        if variants[variant]['adopted']:
            delivered = []
            for row in range(len(service)):
                values = service.iloc[[row]]
                if classification:
                    delivered.append(2 if model.predict_proba(values)[0, 1] > threshold else 1)
                else:
                    delivered.append(max(float(model.predict(values)[0]), 0.))
            full[future] = delivered
        predictions[variant] = full
        variants[variant]['serving_seconds'] = time.monotonic()-started
    np.savez_compressed(out/'issued.npz', ids=np.asarray(oracle['service_ids']), **predictions)
    # 全部选择与真实交付已经封存，才读取后续答案。
    with np.load(source/'private-answers.npz', allow_pickle=False) as z:
        answer, ids = z['y'], z['ids']
    if not np.array_equal(ids, np.asarray(oracle['service_ids'])):
        raise ValueError('服务答案ID不同')
    groups = np.asarray(oracle['service_groups'])
    for name, values in predictions.items():
        if not np.array_equal(values[~future], baseline[~future]):
            raise ValueError('维护前交付改变')
        variants[name].update(complete=metric(task, answer, values, groups),
                              future=metric(task, answer[future], values[future], groups[future]))
        if classification:
            variants[name].update(future_detected=int(np.sum((answer[future] == 2) & (values[future] == 2))),
                                  future_false_alarms=int(np.sum((answer[future] != 2) & (values[future] == 2))))
    for name, selected in pools.items():
        record = variants[selected['selected']]
        selected.update(complete=record['complete'], future=record['future'])
    result = {'task': task, 'variants': variants, 'pools': pools,
              'baseline_complete': metric(task, answer, baseline, groups),
              'source_parent_fits': len(parents), 'candidate_fits': len(VARIANTS), 'research_model_calls': 0,
              'issued_sha256': sha256((out/'issued.npz').read_bytes()).hexdigest(),
              'scope': '固定树模型的父预测特征因果消融与合法维护选择；非新完整LLM搜索'}
    atomic_json(out/'scores.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', nargs='+', choices=TASKS, default=TASKS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不得覆盖既有消融或负结果')
    args.output.mkdir(parents=True)
    # 原父版以数组训练，保留其输入；该已知特征名提示不影响算法，避免逐行重复日志。
    warnings.filterwarnings('ignore', message='X does not have valid feature names, but (StandardScaler|HistGradientBoostingRegressor) was fitted with feature names')
    results = {task: run_case(task, args) for task in args.task}
    atomic_json(args.output/'scores.json', results)


if __name__ == '__main__':
    main()
