"""回放真实冻结预测，严格按批反馈比较完整评分与逐批评分的持续选案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import accepted, metric
from harness.models.cumulative_classification_control import MODES, CumulativeClassificationControl, confusion, macro_f1
from harness.runtime.storage import atomic_json

TASKS = ['sgp40', 'secom']
THRESHOLDS = [.01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5]
PHASES = ['explore', 'confirm', 'future']


def checked_archive(folder):
    meta = json.loads((folder/'complete.json').read_bytes())
    if not meta['success']:
        raise ValueError('固定模型未成功产生预测')
    for filename, checksum in [('model.pkl', meta['weights_sha256']), ('predictions.npz', meta['predictions_sha256'])]:
        if sha256((folder/filename).read_bytes()).hexdigest() != checksum:
            raise ValueError('固定预测或权重改变')
    with np.load(folder/'predictions.npz', allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    return values, {'weights_sha256': meta['weights_sha256'], 'predictions_sha256': meta['predictions_sha256'], 'folder': str(folder)}


def build_bank(task, base, ablation, bundle, output):
    """只复制真实预测与身份，不读服务答案；候选池对所有选择规则完全相同。"""
    previous = base/('shared-classifier-full-pair' if task == 'sgp40' else 'broad-transfer-full-pair')/task
    sources = {'parent': 'source-parent', 'harness': 'full-s29' if task == 'sgp40' else 'project-published-candidate',
               'forest_parent': 'forest-with-parent', 'forest_raw': 'forest-without-parent'}
    models, identity = {}, {}
    for name, local in sources.items():
        models[name], identity[name] = checked_archive(ablation/task/local)
    with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline = z['future'], z['baseline']
    expected_ids = np.asarray(json.loads((bundle/'oracle.json').read_bytes())['service_ids'])
    published = {}
    for arm in ['project', 'comind']:
        folder = previous/arm
        complete = json.loads((folder/'predictions-complete.json').read_bytes())
        if sha256((folder/'issued.npz').read_bytes()).hexdigest() != complete['issued_sha256']:
            raise ValueError('原完整交付身份改变')
        decision = json.loads((folder/'decision.json').read_bytes())
        if complete['adopted']:
            selected = decision['candidate']
            for weights in [Path(selected['path']).with_suffix('.pkl'), folder/'serving/model.pkl']:
                if sha256(weights.read_bytes()).hexdigest() != selected['weights_sha256']:
                    raise ValueError('实际发布权重改变')
        with np.load(folder/'issued.npz', allow_pickle=False) as z:
            if not np.array_equal(z['ids'], expected_ids):
                raise ValueError('原交付ID顺序改变')
            published[arm] = z['actual'].copy()
            if not np.array_equal(z['actual'][~future], baseline[~future]):
                raise ValueError('原维护前交付不同')
        if arm == 'project':
            if not np.array_equal(models['harness']['future'], published[arm][future]):
                raise ValueError('当前Harness与冻结概率模型的实际类别不同')
        else:
            if complete['adopted']:
                candidate = decision['candidate']; path = Path(candidate['path']).with_suffix('.npz')
                if sha256(path.with_suffix('.pkl').read_bytes()).hexdigest() != candidate['weights_sha256']:
                    raise ValueError('CoMind实际权重变化')
                with np.load(path, allow_pickle=False) as z:
                    models['comind'] = {'explore': z['validation'], 'confirm': z['confirmation'], 'future': published[arm][future]}
            else:
                models['comind'] = {p: models['parent'][p] for p in PHASES}
            identity['comind'] = {'issued_sha256': complete['issued_sha256'], 'adopted': complete['adopted'],
                                  'selected_weights_sha256': decision['candidate']['weights_sha256'] if decision['candidate'] else None}
    if not np.array_equal(models['parent']['future'], baseline[future]):
        raise ValueError('源父版预测不能逐条复现')
    # 固定原发布版始终保留独立动作，避免概率恰为0.5的数值平手改写原行为。
    names = ['harness_fixed', 'comind_fixed', 'parent_fixed', 'constant_1', 'constant_2']
    arrays = {phase: [models['harness'][phase], models['comind'][phase], models['parent'][phase],
                       np.ones_like(models['parent'][phase]), np.full_like(models['parent'][phase], 2)] for phase in PHASES}
    for family in sources:
        for threshold in THRESHOLDS:
            names.append(f'{family}@{threshold:g}')
            for phase in PHASES:
                probability = models[family][phase+'_probability']
                if not np.isfinite(probability).all():
                    raise ValueError('冻结概率非有限')
                arrays[phase].append(np.where(probability > threshold, 2, 1))
    matrices = {phase: np.column_stack(values) for phase, values in arrays.items()}
    np.savez_compressed(output/'bank.npz', **matrices)
    atomic_json(output/'bank-identity.json', {'names': names, 'models': identity,
                'bank_sha256': sha256((output/'bank.npz').read_bytes()).hexdigest(),
                'scope': '既有真实权重逐条预测的缓存；本次不重新训练或运行完整研究控制器'})
    return names, matrices, future, baseline, published


def batches_for(task, meta, total, future):
    if task == 'sgp40':
        if total != 400:
            raise ValueError('冻结SGP40契约为8批各50行')
        cuts = list(range(0, 401, 50))
    else:
        cuts = meta['target_batch_cuts']
    if len(cuts) != 9 or cuts[0] != 0 or cuts[-1] != total or not np.array_equal(future, np.arange(total) >= cuts[4]):
        raise ValueError('批次边界与原发布时点不同')
    return cuts


def score(task, y, values, groups, future):
    positive = y[future] == 2
    return {'complete': metric(task, y, values, groups), 'future': metric(task, y[future], values[future], groups[future]),
            'future_positives': int(positive.sum()), 'future_detected': int(np.sum(positive & (values[future] == 2))),
            'future_false_alarms': int(np.sum(~positive & (values[future] == 2)))}


def run_case(task, args):
    source, output = args.bundle/task, args.output/task
    output.mkdir()
    task_identity = json.loads((source/'task-identity.json').read_bytes())
    for key, prefix in [('task_files', source/'task'), ('private_files', source)]:
        for name, checksum in task_identity[key].items():
            if sha256((prefix/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('原合法输入改变')
    names, bank, future, baseline, original = build_bank(task, args.base, args.ablation, source, output)
    oracle = json.loads((source/'oracle.json').read_bytes())
    meta = json.loads((source/'task/task.json').read_bytes())
    groups = np.asarray(oracle['service_groups'])
    # 可部署的强固定对照仅按探索选案；只确认这一次选择，不根据确认重挑。
    exploration = [metric(task, oracle['explore_y'], bank['explore'][:, i], oracle['explore_groups']) for i in range(len(names))]
    best = max(range(len(names)), key=lambda i: exploration[i]['score'])
    confirmation = metric(task, oracle['confirm_y'], bank['confirm'][:, best], oracle['confirm_groups'])
    parent_explore = metric(task, oracle['explore_y'], bank['explore'][:, 2], oracle['explore_groups'])
    parent_confirm = metric(task, oracle['confirm_y'], bank['confirm'][:, 2], oracle['confirm_groups'])
    chosen_fixed = best if accepted(task, parent_explore, exploration[best]) and accepted(task, parent_confirm, confirmation) else 2
    atomic_json(output/'fixed-selection.json', {'nominated': names[best], 'selected': names[chosen_fixed],
                'exploration': exploration[best], 'confirmation': confirmation})
    # 外层评分器持有答案；控制器只接收已到期前缀计数以及每次封存后释放的当前批标签。
    with np.load(source/'private-answers.npz', allow_pickle=False) as z:
        y, ids = z['y'], z['ids']
    cuts = batches_for(task, meta, len(y), future)
    prefix = confusion(y[:cuts[4]], baseline[:cuts[4]])
    controls = {initial+'/'+mode: CumulativeClassificationControl(mode, len(names), index, prefix, 4)
                for initial, index in [('project', 0), ('comind', 1)] for mode in MODES}
    trajectories = {key: baseline.copy() for key in controls}
    choices = {key: [] for key in controls}
    for batch, (start, end) in enumerate(zip(cuts[4:-1], cuts[5:]), start=5):
        local = slice(start-cuts[4], end-cuts[4])
        current = bank['future'][local]
        decisions = {}
        for key, control in controls.items():
            if int(control.actual.sum()) != start:
                raise ValueError('控制器已见标签数与合法时序不同')
            selected, weights = control.choose(end-start)
            trajectories[key][start:end] = current[:, selected]
            choices[key].append(names[selected])
            decisions[key] = {'arm': names[selected], 'index': selected, 'weights': weights,
                              'feedback_rows_before': start, 'prediction_rows': [start, end]}
        # 同一批所有规则均完成不可改写交付，才把标签交给各自控制器。
        issued = output/f'batch-{batch}-issued.npz'
        np.savez_compressed(issued, ids=ids[start:end], **{k: v[start:end] for k, v in trajectories.items()})
        atomic_json(output/f'batch-{batch}-decisions.json', {'predictions_sha256': sha256(issued.read_bytes()).hexdigest(), 'decisions': decisions})
        for key, control in controls.items():
            observed = control.observe(y[start:end], current)
            state = control.state()
            atomic_json(output/f'batch-{batch}-{key.replace("/", "-")}-state.json', {'feedback': observed, 'state': state})
            # 每批强制重载状态，持续交付的结果同时覆盖接续路径。
            controls[key] = CumulativeClassificationControl.restore(state)
    for initial in ['project', 'comind']:
        if not np.array_equal(trajectories[initial+'/fixed'], original[initial]):
            raise ValueError('固定对照未逐条复现原真实交付')
    trajectories['static_explore'] = baseline.copy()
    trajectories['static_explore'][future] = bank['future'][:, chosen_fixed]
    for key, values in trajectories.items():
        if not np.array_equal(values[~future], baseline[~future]):
            raise ValueError('不可改写的前缀发生变化')
    np.savez_compressed(output/'issued.npz', ids=ids, **trajectories)
    results = {key: {**score(task, y, values, groups, future), 'choices': choices.get(key, [names[chosen_fixed]]),
                     'updates': controls[key].updates if key in controls else 0}
               for key, values in trajectories.items()}
    fixed = {}
    for i, name in enumerate(names):
        values = baseline.copy(); values[future] = bank['future'][:, i]
        fixed[name] = score(task, y, values, groups, future)
    hindsight = max(names, key=lambda name: fixed[name]['complete']['score'])
    result = {'task': task, 'scope': '已见资料上的真实预测缓存回放；不是新完整模型搜索对照',
              'bank_size': len(names), 'future_batches': 4, 'policies': results,
              'fixed_arms': fixed, 'best_fixed_hindsight': hindsight,
              'hindsight_not_used_for_choices': True, 'original_fixed_predictions_exact': True,
              'new_training': 0, 'new_research_model_calls': 0,
              'issued_sha256': sha256((output/'issued.npz').read_bytes()).hexdigest()}
    atomic_json(output/'scores.json', result)
    print(json.dumps({'task': task, 'bank_size': len(names), 'policies': results,
                       'best_fixed_hindsight_score': fixed[hindsight]['complete']['score']}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--ablation', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖既有回放')
    args.output.mkdir(parents=True)
    results = {task: run_case(task, args) for task in TASKS}
    atomic_json(args.output/'scores.json', results)


if __name__ == '__main__':
    main()
