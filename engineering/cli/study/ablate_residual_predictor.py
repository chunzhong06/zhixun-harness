"""先复现实际发布模型，再拆分固定组件，保留官方与既有跨设备交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.study.audit_publication_effect import replay
from engineering.cli.workflow.official_components import accepted
from harness.data.cmapss_input import read_cmapss, test_answers, REGISTERED_SHA256
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.author_mlestar import ACTOR
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json

FAMILIES = {
    'balanced-tree': {
        'file': 'balanced_tree_residual.py', 'class': 'BalancedTreeResidualRegressor',
        'variants': {'full': {}, 'tree_only': {'local_scale': 0.}, 'local_only': {'global_weight': 0.},
                     'uniform_rows': {'balance_devices': False},
                     'without_parent_distance': {'parent_distance': False}, 'eight_anchors': {'anchors': 8}}},
    'robust-network': {
        'file': 'robust_residual_network.py', 'class': 'RobustResidualNetwork',
        'variants': {'full': {}, 'squared_loss': {'loss': 'squared'},
                     'without_parent_feature': {'include_parent': False}, 'width64': {'hidden_width': 64},
                     'anchors64': {'anchors': 64}, 'learning_rate001': {'learning_rate': .001}}},
}


def run(args):
    bundle, previous, transfer, output = (p.resolve() for p in
                                        (args.bundle, args.previous, args.transfer, args.output))
    output.mkdir(parents=True, exist_ok=False)
    family = FAMILIES[args.family]
    variants = family['variants']
    oracle = json.loads((bundle/'oracle.json').read_bytes())
    identity = json.loads((bundle/'task-identity.json').read_bytes())
    source = (ROOT/'engineering/src/harness/models'/family['file']).read_text(encoding='utf-8')
    old_decision = json.loads((previous/'decision.json').read_bytes())
    assert old_decision['adopted']
    original = Path(old_decision['candidate']['path'])
    assert sha256(original.with_suffix('.pkl').read_bytes()).hexdigest() == old_decision['candidate']['weights_sha256']
    transfer_identity = json.loads((transfer/'predictions-complete.json').read_bytes())
    transfer_input = transfer/'service.csv'
    assert sha256(transfer_input.read_bytes()).hexdigest() == transfer_identity['input_sha256']
    assert transfer_identity['published']['project']['weights_sha256'] == old_decision['candidate']['weights_sha256']
    with np.load(transfer/'issued.npz', allow_pickle=False) as z:
        target_ids, target_subsets = z['ids'], z['subsets']
        target_parent, target_original = z['parent'], z['project']
    with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline, x = z['future'], z['baseline'], z['service_x']
    columns = json.loads((bundle/'task/task.json').read_bytes())['feature_columns']
    service = output/'official-service.csv'
    pd.DataFrame(x[future], columns=columns).to_csv(service, index=False)
    parent_checks = [metric('engine', oracle[key+'_y'], pd.read_csv(bundle/'task'/file).iloc[:, -1], oracle[key+'_groups'])
                     for key, file in [('explore', 'validation.csv'), ('confirm', 'test.csv')]]
    workspace = output/'workspace'
    work = workspace/'component'
    shutil.copytree(bundle/'task', work/'input')
    def measured(prediction):
        values = metric('engine', oracle['explore_y'], prediction, oracle['explore_groups'])
        return {**values, 'loss': values['score']}
    executor = AuthorExecutor(output/'executions', workspace,
        {'validation_y': np.asarray(oracle['explore_y']), 'confirmation_y': np.asarray(oracle['confirm_y'])}, scorer=measured)
    records = {}
    for variant, parameters in variants.items():
        ACTOR.set('residual_component_ablation')
        code = source + '\nfrom harness_task import load_data,evaluate_model\nimport json\nfrom pathlib import Path\n'
        code += 'X,y,V,T=load_data()\ngroups=json.loads(Path("input/task.json").read_text(encoding="utf-8"))["train_groups"]\n'
        code += f'model={family["class"]}(**{parameters!r}).fit(X,y,groups)\nevaluate_model(model,label={variant!r})\n'
        result = executor(code, work, 'candidate.py', 90)
        if result['returncode']:
            raise RuntimeError(variant+': '+result['stderr'][-1500:])
        record = executor.records[-1]
        assert len(record['candidates']) == 1
        candidate = record['candidates'][0]
        path = Path(candidate['path'])
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            checks = [metric('engine', oracle[key+'_y'], z[kind], oracle[key+'_groups'])
                      for key, kind in [('explore', 'validation'), ('confirm', 'confirmation')]]
            if variant == 'full':
                with np.load(original.with_suffix('.npz'), allow_pickle=False) as old:
                    for kind in ('validation', 'confirmation'):
                        np.testing.assert_allclose(z[kind], old[kind], rtol=0, atol=1e-8)
        adopted = all(accepted('engine', a, b) for a, b in zip(parent_checks, checks, strict=True))
        folder = output/variant
        official_prediction = replay(path.with_suffix('.pkl'), service, folder/'official', timeout=240)
        target_prediction = replay(path.with_suffix('.pkl'), transfer_input, folder/'transfer', timeout=240)
        if variant == 'full':
            np.testing.assert_allclose(official_prediction, np.load(previous/'serving/prediction.npy'), rtol=0, atol=1e-8)
            np.testing.assert_allclose(target_prediction, target_original, rtol=0, atol=1e-8)
        direct = baseline.copy()
        direct[future] = official_prediction
        actual = direct if adopted else baseline.copy()
        np.savez_compressed(folder/'issued.npz', official_ids=np.asarray(oracle['service_ids']),
            official_direct=direct, official_actual=actual, target_ids=target_ids, target_subsets=target_subsets,
            target_direct=target_prediction, target_actual=target_prediction if adopted else target_parent)
        records[variant] = {'parameters': parameters, 'adopted': adopted, 'checks': checks,
                            'weights_sha256': candidate['weights_sha256'], 'source_sha256': record['source_sha256'],
                            'fit_and_checks_seconds': record['execution_time']}
        atomic_json(folder/'decision.json', records[variant])
    # 所有预先列出的变体先完成交付；评分不改变变体、权重或原采用规则。
    atomic_json(output/'predictions-complete.json', {'family': args.family, 'variants': list(variants), 'all_predictions_sealed': True,
        'original_official_and_transfer_reproduced': True, 'tolerance': 1e-8, 'source_model': str(previous),
        'reference_transfer_input_sha256': transfer_identity['input_sha256']})
    from engineering.cli.workflow import public_maintenance as public
    package = public.PACKAGES['engine'][2](Path(identity['archive']))
    answers = {r.sample_id: r.label for r in package.validation}
    official_truth = np.asarray([answers[k] for k in oracle['service_ids']])
    archive = ROOT/'data/public/cmapss/CMAPSSData.zip'
    assert sha256(archive.read_bytes()).hexdigest() == REGISTERED_SHA256
    target_answers = {}
    for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
        target_answers.update(test_answers(archive, subset, read_cmapss(archive, subset=subset, partition='test')))
    target_truth = np.asarray([target_answers[key] for key in target_ids])
    for variant, report in records.items():
        with np.load(output/variant/'issued.npz', allow_pickle=False) as z:
            report['official'] = {kind: metric('engine', official_truth, z['official_'+kind], oracle['service_groups'])
                                  for kind in ('direct', 'actual')}
            report['transfer'] = {kind: {'pooled_rmse': float(np.sqrt(np.mean((z['target_'+kind]-target_truth)**2))),
                'by_subset': {subset: float(np.sqrt(np.mean((z['target_'+kind][target_subsets == subset]-target_truth[target_subsets == subset])**2)))
                              for subset in np.unique(target_subsets)}} for kind in ('direct', 'actual')}
        atomic_json(output/variant/'scores.json', report)
        print(json.dumps({'variant': variant, 'adopted': report['adopted'],
                          'official_rmse': report['official']['actual']['score'],
                          'transfer_rmse': report['transfer']['actual']['pooled_rmse']}, ensure_ascii=False), flush=True)
    atomic_json(output/'scores.json', {'scope': '已见官方及NASA资料上的组件开发消融，不是新完整驱动对照',
        'family': args.family, 'results': records, 'new_training': len(variants), 'new_model_calls': 0,
        'default_model_reproduced': True, 'no_target_fitting_or_variant_selection': True})


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--family', choices=FAMILIES, required=True)
    for name in ('bundle', 'previous', 'transfer', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
