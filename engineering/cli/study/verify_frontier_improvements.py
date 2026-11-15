"""走真实入口核验作者发布修复和新组件登记，不用新搜索替代回归验证。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from engineering.cli.study.audit_publication_effect import cases
from engineering.cli.workflow import london_field_driver as london
from engineering.cli.workflow.london_complete import masks
from engineering.cli.workflow.mlestar_author import finish_delivery
from harness.integrations.recipe_regressor import code_inputs
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.frontier_task import sources
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json


class RejectRetraining:
    records = []

    def __call__(self, *args, **kwargs):
        raise AssertionError('作者原生发布不得调用补训执行器')


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    london.configure('2026-07-01')
    archive = ROOT/'data/public/london-air-2026/stations.zip'
    features, names = code_inputs(london.prepare(archive, True, True))
    report = {'author_publication': {}}
    for site, arm, work, before, after in cases(args.baseline):
        if arm != 'comind':
            continue
        truth = london.readings(archive, site).to_numpy()
        old, fit0, win0, service, returned = london.roles(truth)
        start = int(service[0])
        at = start+28*24
        labels = np.r_[old[:start], returned]
        _, windows, history = masks(labels, at)
        with np.load(work/'issued.npz', allow_pickle=False) as z:
            parent = z['baseline']
        decision = json.loads((work/'decision.json').read_bytes())
        records = json.loads((work/'executions/index.json').read_bytes())
        selected = dict(next(r for r in records if r['sequence'] == decision['selected_execution']))
        selected['candidates'] = [before]
        destination = output/site
        destination.mkdir()
        local_args = SimpleNamespace(arm='comind', site=site, start='2026-07-01', frontier_task=True)
        broker = SimpleNamespace(calls=0, search_calls=0, reused_model_calls=0, reused_search_calls=0,
                                 prior_model_invocations=0)
        finish_delivery(local_args, destination, features, names, labels, history, service, windows,
            at, parent, 'London_'+site, work/'tasks'/('London_'+site), RejectRetraining(),
            destination/'workspace', broker, selected, truth,
            summary_extra={'scope': '原选案的真实发布入口回归，无新搜索或训练'})
        with np.load(destination/'issued.npz', allow_pickle=False) as z, np.load(args.audit/site/'comind/issued.npz', allow_pickle=False) as old:
            np.testing.assert_array_equal(z['ids'], old['ids'])
            np.testing.assert_allclose(z['comind'], old['accepted_weights'], rtol=0, atol=1e-9)
        publication = json.loads((destination/'publication.json').read_bytes())
        assert publication['rule'] == 'retain_author_selected_weights'
        assert publication['published_weights_sha256'] == before['weights_sha256']
        report['author_publication'][site] = {'no_refit': True, 'predictions_match': True, **publication}
    # 新组件必须通过实际受限执行器及序列化，而非只在研究脚本内可用。
    bundle = args.bundle/'engine'
    oracle = json.loads((bundle/'oracle.json').read_bytes())
    workspace = output/'engine/workspace'
    inputs = workspace/'task/input'
    inputs.mkdir(parents=True)
    import shutil
    for p in (bundle/'task').iterdir():
        shutil.copy2(p, inputs/p.name)
    def scorer(prediction):
        values = metric('engine', oracle['explore_y'], prediction, oracle['explore_groups'])
        return {**values, 'loss': values['score']}
    executor = AuthorExecutor(output/'engine/executions', workspace,
        {'validation_y': np.asarray(oracle['explore_y']), 'confirmation_y': np.asarray(oracle['confirm_y'])}, scorer=scorer)
    source = sources('engine')['balanced-device-hybrid-residual']
    result = executor(source, workspace/'task', 'new_component.py', 90)
    if result['returncode']:
        raise RuntimeError('新登记组件未通过真实执行器：'+str(result)[-1600:])
    candidate = executor.records[-1]['candidates'][-1]
    original = json.loads((args.baseline/'engine/comind-recovered/decision.json').read_bytes())['candidate']
    with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z, np.load(Path(original['path']).with_suffix('.npz'), allow_pickle=False) as old:
        for name in ('validation', 'confirmation'):
            np.testing.assert_allclose(z[name], old[name], rtol=0, atol=1e-5)
    report['registered_component'] = {'real_executor_passed': True, 'author_predictions_reproduced': True,
                                      'source_sha256': sha256(source.encode()).hexdigest(),
                                      'weights_sha256': candidate['weights_sha256']}
    report.update(new_training=1, new_model_calls=0)
    atomic_json(output/'verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    for name in ('baseline', 'audit', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
