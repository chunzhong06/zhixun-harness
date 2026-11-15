"""固定实际搜索轨迹，检验更晚停止是否改善完整交付。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.study.audit_publication_effect import replay
from engineering.cli.workflow import public_maintenance as public
from engineering.cli.workflow.official_components import accepted
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json


def run(args):
    # 重放子进程会切换工作目录，输入与输出路径必须先在调用点解析。
    args.run, args.bundle, args.output = args.run.resolve(), args.bundle.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.run/'executions/index.json').read_bytes())
    oracle = json.loads((args.bundle/'oracle.json').read_bytes())
    identity = json.loads((args.bundle/'task-identity.json').read_bytes())
    parent_checks = [metric('engine', oracle[key+'_y'], pd.read_csv(args.bundle/'task'/file).iloc[:, -1], oracle[key+'_groups'])
                     for key, file in [('explore', 'validation.csv'), ('confirm', 'test.csv')]]
    with np.load(args.bundle/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline = z['future'], z['baseline']
    decisions, predictions = {}, {}
    for prefix in (1, 5, 10, 17, 26):
        valid = [r for r in records[:prefix] if r['returncode'] == 0 and r['candidates']]
        selected = min(valid, key=lambda r: r['candidates'][-1]['score'])
        candidate = selected['candidates'][-1]
        model = Path(candidate['path']).with_suffix('.pkl')
        assert sha256(model.read_bytes()).hexdigest() == candidate['weights_sha256']
        with np.load(model.with_suffix('.npz'), allow_pickle=False) as z:
            checks = [metric('engine', oracle[key+'_y'], z[kind], oracle[key+'_groups'])
                      for key, kind in [('explore', 'validation'), ('confirm', 'confirmation')]]
        adopted = all(accepted('engine', old, new) for old, new in zip(parent_checks, checks, strict=True))
        key = 'prefix_'+str(prefix)
        decisions[key] = {'executions_available': prefix, 'selected_execution': selected['sequence'],
                          'adopted': adopted, 'checks': checks, 'weights_sha256': candidate['weights_sha256']}
        # 选择与采用只用合法过去；此时尚未读取服务答案。
        atomic_json(args.output/(key+'-decision.json'), decisions[key])
        predicted = replay(model, args.run/'serving/service.csv', args.output/key)
        direct = baseline.copy()
        direct[future] = predicted
        actual = direct if adopted else baseline.copy()
        predictions[key] = actual
        np.savez_compressed(args.output/(key+'-issued.npz'), ids=np.asarray(oracle['service_ids']), actual=actual, direct=direct)
    with np.load(args.run/'issued.npz', allow_pickle=False) as z:
        np.testing.assert_allclose(predictions['prefix_26'], z['actual'], rtol=0, atol=1e-9)
    package = public.PACKAGES['engine'][2](Path(identity['archive']))
    answers = {r.sample_id: r.label for r in package.validation}
    truth = np.asarray([answers[k] for k in oracle['service_ids']])
    groups = np.asarray(oracle['service_groups'])
    for key, values in predictions.items():
        decisions[key]['complete_rmse'] = metric('engine', truth, values, groups)['score']
        decisions[key]['after_maintenance_rmse'] = metric('engine', truth[future], values[future], groups[future])['score']
    report = {'scope': '已见搜索轨迹的停止位置诊断，非新停止策略的独立验证',
              'prefix_policy': '在已执行前缀中取实际探索损失最低的最后候选，再沿原确认规则交付',
              'final_delivery_reproduced': True, 'new_training': 0, 'new_model_calls': 0, 'results': decisions}
    atomic_json(args.output/'scores.json', report)
    print(json.dumps({k: {name: row[name] for name in ('selected_execution', 'adopted', 'complete_rmse', 'after_maintenance_rmse')}
                      for k, row in decisions.items()}, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    for name in ('run', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
