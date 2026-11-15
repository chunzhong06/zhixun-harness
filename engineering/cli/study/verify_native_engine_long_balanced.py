"""在实际父版链上执行无扩容的均衡续训，保存最终权重的完整评价。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.audit_native_run import audit
from engineering.cli.study.evaluate_native_transfer import dataset
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.study.verify_native_balanced_cross_task import predict, scores
from engineering.cli.workflow.native_research import run
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'native-engine-long-balanced'


class Broker:
    """脚本动作和模型服务调用分开记；提名仍须接受共同确认。"""
    def __init__(self, count, calls):
        self.count, self.max_calls, self.calls = count, calls, 0

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('固定工程驱动不搜索')
        self.calls += 1; self.count['calls'] += 1
        if self.calls == 1:
            action = {'action': 'reference', 'name': 'long-balanced-fine-tuning',
                      'hypothesis': '按设备均衡续训12轮，保留原结构并检查确认'}
        elif self.calls == 2:
            action = {'action': 'finish', 'execution': 1, 'candidate': 'candidate-001',
                      'reason': '提交实际训练候选，交共同规则决定'}
        else:
            raise ValueError('固定驱动超出两次动作')
        return {'text': json.dumps(action, ensure_ascii=False)}


async def main():
    counter = {'calls': 0}
    factory = lambda folder, calls: Broker(counter, calls)
    options = {'driver': 'fixture', 'calls': 2, 'stages': 3, 'broker_factory': factory}
    def interrupt(event):
        if event['stage'] == 1:
            raise KeyboardInterrupt('第1轮真实交付后注入中断')
    try:
        await run('engine', OUT / 'runtime', stage_hook=interrupt, **options)
    except KeyboardInterrupt:
        pass
    else:
        raise ValueError('预期中断没有发生')
    first_decision = (OUT / 'runtime/stage-1/decision.json').read_bytes()
    actual = await run('engine', OUT / 'runtime', resume=True, **options)
    if first_decision != (OUT / 'runtime/stage-1/decision.json').read_bytes():
        raise ValueError('恢复改写了首轮采用')
    before = counter['calls']
    restored = await run('engine', OUT / 'runtime', resume=True, **options)
    if before != 6 or counter['calls'] != before or actual != restored or actual['driver_completed_stages'] != 3:
        raise ValueError('三轮执行或完成恢复异常')
    report = audit(OUT / 'runtime')
    parent = json.loads((BASE / 'full-validation-search/engine/selection.json').read_bytes())
    frozen = next(c for c in parent['candidates'] if c['name'] == 'balanced-e12-all')
    first = actual['stages'][0]['selected']['checkpoint']
    with np.load(first, allow_pickle=False) as a, np.load(frozen['checkpoint'], allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('首轮状态没有复现已选定原结构候选')
    final = Path(actual['final_checkpoint'])
    identity = {'checkpoint': str(final), 'sha256': sha256(final.read_bytes()).hexdigest(),
        'preselected_sha256': sha256(Path(frozen['checkpoint']).read_bytes()).hexdigest()}
    atomic_json(OUT / 'final-selection.json', identity)
    with np.load(final, allow_pickle=False) as stored:
        metadata = json.loads(str(stored['metadata']))
    filename, _, reader = PACKAGES['engine']
    package = reader(ROOT / 'data/official/engine' / filename)
    vx = np.asarray([r.values for r in package.validation])
    vp = predict('engine', OUT / 'runtime/release', vx, metadata)
    np.savez_compressed(OUT / 'official-issued.npz', ids=np.asarray([r.sample_id for r in package.validation]), actual=vp)
    x, ids, groups, truth = dataset('engine')
    p = predict('engine', OUT / 'runtime/release', x, metadata)
    np.savez_compressed(OUT / 'external-issued.npz', ids=np.asarray(ids), groups=np.asarray(groups), actual=p)
    full = scores('engine', np.asarray([r.label for r in package.validation]), {'actual': vp},
        [r.unit_id.split(':')[0] for r in package.validation])['actual']
    external = scores('engine', truth(), {'actual': p}, groups)['actual']
    report.update(completed_resume_new_calls=0, first_stage_interruption_recovered=True,
        first_candidate_arrays_identical=True, final_matches_preselected=identity['sha256'] == identity['preselected_sha256'],
        final_official=full, final_external=external, fixture_calls=counter['calls'], actual_model_calls=0,
        issued_sha256={name: sha256((OUT / name).read_bytes()).hexdigest() for name in ('official-issued.npz', 'external-issued.npz')})
    atomic_json(OUT / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
