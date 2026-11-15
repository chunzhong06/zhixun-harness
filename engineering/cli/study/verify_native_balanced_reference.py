"""把已选定的均衡程序接入共同工具；记录连续接续，不代替真实驱动验收。"""
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.audit_native_run import audit
from engineering.cli.workflow.native_research import run
from harness.runtime.storage import atomic_json

OUT = ROOT / 'results/experiments/contest-continuation/native-balanced-reference'
RECIPE = 'extended-balanced-fine-tuning'


class Broker:
    """固定动作只用于验证工具的执行结果，调用数与真实模型服务分开记录。"""
    def __init__(self, count, calls):
        self.count, self.max_calls, self.calls = count, calls, 0

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('本夹具不使用搜索')
        self.calls += 1
        self.count['calls'] += 1
        if self.calls == 1:
            action = {'action': 'reference', 'name': RECIPE,
                      'hypothesis': '复用较小容量和较长均衡续训，检验实际父版接续'}
        elif self.calls == 2:
            action = {'action': 'finish', 'execution': 1, 'candidate': 'candidate-001',
                      'reason': '固定工程夹具提交已实际训练候选，交共同确认'}
        else:
            raise ValueError('超出固定动作数')
        return {'text': json.dumps(action, ensure_ascii=False)}


async def main():
    counter = {'calls': 0}
    factory = lambda folder, calls: Broker(counter, calls)
    options = {'driver': 'fixture', 'calls': 2, 'stages': 3, 'broker_factory': factory}

    def interrupt(event):
        if event['stage'] == 1:
            raise KeyboardInterrupt('第1轮交付完成后注入中断')

    try:
        await run('bearing', OUT, stage_hook=interrupt, **options)
    except KeyboardInterrupt:
        pass
    else:
        raise ValueError('预定中断未发生')
    preserved = (OUT / 'stage-1/decision.json').read_bytes()
    report = await run('bearing', OUT, resume=True, **options)
    if preserved != (OUT / 'stage-1/decision.json').read_bytes():
        raise ValueError('恢复改变第一轮结果')
    before = counter['calls']
    if await run('bearing', OUT, resume=True, **options) != report or before != counter['calls']:
        raise ValueError('完成恢复重复执行或改变结果')
    if before != 6 or report['driver_completed_stages'] != 3:
        raise ValueError('共同驱动没有完成三轮')
    selection = json.loads((ROOT / 'results/experiments/contest-continuation/bearing-adoption-audit/selection.json').read_bytes())
    first = report['stages'][0]['selected']['checkpoint']
    with np.load(first, allow_pickle=False) as actual, np.load(selection['candidate']['checkpoint'], allow_pickle=False) as expected:
        if set(actual.files) != set(expected.files) or any(not np.array_equal(actual[k], expected[k]) for k in expected.files):
            raise ValueError('参考工具第一轮没有重建已选定全部状态')
    verified = audit(OUT)
    verified.update(reference=RECIPE, fixture_calls=counter['calls'], actual_model_calls=0,
        completed_resume_new_calls=0, first_stage_interruption_recovered=True,
        first_stage_matches_development_candidate=True,
        accepted_stages=[s['stage'] for s in report['stages'] if s['accepted']])
    atomic_json(OUT / 'verification.json', verified)
    print(json.dumps(verified, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
