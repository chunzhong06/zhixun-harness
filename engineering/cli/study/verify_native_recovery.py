"""用明确标记的工程驱动注入中断，核验三任务真实训练与完整发布恢复。"""
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch
import asyncio
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]

from engineering.cli.workflow.native_research import run
from engineering.cli.study.audit_native_run import audit
from harness.integrations.native_research import NativeExecutor
from harness.runtime.storage import atomic_json


class EngineeringBroker:
    """脚本夹具只选官方一轮再提名，验证恢复而不报告模型自主能力。"""
    def __init__(self, calls, counter):
        self.max_calls, self.counter, self.calls = calls, counter, 0

    async def exchange(self, kind, payload):
        self.counter['model_requests'] += 1
        self.calls += 1
        context = json.loads(payload['request']['contents'][0]['parts'][0]['text'])
        if len(context['history']) <= 1:
            action = {'action': 'reference', 'name': 'official-one-epoch', 'hypothesis': '工程夹具核查父版真实一轮续训'}
        else:
            action = {'action': 'finish', 'execution': 1, 'candidate': 'candidate-001', 'reason': '工程夹具提名已保存候选供外层确认'}
        return {'text': json.dumps(action, ensure_ascii=False)}


def verify(task, folder):
    counter = {'model_requests': 0}
    factory = lambda directory, calls: EngineeringBroker(calls, counter)
    original = NativeExecutor.__call__
    interrupted = []
    def after_completed_trial(executor, *args, **kwargs):
        result = original(executor, *args, **kwargs)
        interrupted.append('已保存训练后、外层决策前')
        raise KeyboardInterrupt('工程注入：训练完成后中断')
    options = {'driver': 'fixture', 'calls': 2, 'stages': 2, 'broker_factory': factory}
    with patch.object(NativeExecutor, '__call__', after_completed_trial):
        try:
            asyncio.run(run(task, folder, **options))
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('未触发训练后中断')
    assert counter['model_requests'] == 1
    first_program = folder / 'stage-1/executions/0001/execution.json'
    original_receipt = first_program.read_bytes()
    def after_completed_stage(event):
        interrupted.append('已交付阶段后、后续阶段前')
        raise KeyboardInterrupt('工程注入：阶段交付后中断')
    try:
        asyncio.run(run(task, folder, resume=True, stage_hook=after_completed_stage, **options))
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('未触发阶段后中断')
    assert counter['model_requests'] == 2
    assert first_program.read_bytes() == original_receipt
    report = asyncio.run(run(task, folder, resume=True, **options))
    assert counter['model_requests'] == 4
    before = (folder / 'run.json').read_bytes()
    repeated = asyncio.run(run(task, folder, resume=True, **options))
    assert repeated == report and counter['model_requests'] == 4
    assert (folder / 'run.json').read_bytes() == before
    # 模拟发布写完部分文件但最终完成标记尚未落盘：只补回相同来源的缺件。
    (folder / 'completion.json').unlink()
    prediction_code = folder / 'release/baselines/predict.py'
    expected_code = prediction_code.read_bytes()
    prediction_code.unlink()
    recovered_publication = asyncio.run(run(task, folder, resume=True, **options))
    assert recovered_publication == report and counter['model_requests'] == 4
    assert prediction_code.read_bytes() == expected_code
    result = audit(folder)
    result.update(scope='真实官方全训练资料上的工程恢复夹具，不是Luna性能实验',
                  injected_interruptions=interrupted, actual_fixture_calls=counter['model_requests'],
                  completed_trial_reused=True, completed_stage_reused=True,
                  completed_run_reused=True, original_execution_receipt_unchanged=True)
    result['partial_publication_recovered_without_new_training'] = True
    atomic_json(folder / 'recovery-verification.json', result)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-pilot', type=Path, action='append', default=[])
    args = parser.parse_args()
    results = [verify(task, (args.output / task).resolve()) for task in ('gas', 'bearing', 'engine')]
    pilot = [audit(folder) for folder in args.audit_pilot]
    # 模拟平台传入实际上一轮父版，从该版重新开始一项明确的新运行。
    parent = args.output.resolve() / 'gas/release/checkpoint.npz'
    external_counter = {'model_requests': 0}
    external = args.output.resolve() / 'external-gas-parent'
    asyncio.run(run('gas', external, driver='fixture', calls=2, stages=1,
                    parent_checkpoint=parent,
                    broker_factory=lambda directory, calls: EngineeringBroker(calls, external_counter)))
    external_result = audit(external)
    from hashlib import sha256
    assert external_result['parent_chain'][0]['parent_sha256'] == sha256(parent.read_bytes()).hexdigest()
    verified = {'passed': True, 'cases': results, 'luna_pilot_audit': pilot,
                'external_selected_parent_verified': external_result}
    atomic_json(args.output / 'verification.json', verified)
    print(json.dumps(verified, ensure_ascii=False))


if __name__ == '__main__':
    main()
