"""真实三类十轮配对：来源明确的固定程序、父版接续、复用及中断恢复。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.workflow.native_research import run
from engineering.cli.study.audit_native_run import audit
from harness.integrations.native_research import NativeExecutor, measure
from harness.models.continuation_recipes import reference_programs
from harness.runtime.storage import atomic_json

DEST = ROOT / 'results/experiments/contest-continuation/native-ten-stages'
RECIPES = {'gas': 'balanced-fine-tuning', 'bearing': 'expanded-balanced-fine-tuning', 'engine': 'stable-fine-tuning'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class FixedRecipeBroker:
    """每轮按冻结程序提议再提交一案；仅用于实测工程，不代表语言模型行为。"""
    def __init__(self, task, calls, count):
        self.task, self.max_calls, self.count, self.calls = task, calls, count, 0

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('本工程夹具不搜索')
        self.calls += 1
        self.count['calls'] += 1
        if self.calls == 1:
            action = {'action': 'reference', 'name': RECIPES[self.task], 'hypothesis': '固定续训程序检验真实父版接续和采用'}
        elif self.calls == 2:
            action = {'action': 'finish', 'execution': 1, 'candidate': 'candidate-001', 'reason': '冻结工程夹具提名已执行候选'}
        else:
            raise ValueError('固定夹具超出两步')
        return {'text': json.dumps(action, ensure_ascii=False)}


async def one(task, cached):
    output = DEST / task / ('cached' if cached else 'uncached')
    counter = {'calls': 0}
    factory = lambda directory, calls: FixedRecipeBroker(task, calls, counter)
    options = {'driver': 'fixture', 'calls': 2, 'stages': 10, 'broker_factory': factory, 'candidate_cache': cached}
    if cached:
        def interrupt(event):
            if event['stage'] == 4:
                raise KeyboardInterrupt('十轮工程核验：第4轮交付后中断')
        try:
            await run(task, output, stage_hook=interrupt, **options)
        except KeyboardInterrupt:
            pass
        else:
            raise ValueError('第4轮中断未发生')
        if counter['calls'] != 8:
            raise ValueError('中断前调用计数不符')
        original = (output / 'stage-1/decision.json').read_bytes()
        report = await run(task, output, resume=True, **options)
        if original != (output / 'stage-1/decision.json').read_bytes():
            raise ValueError('恢复改写了已采用决定')
    else:
        report = await run(task, output, **options)
    before = counter['calls']
    restored = await run(task, output, resume=True, **options)
    if report != restored or counter['calls'] != before or before != 20:
        raise ValueError('完成恢复重复调用或十轮调用数不正确')
    verified = audit(output)
    copied_training = list(output.glob('stage-*/workspace/project/input/train.npz'))
    if copied_training:
        raise ValueError('原生训练资料仍被重复复制进每轮工作区')
    verified.update(fixture_calls=counter['calls'], actual_model_calls=0,
                    completed_resume_new_calls=0, unused_training_copies=0,
                    fourth_stage_interruption_recovered=cached)
    return verified


def compare(task, before, after):
    parent = DEST / task
    left, right = read(parent / 'uncached/run.json'), read(parent / 'cached/run.json')
    if [r['accepted'] for r in left['stages']] != [r['accepted'] for r in right['stages']]:
        raise ValueError('复用改变采用决定')
    with np.load(parent / 'uncached/issued.npz', allow_pickle=False) as a, np.load(parent / 'cached/issued.npz', allow_pickle=False) as b:
        delta = float(np.max(np.abs(a['actual']-b['actual'])))
    if delta > 1e-12 or left['scores'] != right['scores']:
        raise ValueError('复用改变完整交付')
    with np.load(left['final_checkpoint'], allow_pickle=False) as a, np.load(right['final_checkpoint'], allow_pickle=False) as b:
        if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('最终发布权重或元信息不等价')
    return {'task': task, 'stages': 10, 'uncached_training_workers': before['launched_training_workers'],
            'cached_training_workers': after['launched_training_workers'], 'reused_candidates': after['reused_candidates'],
            'accepted_stages': [r['stage'] for r in right['stages'] if r['accepted']],
            'prediction_max_absolute_difference': delta, 'final_checkpoint_arrays_equal': True,
            'scores': right['scores'], 'release_bytes': after['release_bytes'],
            'cached_completed_resume_new_calls': 0}


def verify_cache_boundaries():
    output = DEST / 'gas/cached'
    task_dir = output / 'task'
    oracle = read(output / 'oracle.json')
    program = reference_programs()[RECIPES['gas']]
    with TemporaryDirectory(prefix='cache-boundary-', dir=DEST) as tmp:
        root = Path(tmp)
        executor = NativeExecutor(root / 'executions', task_dir, task_dir / 'initial.npz', oracle,
                                  max_epochs=12, cache_dir=root / 'cache', cacheable_sources=reference_programs().values())
        if executor._cache_key(program+'\n# 自由程序的独立试验\n') is not None:
            raise ValueError('未经登记的自由源码被自动复用')
        key = executor._cache_key(program)
        pointer = read(output / 'candidate-cache' / (key+'.json'))
        pointer['execution_sha256'] = '0'*64
        atomic_json(root / 'cache' / (key+'.json'), pointer)
        try:
            executor(program, root, 'ignored.py', 90)
        except ValueError as exc:
            if '来源回执改变' not in str(exc):
                raise
        else:
            raise ValueError('损坏的缓存来源被接受')
    # 稀有类别缺失的单轮也固定六类等权。
    probabilities = np.zeros((2, 6)); probabilities[:, 0] = 1.
    if abs(measure('gas', [1, 1], probabilities)['score']-1/6) > 1e-12:
        raise ValueError('气体分轮评分未固定六类')
    return {'unregistered_program_not_cached': True, 'corrupt_origin_rejected': True,
            'gas_six_class_metric_verified': True}


async def main():
    DEST.mkdir(parents=True, exist_ok=False)
    paths = ['engineering/cli/study/verify_native_ten_stages.py', 'engineering/cli/workflow/native_research.py',
             'engineering/cli/workflow/native_continuation.py', 'engineering/cli/study/audit_native_run.py',
             'engineering/src/harness/integrations/native_research.py', 'engineering/src/harness/integrations/native_trace.py',
             'engineering/src/harness/integrations/research_recovery.py', 'engineering/src/harness/integrations/free_research_driver.py',
             'engineering/src/harness/models/native_continuation.py', 'engineering/src/harness/models/continuation_recipes.py',
             'engineering/src/harness/integrations/research_tracking.py', 'engineering/cli/study/run_tracked_experiment.py',
             'engineering/configs/experiments/native-ten-stage-recording.json']
    for name in paths:
        target = DEST / 'frozen-code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    results = []
    for task in ('gas', 'bearing', 'engine'):
        uncached = await one(task, False)
        cached = await one(task, True)
        results.append(compare(task, uncached, cached))
    boundary = verify_cache_boundaries()
    summary = {'passed': True, 'tasks': results, 'boundary_checks': boundary,
               'actual_model_calls': 0, 'fixture_calls': 120,
               'expected_unique_cloud_stages': 60, 'scope': '三类十轮工程配对，固定程序实测不替代正式模型验收'}
    atomic_json(DEST / 'verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
