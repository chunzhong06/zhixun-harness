"""用真实已完成轨迹验证请求中的试验台账；仅重放动作，不调用模型或训练。"""
import asyncio
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.free_research_driver import run_free_research
from harness.runtime.storage import atomic_json


async def verify(run, output):
    records = json.loads((run/'executions/index.json').read_bytes())
    actions = json.loads((run/'workspace/project-decisions.json').read_bytes())
    assert actions[-1]['action'] == 'finish'
    assert all(a['action'] in ('experiment', 'finish') for a in actions)

    class ExecutorReplay:
        def __init__(self):
            self.records = []

        def __call__(self, source, *_):
            record = records[len(self.records)]
            assert sha256(source.encode()).hexdigest() == record['source_sha256']
            self.records.append(record)
            return {k: record[k] for k in ('returncode', 'stdout', 'stderr', 'execution_time')}

    class BrokerReplay:
        max_calls = 48
        prior_model_invocations = 0
        reused_model_calls = 0

        def __init__(self):
            self.calls = 0
            self.contexts = []

        async def exchange(self, kind, payload):
            assert kind == 'model'
            self.contexts.append(json.loads(payload['request']['contents'][0]['parts'][0]['text']))
            action = actions[self.calls]
            self.calls += 1
            return {'text': json.dumps(action, ensure_ascii=False)}

    executor, broker = ExecutorReplay(), BrokerReplay()
    # 临时目录只供真实入口复制空任务及写重放决策，结束立即清理。
    parent = ROOT/'results/engineering/research-workflow'
    with TemporaryDirectory(prefix='ledger-check-', dir=parent) as temporary:
        temporary = Path(temporary).resolve()
        assert temporary.is_relative_to(parent.resolve())
        task = temporary/'task'
        task.mkdir()
        selected = await run_free_research((run/'task_description.txt').read_text(encoding='utf-8'),
                                          task, temporary/'workspace', executor, broker)
    context = broker.contexts[-1]
    history = context['history']
    experiments = [a for a in actions if a['action'] == 'experiment']
    assert len(history) == len(records) == len(experiments)
    assert [h['hypotheses'][0] for h in history] == [a['hypothesis'] for a in experiments]
    assert [h['source_sha256'] for h in history] == [r['source_sha256'] for r in records]
    assert selected['sequence'] == actions[-1]['execution']
    before_repeat = broker.contexts[13]['history']
    assert before_repeat[6]['hypotheses'] == [experiments[6]['hypothesis']]
    for h in history:
        assert set(h) <= {'execution', 'hypotheses', 'source_sha256', 'returncode', 'seconds', 'candidates', 'same_source_as'}
        for c in h['candidates']:
            assert set(c) == {'candidate_id', 'label', 'exploration_errors', 'score'}
    report = {'actual_request_path_verified': True, 'replayed_model_actions': len(actions),
              'new_model_calls': 0, 'new_training': 0, 'all_executed_hypotheses_preserved': True,
              'hypothesis_available_before_observed_repeat': True, 'no_new_evaluation_information': True,
              'temporary_workspace_removed': not temporary.exists(),
              'driver_sha256': sha256((ROOT/'engineering/src/harness/integrations/free_research_driver.py').read_bytes()).hexdigest(),
              'ledger_sha256': sha256((ROOT/'engineering/src/harness/integrations/research_ledger.py').read_bytes()).hexdigest(),
              'scope': '真实驱动构造请求的工程回归；重放旧动作不验证新模型行为或性能收益'}
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    asyncio.run(verify(args.run.resolve(), args.output.resolve()))
