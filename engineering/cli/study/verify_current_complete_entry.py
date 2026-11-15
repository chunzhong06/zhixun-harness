"""三类完整当前模型接入统一入口，核对保持父版、完成恢复和独立预测。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from engineering.cli.study.audit_native_run import audit
from engineering.cli.workflow.native_research import CURRENT_MODELS, current_parent, run
from harness.runtime.storage import atomic_json

OUT = ROOT / 'results/experiments/contest-continuation/current-complete-entry'


class Preserve:
    def __init__(self, counter): self.calls, self.max_calls, self.counter = 0, 1, counter

    async def exchange(self, kind, payload):
        if kind != 'model': raise ValueError('保持父版只需一次工程动作')
        self.calls += 1; self.counter['calls'] += 1
        return {'text': json.dumps({'action': 'finish', 'execution': 0, 'candidate': 'parent',
                                   'reason': '核对当前完整父版的直接交付，不另行搜索'}, ensure_ascii=False)}


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    result = {}
    for task in ('gas', 'bearing', 'engine'):
        checkpoint = current_parent(task)
        counter = {'calls': 0}
        options = {'driver': 'fixture', 'calls': 1, 'stages': 1, 'parent_checkpoint': checkpoint,
                   'broker_factory': lambda folder, calls: Preserve(counter)}
        first = await run(task, OUT / task, **options)
        second = await run(task, OUT / task, resume=True, **options)
        if first != second or counter['calls'] != 1:
            raise ValueError('完成恢复重跑了动作或改变了结果')
        checked = audit(OUT / task)
        if checked['launched_training_workers'] != 0 or checked['actual_model_calls'] != 0:
            raise ValueError('保持父版不应产生训练或模型服务调用')
        if sha256(Path(first['final_checkpoint']).read_bytes()).hexdigest() != sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('统一入口未保留指定完整父版')
        result[task] = {'parent_checkpoint': str(checkpoint), 'resume_new_calls': 0, **checked}
        print(json.dumps({task: result[task]}, ensure_ascii=False), flush=True)
    atomic_json(OUT / 'verification.json', {'current_selection_sha256': sha256(CURRENT_MODELS.read_bytes()).hexdigest(),
                'tasks': result, 'new_training': 0, 'actual_model_calls': 0,
                'total_current_artifact_bytes': sum(v['release_bytes'] for v in result.values())})


if __name__ == '__main__': asyncio.run(main())
