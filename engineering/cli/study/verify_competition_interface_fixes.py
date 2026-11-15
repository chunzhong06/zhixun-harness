"""验证完整模型两种独立预测入口及双方驱动的正式服务消息适配，不访问服务。"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from engineering.cli.workflow.native_research import current_parent
from harness.integrations.native_research import export_release
from harness.integrations.pytrio_planner import ChatResponse, PyTrioConfig
from harness.integrations.service_research_broker import ServiceResearchBroker
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'competition-interface-fixes'


class RecordingTransport:
    def __init__(self): self.requests = []

    def send(self, request):
        self.requests.append(request)
        return ChatResponse(200, {'choices': [{'message': {'content': '{"action":"finish","execution":0,"candidate":"parent"}'},
                                              'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                                  'model': 'offline-fixture', 'id': 'offline-format-check'})


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'prediction': {}, 'actual_network_requests': 0, 'new_training': 0}
    for task in ('gas', 'bearing', 'engine'):
        task_dir = BASE / 'current-complete-entry' / task / 'task'
        spec = json.loads((task_dir / 'task.json').read_bytes())
        checkpoint = current_parent(task)
        release = OUT / task
        export_release(task_dir, checkpoint, release)
        with ZipFile(spec['archive']) as package:
            rows = [json.loads(line) for line in package.read('public/validation.jsonl').splitlines()[:12]]
        requests = '\n'.join(json.dumps({'inputs': row['inputs']}) for row in rows)
        env = os.environ.copy();env.pop('PYTHONPATH', None);env['PYTHONDONTWRITEBYTECODE'] = '1'
        outputs = {}
        for name, command in {'module': ['-m', 'baselines.predict'], 'script': ['baselines/predict.py']}.items():
            result = subprocess.run([sys.executable, '-B', *command, '--checkpoint', 'checkpoint.npz'],
                                    cwd=release, env=env, input=requests, capture_output=True,
                                    text=True, encoding='utf-8', timeout=30)
            (release / (name + '-stderr.log')).write_text(result.stderr, encoding='utf-8')
            if result.returncode: raise ValueError(task + '/' + name + ': ' + result.stderr)
            outputs[name] = [json.loads(line) for line in result.stdout.splitlines()]
        if len(outputs['module']) != 12 or outputs['module'] != outputs['script']:
            raise ValueError('两个独立入口的完整预测不同')
        report['prediction'][task] = {'rows': 12, 'both_forms_passed': True,
            'same_prediction': True, 'checkpoint_unchanged': checkpoint.read_bytes() == (release / 'checkpoint.npz').read_bytes(),
            'source_checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest()}
    formats = {
        'project': {'request': {'config': {'system_instruction': '系统'}, 'contents': [
            {'role': 'user', 'parts': [{'text': '任务'}]}]}},
        'author': {'request': {'messages': [{'role': 'system', 'content': '系统'}, {'role': 'user', 'content': '任务'}]}}}
    actual = []
    for name, payload in formats.items():
        transport = RecordingTransport()
        broker = ServiceResearchBroker(OUT / ('service-' + name), PyTrioConfig(model='Qwen/Qwen3.8-27B', max_tokens=4096), transport, max_calls=1)
        await broker.exchange('model', payload)
        if len(transport.requests) != 1 or broker.usage[0]['total_tokens'] != 120:
            raise ValueError('离线服务消息或用量记录缺失')
        actual.append(transport.requests[0].body)
    if actual[0] != actual[1]: raise ValueError('项目与作者框架未使用相同服务请求')
    report['service_message_equivalence'] = True
    report['usage_fixture_only'] = True
    atomic_json(OUT / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': asyncio.run(main())
