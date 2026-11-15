"""一次最小正式推理诊断；只保留脱敏错误或公开回答，不自动重试。"""
from pathlib import Path
from argparse import ArgumentParser
import json
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]

from engineering.cli.workflow.native_research import credential
from harness.integrations.pytrio_planner import ChatRequest, PyTrioConfig
from harness.integrations.service_transport import HttpChatTransport
from harness.runtime.storage import atomic_json


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/engineering/contest-continuation/pytrio-inference-diagnostic.json')
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--raw-completion', action='store_true')
    args = parser.parse_args()
    # 旧失败回执也属于实验事实；复测必须写到新位置。
    if args.output.exists():
        raise FileExistsError('输出已存在，请指定本次诊断的新路径')
    config = PyTrioConfig(model='Qwen/Qwen3.8-27B', max_tokens=50, timeout_seconds=90,raw_completion=args.raw_completion)
    transport = HttpChatTransport(config, credential(), allow_network=True)
    request = {'model': config.model, 'messages': [{'role': 'user', 'content': '仅回答：连接成功'}],
               'max_tokens': 50, 'temperature': 0}
    if args.no_thinking:
        request['chat_template_kwargs'] = {'enable_thinking': False}
    if args.raw_completion:
        from harness.integrations.qwen_text_template import render
        request['prompt']=render(request.pop('messages'))
        request['stop']=['<|im_end|>','<|endoftext|>']
    started = perf_counter()
    response = transport.send(ChatRequest(transport.endpoint, request, config.timeout_seconds))
    report = {'model': config.model, 'http_status': response.status, 'diagnostic': response.body,
              'request': request, 'automatic_retries': 0, 'elapsed_seconds': perf_counter() - started}
    usage = response.body.get('usage', {})
    if type(usage.get('prompt_tokens')) is int and type(usage.get('completion_tokens')) is int:
        # 按公开单价全价估计，缓存也按全价计；实际扣费以服务账单为准。
        report['estimated_cost_cny'] = (usage['prompt_tokens'] * 3.15 + usage['completion_tokens'] * 25.29) / 1_000_000
    atomic_json(args.output, report)
    print(json.dumps({'http_status': response.status, 'elapsed_seconds': report['elapsed_seconds'],
                     'usage': usage, 'estimated_cost_cny': report.get('estimated_cost_cny'),
                     'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
