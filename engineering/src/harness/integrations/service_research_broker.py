"""把同一源码研究请求发送给正式兼容服务，记录调用与用量，不保存认证信息。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path

from harness.integrations.pytrio_planner import ChatRequest
from harness.integrations.research_broker import ACTOR
from harness.runtime.storage import atomic_json


class ServiceResearchBroker:
    def __init__(self, folder, config, transport, *, max_calls=4):
        self.folder = Path(folder)
        self.config, self.transport, self.max_calls = config, transport, max_calls
        self.timeout = config.timeout_seconds
        self.calls = self.search_calls = self.prior_model_invocations = self.reused_model_calls = 0
        self.usage = []

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('正式续训阶段仅使用已提供源码，不启用外部检索')
        if self.calls >= self.max_calls:
            raise RuntimeError('达到本阶段模型调用上限')
        self.calls += 1
        request = payload['request']
        if 'messages' in request:
            # 作者框架保留标准chat消息，项目驱动沿用contents；两者使用相同服务配置和计费记录。
            messages = [dict(message) for message in request['messages']]
        else:
            messages = [{'role': 'system', 'content': request['config']['system_instruction']}]
            for message in request['contents']:
                messages.append({'role': 'assistant' if message['role'] == 'model' else message['role'],
                                 'content': '\n'.join(part.get('text', '') for part in message['parts'])})
        if not messages or any(message.get('role') not in ('system', 'user', 'assistant')
                               or not isinstance(message.get('content'), str) for message in messages):
            raise ValueError('正式研究服务需要非空的文本chat消息')
        body = {'model': self.config.model, 'messages': messages, 'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens}
        if self.config.raw_completion:
            from harness.integrations.qwen_text_template import render
            body['prompt']=render(body.pop('messages'))
            body['stop']=['<|im_end|>','<|endoftext|>']
        identity = sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        prefix = f'{self.calls:04d}-{identity[:16]}'
        atomic_json(self.folder / (prefix + '-request.json'), {'actor': ACTOR.get(), 'request_sha256': identity, 'body': body})
        response = await asyncio.to_thread(self.transport.send, ChatRequest(
            self.transport.endpoint, body, self.config.timeout_seconds))
        atomic_json(self.folder / (prefix + '-raw-response.json'), {'status': response.status,'body':response.body})
        if response.status != 200:
            atomic_json(self.folder / (prefix + '-response.json'), {'status': response.status,
                'request_sha256': identity, 'detail': response.body})
            raise RuntimeError('正式模型HTTP状态：' + str(response.status))
        choices = response.body.get('choices', [])
        text = (choices[0].get('text') if self.config.raw_completion else choices[0].get('message', {}).get('content')) if len(choices)==1 else None
        if not isinstance(text,str):
            raise ValueError('正式模型没有返回唯一文本提案')
        usage = response.body.get('usage', {})
        self.usage.append(usage)
        atomic_json(self.folder / (prefix + '-response.json'), {'status': response.status, 'text': text,
            'usage': usage, 'model': response.body.get('model'), 'request_id': response.body.get('id'),
            'finish_reason': choices[0].get('finish_reason'), 'request_sha256': identity})
        # 格式不匹配留待恢复，不把其他模型或开发者新写的程序替代原响应。
        return {'text': text}
