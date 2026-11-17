"""跨进程共享调用预算；先保守预留，收到用量再结算，未知结果保留预留。"""
from hashlib import sha256
import json
from pathlib import Path
import uuid
import time
from contextlib import contextmanager
from harness.adapters.storage.local import LocalStore


class ServiceBudget:
    def __init__(self, folder):
        self.store = LocalStore(Path(folder))

    @contextmanager
    def locked(self):
        deadline=time.monotonic()+10
        while True:
            lock=self.store.exclusive()
            try:
                lock.__enter__()
                break
            except OSError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
        try:yield
        finally:lock.__exit__(None,None,None)

    def initialize(self, limit=30., previous_cost=0.):
        with self.locked():
            if self.store.load() is None:
                self.store.save({'limit_cny': limit, 'previous_cost_cny': previous_cost,
                                 'input_per_million': 3.15, 'output_per_million': 25.29, 'calls': []})

    def send(self, request, invoke):
        body = request.body
        if body.get('model') != 'Qwen/Qwen3.8-27B' or type(body.get('max_tokens')) is not int:
            raise ValueError('预算价格仅核对了规定Qwen模型与显式输出上限')
        if not 1 <= body['max_tokens'] <= 32768 or body.get('n', 1) != 1:
            raise ValueError('仅允许一次生成及有效输出上限')
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        # UTF-8字节数高估文本token；额外预留消息模板。缓存一律按全价。
        reserve = ((len(raw) + 2048) * 3.15 + body['max_tokens'] * 25.29) / 1e6
        identity = uuid.uuid4().hex
        with self.locked():
            state = self.store.load()
            if state is None:
                raise ValueError('尚未初始化共享预算')
            used = state['previous_cost_cny'] + sum(c['accounted_cny'] for c in state['calls'])
            if used + reserve > state['limit_cny']:
                raise RuntimeError('本轮30元预算不足，未发送请求')
            state['calls'].append({'id': identity, 'request_sha256': sha256(raw).hexdigest(),
                'reserved_cny': reserve, 'accounted_cny': reserve, 'state': 'reserved'})
            self.store.save(state)
        response = invoke(request)  # 超时或异常保留预留，不自动重发可能已经扣费的调用。
        usage = response.body.get('usage', {})
        with self.locked():
            state = self.store.load()
            entry = next(c for c in state['calls'] if c['id'] == identity)
            entry.update(http_status=response.status, state='unknown_usage')
            if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                entry.update(usage=usage, state='usage_received',
                    accounted_cny=(usage['prompt_tokens'] * 3.15 + usage['completion_tokens'] * 25.29) / 1e6)
            self.store.save(state)
        return response
