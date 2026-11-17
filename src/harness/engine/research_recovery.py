"""源码研究的持久化模型调用回放；已发但未取得响应的请求不自动重发。"""
from hashlib import sha256
import json
from pathlib import Path

from harness.adapters.providers.research_broker import ACTOR
from harness.adapters.storage.local import atomic_json


class RecoverableBroker:
    """按原请求顺序回放已完成调用，恢复时仍占用原阶段预算。"""

    def __init__(self, broker, folder):
        self.backend, self.folder = broker, Path(folder)
        self.max_calls = broker.max_calls
        self.calls = self.prior_model_invocations = self.reused_model_calls = 0
        self.new_calls = self.replayed_calls = 0
        self.usage = []

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('原生续训恢复仅支持本地源码研究的模型请求')
        if self.calls >= self.max_calls:
            raise RuntimeError('包含已完成调用后达到本阶段模型预算')
        self.calls += 1
        identity = sha256(json.dumps({'kind': kind, 'actor': ACTOR.get(), 'payload': payload},
                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.folder / f'{self.calls:04d}.json'
        if path.exists():
            record = json.loads(path.read_bytes())
            if record['request_sha256'] != identity:
                raise ValueError('恢复请求与原观察或原预算不同，不能复用响应')
            if record['status'] != 'completed':
                raise RuntimeError('已有模型请求未取得完整回执，先核查服务或队列响应，不自动重发')
            self.replayed_calls += 1
            self.usage.extend(record.get('usage', []))
            return record['answer']
        # 外部调用前先落盘：中断后的不确定调用仍计入预算，不能假定未执行。
        record = {'request_sha256': identity, 'status': 'issued', 'sequence': self.calls,
                  'request_context': {'kind': kind, 'actor': ACTOR.get(), 'payload': payload}}
        atomic_json(path, record)
        self.backend.calls = self.calls - 1
        if hasattr(self.backend, 'sequence'):
            self.backend.sequence = self.calls - 1
        self.new_calls += 1
        try:
            answer = await self.backend.exchange(kind, payload)
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            atomic_json(path, record)
            raise
        usage = getattr(self.backend, 'usage', [])
        record.update(status='completed', answer=answer, usage=usage[-1:] if usage else [])
        atomic_json(path, record)
        self.usage.extend(record['usage'])
        return answer


def file_identity(paths):
    """冻结本次恢复真正依赖的输入，不扫描历史结果。"""
    return {str(Path(path).resolve()): sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def verify_files(expected):
    for name, identity in expected.items():
        path = Path(name)
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != identity:
            raise ValueError('恢复依赖或原产物已变化：' + name)


def freeze_invocation(path, options, inputs, *, resume):
    """恢复必须匹配原配置和数据；新增阶段或改源码应开启新运行。"""
    path = Path(path)
    if resume:
        saved = json.loads(path.read_bytes())
        if saved['options'] != options:
            raise ValueError('恢复参数不同，请沿用原配置')
        verify_files(saved['inputs'])
        return saved
    if path.exists():
        raise ValueError('运行身份已存在')
    result = {'options': options, 'inputs': file_identity(inputs)}
    atomic_json(path, result)
    return result
