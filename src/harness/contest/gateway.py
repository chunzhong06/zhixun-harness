"""让既有研究驱动使用SDK宿主RPC；不读取凭据，也不建立外部HTTP连接。"""
import re
from types import SimpleNamespace

from harness.adapters.providers.pytrio_planner import ChatResponse
from harness.adapters.providers.service_research_broker import ServiceResearchBroker
from harness.adapters.storage.local import atomic_json


def model_options(model_name, options):
    """为规定的Qwen基础模型保留JSON动作的输出预算，其他模型沿用原参数。"""
    options = dict(options)
    if model_name == 'Qwen/Qwen3.8-27B':
        # 官方SDK支持该选项。全量请求曾耗尽8192个思考token而无正文，
        # 因此这里明确关闭额外思考，仍由同一基础模型生成和选择方案。
        options.setdefault('enable_thinking', False)
    return options


class GatewayTransport:
    endpoint = 'https://contest-host.invalid/v1/chat/completions'

    def __init__(self, rnd, folder):
        self.rnd, self.folder = rnd, folder

    @staticmethod
    def _prompt_messages(prompt):
        """把既有Qwen ChatML模板还原成宿主支持的标准chat消息。"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('raw completion请求缺少prompt')
        matches = re.findall(
            r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>',
            prompt,
            flags=re.DOTALL,
        )
        messages = []
        for role, content in matches:
            content = content.rstrip('\n')
            if role == 'assistant' and content.startswith('<think>\n\n</think>\n\n'):
                content = content[len('<think>\n\n</think>\n\n'):]
            if content:
                messages.append({'role': role, 'content': content})
        if not messages:
            # 兼容不含模板标记的纯文本调用；仍通过宿主模型，不建立外部连接。
            messages = [{'role': 'user', 'content': prompt}]
        return messages

    @classmethod
    def _request_messages(cls, body):
        messages = body.get('messages')
        if messages is None:
            messages = cls._prompt_messages(body.get('prompt'))
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(item, dict)
                       or item.get('role') not in ('system', 'user', 'assistant')
                       or not isinstance(item.get('content'), str)
                       for item in messages)):
            raise ValueError('研究请求必须包含有效的文本chat消息')
        return [dict(item) for item in messages]

    def _response(self, raw_request, response):
        if not raw_request:
            return response
        choices = response.get('choices', [])
        if len(choices) != 1:
            return response
        message = choices[0].get('message', {})
        text = message.get('content')
        if not isinstance(text, str):
            return response
        return {
            **response,
            'choices': [{
                'text': text,
                'finish_reason': choices[0].get('finish_reason'),
            }],
        }

    def send(self, request):
        body = request.body
        if body.get('model') != self.rnd.model_name:
            raise ValueError('研究请求与本轮宿主模型不一致')
        messages = self._request_messages(body)
        raw_request = 'messages' not in body
        available = {key: body[key] for key in ('temperature', 'max_tokens', 'enable_thinking')
                     if key in body}
        options = model_options(self.rnd.model_name, available)
        from .round_budget import current, CloseResearch, HostCallInterrupted
        budget = current.get()
        if budget and not budget.allowance(budget.program_limit,request=True):
            budget.marker('skip_research_request')
            raise CloseResearch('时间留给已有产物的选案与交接')
        if budget:
            budget.marker('research_request')
            started = budget.clock()
        try:
            receipt = self.rnd.model(messages, **options)
        except (TimeoutError,RuntimeError,ValueError) as exc:
            if getattr(self.rnd,'_outcome_unknown',False):
                raise HostCallInterrupted('宿主模型请求结果未知') from exc
            raise
        if budget:
            budget.observe('model',budget.clock()-started)
        atomic_json(self.folder/f'host-call-{len(self.rnd.call_ids):04d}.json', {
            'call_id': receipt.call_id, 'model': self.rnd.model_name,
            'usage': receipt.response.get('usage', {}),
            'request_url': request.url,
            'raw_completion_request': raw_request,
        })
        return ChatResponse(200, self._response(raw_request, receipt.response))


def broker_factory(rnd):
    def create(folder, calls):
        config = SimpleNamespace(model=rnd.model_name, max_tokens=rnd.max_output_tokens,
                                 temperature=0., raw_completion=False, timeout_seconds=600.)
        return ServiceResearchBroker(folder/'gateway', config,
                                     GatewayTransport(rnd, folder/'gateway'), max_calls=calls)
    return create
