"""研究请求队列与预算；不依赖作者框架或特定模型SDK。"""
import asyncio
from collections import defaultdict, deque
from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path
import time
from harness.adapters.storage.local import atomic_json

ACTOR=ContextVar('mlestar_actor',default='unknown')


class AuthorBroker:
    """一个编号对应一次真实请求；超时与无效响应不会被静默重发。"""
    def __init__(self,folder,*,max_calls=48,timeout=200,replay_from=None,prior_model_invocations=0,prior_searches=0):
        self.folder=Path(folder);self.max_calls=max_calls;self.timeout=timeout
        self.calls=0;self.search_calls=0;self.sequence=0
        self.prior_model_invocations=prior_model_invocations;self.prior_searches=prior_searches
        self.reused_model_calls=0;self.reused_search_calls=0;self.replays=[];self.previous=defaultdict(deque)
        if replay_from:
            previous=Path(replay_from)/'queue'
            for path in sorted((previous/'requests').glob('*/*.json')):
                response=previous/'responses'/path.parent.name/path.name
                if not response.is_file():continue
                request=json.loads(path.read_bytes());answer=json.loads(response.read_bytes())
                if request['request_id']!=answer.get('request_id'):raise ValueError('待恢复响应身份不同')
                self.previous[self.semantic_key(request['context'])].append((path,answer['plan']))

    @staticmethod
    def semantic_key(envelope):
        return sha256(json.dumps({k:envelope[k] for k in ('kind','actor','payload')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    async def exchange(self,kind,payload):
        self.sequence+=1
        if kind=='model':
            self.calls+=1
        else:
            self.search_calls+=1
        envelope={'kind':kind,'sequence':self.sequence,'actor':ACTOR.get(),'payload':payload}
        key=sha256(json.dumps(envelope,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        task='mle-model' if kind=='model' else 'mle-search'
        # 完整身份在JSON中校验；短文件名减少跨工具抄写和Windows深目录访问失败。
        name=f'{self.sequence:04d}-{ACTOR.get()}-{key[:16]}.json'
        request={'request_id':key,'instruction':
            '仅执行context.payload.request中的角色与原始指令。按隔离调用契约保存响应原文，文件封装交由发布工具处理，不手工填写request_id、plan或额外text包装。仅当search_allowed为真时可按契约申请公开检索。不得自行读任务数据或其他请求，不从请求外取得研究信息。',
            'context':envelope}
        atomic_json(self.folder/'requests'/task/name,request)
        response=self.folder/'responses'/task/name
        semantic=self.semantic_key(envelope)
        if self.previous[semantic]:
            origin,plan=self.previous[semantic].popleft()
            if kind=='model':self.reused_model_calls+=1
            else:self.reused_search_calls+=1
            atomic_json(response,{'request_id':key,'plan':plan})
            self.replays.append({'original_request':str(origin),'current_request':str(self.folder/'requests'/task/name),
                'semantic_sha256':semantic,'kind':kind})
            atomic_json(self.folder/'replays.json',self.replays)
        if self.prior_model_invocations+self.calls-self.reused_model_calls>self.max_calls:
            raise RuntimeError('包含原失败和恢复新增调用后达到模型预算')
        if self.prior_searches+self.search_calls-self.reused_search_calls>12:
            raise RuntimeError('包含原失败和恢复新增检索后达到搜索预算')
        started=time.monotonic()
        while not response.exists():
            if time.monotonic()-started>=self.timeout:raise TimeoutError(f'{task}/{name}响应超时')
            await asyncio.sleep(.25)
        raw=json.loads(response.read_bytes())
        if raw.get('request_id')!=key or not isinstance(raw.get('plan'),dict):raise ValueError('作者流程响应身份或格式不同')
        return raw['plan']

