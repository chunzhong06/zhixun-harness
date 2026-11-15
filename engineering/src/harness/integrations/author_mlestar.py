"""保留固定作者流程，只替换模型和内建搜索的传输边界。

依赖单独的 MLE-STAR 环境；不在普通三任务入口导入 ADK。
每个模型请求由新的隔离 Luna 实例回答，搜索结果经独立队列回传。
"""
import asyncio
import functools
from collections import defaultdict,deque
from contextvars import ContextVar
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from pydantic import Field

from harness.runtime.storage import atomic_json

from harness.integrations.research_broker import ACTOR, AuthorBroker


class InboxAuthorLlm(BaseLlm):
    broker:AuthorBroker=Field(exclude=True)

    async def generate_content_async(self,llm_request,stream=False):
        # Gemini 的搜索原本在供应方完成。这里保留其输入/结果闭环，作者回调只接收最终文本。
        data=llm_request.model_dump(mode='json',exclude={'tools_dict'},exclude_none=True)
        search_allowed=any('google_search' in t for t in data.get('config',{}).get('tools',[]))
        observations=[]
        for _ in range(4):
            answer=await self.broker.exchange('model',{'request':data,'search_allowed':search_allowed,'search_observations':observations})
            queries=answer.get('search_queries')
            if queries is not None:
                if not search_allowed or not isinstance(queries,list) or not 1<=len(queries)<=2 or not all(isinstance(q,str) and q.strip() for q in queries):
                    raise ValueError('当前作者角色没有该搜索权限或查询格式无效')
                for query in queries:
                    result=await self.broker.exchange('search',{'query':query,'purpose':'作者模型检索或调试，公开资料'})
                    observations.append({'query':query,'result':result})
                continue
            if not isinstance(answer.get('text'),str) or not answer['text'].strip():raise ValueError('缺少作者角色的真实文本响应')
            yield LlmResponse(content=types.Content(role='model',parts=[types.Part(text=answer['text'])]),partial=False,turn_complete=True)
            return
        raise RuntimeError('单次作者角色检索超过三轮后仍未返回最终响应')


class InboxSearchMarker(BaseTool):
    """仅解除 Google 供应方绑定；搜索实际由 Broker 发出并记录。"""
    def __init__(self):super().__init__(name='google_search',description='公开模型与代码搜索')

    async def process_llm_request(self,*,tool_context,llm_request):
        llm_request.config.tools=llm_request.config.tools or []
        llm_request.config.tools.append(types.Tool(google_search=types.GoogleSearch()))


def identify_actor(callback_context,llm_request):
    ACTOR.set(callback_context.agent_name)


def load_author_pipeline(source,configuration,broker):
    """不执行包首页的云账户发现；作者子模块、提示词、循环及回调按原件导入。"""
    source=Path(source).resolve();package=source/'machine_learning_engineering'
    if 'machine_learning_engineering' in sys.modules:raise RuntimeError('每个作者运行使用新进程，避免复用前一任务状态')
    module=ModuleType('machine_learning_engineering');module.__path__=[str(package)]
    module.__package__='machine_learning_engineering';module.__file__=str(package/'__init__.py')
    module.__spec__=importlib.util.spec_from_file_location(module.__name__,module.__file__,submodule_search_locations=module.__path__)
    sys.modules[module.__name__]=module
    config=importlib.import_module('machine_learning_engineering.shared_libraries.config')
    for key,value in configuration.items():
        if not hasattr(config.CONFIG,key):raise ValueError('作者配置不存在：'+key)
        setattr(config.CONFIG,key,value)
    author=importlib.import_module('machine_learning_engineering.agent')
    from harness.integrations.author_compatibility import submission_condition,complete_refinement_selection
    code_util=importlib.import_module('machine_learning_engineering.shared_libraries.code_util')
    refinement=importlib.import_module('machine_learning_engineering.sub_agents.refinement.agent')
    code_util.get_run_code_condition=functools.partial(submission_condition,code_util.get_run_code_condition)
    pipeline=author.mle_pipeline_agent
    pipeline.parent_agent=None  # 直接启动研究流水线，跳过面向用户的聊天转接入口。
    nodes=[]
    def bind(node):
        nodes.append({'name':node.name,'type':type(node).__name__,'max_iterations':getattr(node,'max_iterations',None)})
        if node.after_agent_callback is refinement.update_outer_loop_states:
            node.after_agent_callback=functools.partial(complete_refinement_selection,refinement.update_outer_loop_states)
        if hasattr(node,'model'):
            node.model=InboxAuthorLlm(model='gpt-6-luna',broker=broker)
            previous=node.before_model_callback
            node.before_model_callback=[identify_actor]+(previous if isinstance(previous,list) else ([previous] if previous else []))
            node.tools=[InboxSearchMarker() if getattr(t,'name',None)=='google_search' else t for t in node.tools]
        for child in node.sub_agents:bind(child)
    bind(pipeline)
    return pipeline,nodes
