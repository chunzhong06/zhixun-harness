"""把当轮规划上下文交给本地代理；只有匹配该请求的原始提案可返回控制器。"""
import json
from pathlib import Path
import re
from time import monotonic, sleep

from harness.contracts import digest
from harness.integrations.pytrio_planner import ChatResponse
from harness.runtime.storage import atomic_json


class ProposalInbox:
    network_enabled = False
    identity = "local-proposal-exchange-v1"
    evidence_kind = "local_proposal_response"

    def __init__(self, folder: Path):
        self.folder = folder

    def send(self, request):
        messages = request.body["messages"]
        context = json.loads(messages[-1]["content"])
        # 公开业务名可重复。摘要将提案绑定到实际父版观察、任务和模型配置。
        task, batch = context["task_id"], context["batch_id"]
        if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,120}", item) for item in (task,batch)):
            raise ValueError("本地提案需要有效的任务和轮次标识")
        key = digest(request.body)
        name = f"{batch}-{key}.json"
        outbound = self.folder / "requests" / task / name
        inbound = self.folder / "responses" / task / name
        payload = {"request_id":key, "instruction":messages[0]["content"], "context":context,
                   "response_template":{"request_id":key,"plan":{"stop":True,"actions":[]}}}
        if outbound.exists():
            if json.loads(outbound.read_bytes()) != payload:
                raise ValueError("已发布的提案请求发生变化")
        else:
            atomic_json(outbound,payload)
        deadline = monotonic() + request.timeout_seconds
        while not inbound.is_file():
            short_name=inbound.with_name(key+".json")
            if short_name.is_file():
                candidate=json.loads(short_name.read_bytes())
                if candidate.get("request_id")!=key:
                    raise ValueError("简名响应的请求身份不符")
                # 请求摘要已经包含站点与轮次，规范化文件名后继续使用同一份原回复。
                short_name.replace(inbound)
                break
            if monotonic() >= deadline:
                raise TimeoutError("本地提案未在当轮时限内返回")
            sleep(0.25)
        response = json.loads(inbound.read_bytes())
        if response.get("request_id") != key:
            raise ValueError("本地提案与当前观察不匹配")
        # 交回既有统一解析器，动作、依据、采用决定仍由公共入口核验。
        if "plan" in response and not ({"stop","actions"} & response.keys()):
            plan=response["plan"]
        elif set(response)=={"request_id","stop","actions"}:
            plan={"stop":response["stop"],"actions":response["actions"]}
        else:
            raise ValueError("响应应携带一份明确的 stop/actions 计划")
        content = json.dumps(plan, ensure_ascii=False, allow_nan=False)
        return ChatResponse(200, {"choices":[{"finish_reason":"stop","message":{"content":content}}]})
