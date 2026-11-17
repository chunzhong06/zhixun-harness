"""首轮试改的共同提案格式。请求只含当时的观察和工具，采用结果随后由 Harness 产生。"""
from dataclasses import asdict
import json

from harness.core.contracts import ActionPlan, GasPairPatch, digest
from harness.adapters.datasets.official_gas import LABELS
from harness.adapters.providers.pytrio_planner import ChatRequest, PyTrioPlanner, _invalid_constant, _strict_object
from harness.learning.models.gas_fit_queries import RemovalQuery
from harness.adapters.storage.local import atomic_json


INSTRUCTION = (
    "你在维护一个气体分类器。根据当前测量、训练覆盖和已见错判提出一次改进假设，说明理由。"
    "你可以筛选训练行后续训一遍（filter），用历史响应修正一个已见误判方向（patch），或继续使用在用版（keep）。"
    "只返回一个 JSON 对象，包含 observation_key、parent_sha256、reason、action。前两项复制输入。"
    "action 可为 {kind:filter,feature:已有字段,op:lt/le/gt/ge,threshold:数值}，条件匹配的训练行将被剔除；"
    "也可为 {kind:patch,reported:模型原判的英文类别,target:真实英文类别} 或 {kind:keep}。"
    "对候选使用相同的探索与独立确认；依据不足时可以保持。"
)


def fixed_action(packet, rank=0):
    eligible = [error for error in packet["observed_errors"] if error["count"] >= 2]
    if len(eligible) <= rank:
        return {"kind":"keep"}
    ordered = sorted(eligible, key=lambda e: (-e["count"], LABELS.index(e["actual"]), LABELS.index(e["predicted"])))
    chosen = ordered[rank]
    return {"kind":"patch", "reported":chosen["predicted"], "target":chosen["actual"]}


def validate_reply(reply, packet):
    if (not isinstance(reply, dict) or set(reply) != {"observation_key","parent_sha256","reason","action"}
            or any(reply[k] != packet[k] for k in ("observation_key","parent_sha256"))
            or not isinstance(reply["reason"],str) or not reply["reason"].strip()):
        raise ValueError("建议须对应本次观察、父权重和具体理由")
    action = reply["action"]
    if not isinstance(action,dict):
        raise ValueError("改法必须是结构化动作")
    fields = set(action)
    kind = action.get("kind")
    if kind == "keep" and fields == {"kind"}:
        return reply
    if kind == "filter" and fields == {"kind","feature","op","threshold"}:
        RemovalQuery(action["feature"],action["op"],action["threshold"])
        return reply
    if kind == "patch" and fields == {"kind","reported","target"}:
        if any((e["actual"],e["predicted"]) == (action["target"],action["reported"]) for e in packet["observed_errors"]):
            return reply
    raise ValueError("动作超出工具目录或指定的错判方向与本次观察不符")


class ExampleProposal:
    """离线服务的固定回复，经与真实传输一致的请求与解码路径进入维护。"""
    def __init__(self, kind, sample_filter):
        self.kind, self.sample_filter = kind, sample_filter
        self.identity = digest(["gas-example",kind,sample_filter])

    def propose(self, packet):
        action = {"kind":"keep"}
        if self.kind == "patch":
            action = fixed_action(packet)
        elif self.kind == "alternate-patch":
            action = fixed_action(packet, rank=1)
        elif self.kind == "filter":
            action = {"kind":"filter", **self.sample_filter}
        return {"observation_key":packet["observation_key"], "parent_sha256":packet["parent_sha256"],
                "reason":"离线复核已登记的首轮动作", "action":action}


def proposal_messages(packet):
    return [{"role":"system","content":INSTRUCTION},
            {"role":"user","content":json.dumps(packet,ensure_ascii=False,allow_nan=False)}]


def obtain_proposal(folder, packet, config, transport, source):
    """先记录调用意图和返回内容，再解析。恢复只复用同一请求的回复，避免重复消费额度。"""
    messages = proposal_messages(packet)
    body = {"model":config.model,"messages":messages,"max_tokens":config.max_tokens,
            "temperature":config.temperature,"stream":False}
    key = digest([source,asdict(config),transport.identity,body])
    path = folder/"exchange.json"
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved["request_id"] != key:
            raise ValueError("此目录已有其他请求")
        if saved["status"] != "received":
            raise RuntimeError("此前请求已开始，尚无可复用回复；请核对服务回执")
    else:
        saved = {"request_id":key,"source":source,"status":"started","model":config.model,
                 "real_service":transport.network_enabled,"observation_key":packet["observation_key"]}
        atomic_json(path,saved)
        response = transport.send(ChatRequest(config.base_url.rstrip("/")+"/chat/completions",body,config.timeout_seconds))
        if response.status != 200:
            saved.update(status="failed",http_status=response.status)
            atomic_json(path,saved)
            raise ValueError(f"服务返回 HTTP {response.status}")
        choices = response.body.get("choices")
        if (not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict)
                or choices[0].get("finish_reason")!="stop" or not isinstance(choices[0].get("message"),dict)):
            saved.update(status="invalid",usage=response.body.get("usage"))
            atomic_json(path,saved)
            raise ValueError("需要一条完整的服务回复")
        content = choices[0]["message"].get("content")
        if not isinstance(content,str) or len(content)>65536:
            saved.update(status="invalid",usage=response.body.get("usage"))
            atomic_json(path,saved)
            raise ValueError("服务回复超出提案边界")
        saved.update(status="received",content=content,reported_model=response.body.get("model"),
                     usage=PyTrioPlanner._usage(response.body))
        atomic_json(path,saved)
    parsed = json.loads(saved["content"],object_pairs_hook=_strict_object,parse_constant=_invalid_constant)
    return validate_reply(parsed,packet),saved


class FirstProposalPlanner:
    def __init__(self, reply):
        self.reply = reply
        self.identity = digest(["first-gas-proposal-v1",reply])

    def propose(self, context):
        action = self.reply["action"]
        if context["trials"] or action["kind"]=="keep":
            return {"actions":[],"stop":True}
        fact = next(f for f in context["facts"]["evidence"] if f["kind"]=="performance")
        common = {"reason":self.reply["reason"], "evidence_refs":(fact["id"],)}
        if action["kind"]=="filter":
            names = context["fit_manifests"]["names"]
            if not names:
                return {"actions":[],"stop":True}
            chosen = ActionPlan("recent",epochs=1,fit_manifest=names[0],**common)
        else:
            patch = GasPairPatch(LABELS.index(action["reported"])+1,LABELS.index(action["target"])+1,"shape-blend")
            chosen = ActionPlan("recent",gas_pair=patch,**common)
        return {"actions":[chosen.to_record()],"stop":False}
