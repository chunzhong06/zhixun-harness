"""传输失败后复用身份完全一致的原提案；保留旧失败及费用，仅补未发生的请求。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from engineering.cli.workflow import london_midstream
from harness.contracts import digest
from harness.integrations.proposal_inbox import ProposalInbox
from harness.runtime.storage import atomic_json


class ReplayInbox(ProposalInbox):
    def __init__(self, folder, original):
        super().__init__(folder);self.original=original;self.records=[];self.guard=Lock()

    def send(self, request):
        context=json.loads(request.body["messages"][-1]["content"]);task=context["task_id"]
        key=digest(request.body);name=context["batch_id"]+'-'+key+'.json'
        prior=self.original/"requests"/task/name;response=self.original/"responses"/task/name
        if prior.exists():
            expected={"request_id":key,"instruction":request.body["messages"][0]["content"],"context":context,
                      "response_template":{"request_id":key,"plan":{"stop":True,"actions":[]}}}
            if json.loads(prior.read_bytes())!=expected:raise ValueError("原请求与本次上下文不同，禁止错配重放")
            if response.exists():
                raw=response.read_text(encoding="utf-8");payload,end=json.JSONDecoder().raw_decode(raw)
                trailing=raw[end:].strip()
                # 已定位的传输错误仅是对象后字面量\n；禁止吞掉任何其他附加内容。
                if trailing not in ('',r'\n'):raise ValueError("原响应存在未定义的额外内容，不能自动修复")
                if payload["request_id"]!=key or not isinstance(payload.get("plan"),dict):raise ValueError("原提案身份或对象错误")
                atomic_json(self.folder/"responses"/task/name,payload)
                record={"task":task,"batch":context["batch_id"],"source_request":str(prior),"source_response":str(response),
                        "source_response_sha256":sha256(response.read_bytes()).hexdigest(),"request_id":key,
                        "removed_literal_newline_suffix":bool(trailing),"plan_changed":False}
                with self.guard:
                    self.records.append(record);atomic_json(self.folder.parent/"replay-provenance.json",self.records)
        elif list((self.original/"requests"/task).glob(context["batch_id"]+'-*.json')):
            raise ValueError("同轮原请求存在但摘要不同，不能把不同训练反馈冒充同一提案")
        return super().send(request)


def main(argv):
    ap=ArgumentParser();ap.add_argument("--replay-queue",type=Path,required=True)
    args,remaining=ap.parse_known_args(argv)
    london_midstream.ProposalInbox=lambda folder:ReplayInbox(folder,args.replay_queue)
    return london_midstream.main(remaining)
