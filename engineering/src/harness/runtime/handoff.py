"""逐批交接：先保存输出信封，取得明确接收回执后才推进取数游标。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from harness.contracts import BatchContext, digest
from harness.runtime.storage import LocalStore, atomic_json


class BatchEndpoint(Protocol):
    """平台适配器保证同一request_id重复提交只确认同一内容。

    fetch(cursor)返回(batch,next_cursor)或None；提交回执包含request_id和
    accepted=True。该内部契约由正式SDK适配，不假设它就是官方字段。
    """
    identity: str
    def fetch(self, cursor): ...
    def submit(self, request_id: str, payload: dict): ...


def run_stream(engine, endpoint: BatchEndpoint, folder: Path, *, after_submit=None):
    store = LocalStore(folder)
    identity = digest([endpoint.identity, asdict(engine.task), asdict(engine.config),
                       engine.tools.identity, engine.planner.identity])
    with store.exclusive():
        state = store.load() or {"identity": identity, "cursor": None, "pending": None, "accepted": []}
        if state["identity"] != identity:
            raise ValueError("逐批交接目录属于不同运行配置")
        while True:
            pending = state["pending"]
            if pending is None:
                received = endpoint.fetch(state["cursor"])
                if received is None:
                    return engine.store.load(), state
                batch, cursor = received
                if cursor == state["cursor"] or any(item["batch_id"] == batch.batch_id for item in state["accepted"]):
                    raise ValueError("平台游标或已接收批次没有推进")
                result = engine.run((batch,))["completed"][batch.batch_id]
                payload = {"batch_id": batch.batch_id, "ordinal": batch.ordinal,
                    "model_id": result["model_id"], "decision": result["decision"],
                    "predictions": result["predictions"], "input_digest": result["input_digest"]}
                request_id = digest([identity, payload])
                pending = {"request_id": request_id, "payload": payload, "next_cursor": cursor}
                state["pending"] = pending
                store.save(state)
            receipt = endpoint.submit(pending["request_id"], pending["payload"])
            if receipt.get("request_id") != pending["request_id"] or receipt.get("accepted") is not True:
                raise RuntimeError("平台尚未确认本次输出；保持原交接游标")
            if after_submit:
                after_submit(receipt)
            # 回执与游标在同一快照提交；确认丢失时重发原信封。
            state["accepted"].append({"request_id":pending["request_id"],
                "batch_id":pending["payload"]["batch_id"], "receipt":receipt})
            state.update(cursor=pending["next_cursor"], pending=None)
            store.save(state)


class LocalAcceptance:
    """在本地保留实际输出；重发相同请求时返回原确认。"""
    def submit(self, request_id, payload):
        import json
        path = self.output / (request_id + ".json")
        store = LocalStore(self.output)
        with store.exclusive():
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8"))["payload"] != payload:
                    raise ValueError("同一输出身份对应不同内容")
            else:
                atomic_json(path, {"request_id":request_id, "payload":payload})
        return {"request_id":request_id, "accepted":True, "evidence_kind":"local_platform_fixture"}


class OfficialBatchEndpoint(LocalAcceptance):
    """本地公开任务的冻结批次依次发放；临时停止取批不改变此前的接收身份。"""
    def __init__(self, batches, output, limit=None):
        self.batches, self.output = batches, output
        self.limit = len(batches) if limit is None else limit
        if type(self.limit) is not int or not 0 <= self.limit <= len(batches):
            raise ValueError("停止位置超出公开批次范围")
        self.identity = "official-local-stream:" + digest([asdict(b) for b in batches])

    def fetch(self, cursor):
        index = 0 if cursor is None else cursor
        if type(index) is not int or not 0 <= index <= len(self.batches):
            raise ValueError("交接游标越界")
        if index >= self.limit:
            return None
        return self.batches[index], index + 1


class LocalPackageEndpoint(LocalAcceptance):
    """工程文件平台，按游标只读当前批；接收目录用于幂等回执测试。"""
    def __init__(self, manifest: Path, output: Path):
        import json
        self.manifest, self.output = Path(manifest).resolve(), Path(output).resolve()
        self.config = json.loads(self.manifest.read_text(encoding="utf-8"))
        from harness.contracts import TaskSpec
        self.task=TaskSpec.parse(self.config["task"])
        self.identity = "local-package-v1:" + digest(self.config)

    def fetch(self, cursor):
        from harness.data.package_input import read_package
        index = 0 if cursor is None else cursor
        if type(index) is not int or not 0 <= index <= len(self.config["batches"]):
            raise ValueError("取批游标无效")
        if index == len(self.config["batches"]):
            return None
        package = read_package(self.manifest, batch_index=index)
        return package.batches[0], index + 1

class PartitionedEndpoint:
    """为首次出现且尚未分用途的fit资料划分；重发沿用原用途与清除决定。

    官方已给探索/采用用途时沿用原声明。过去出现过的服务输入后来返回
    答案时直接保留其合法用途，不将旧输入重新包装为新的采用证据。
    """
    def __init__(self,endpoint,folder,policy,*,metadata=None,task=None):
        self.endpoint,self.store,self.policy=endpoint,LocalStore(folder),policy
        self.metadata=metadata or {}
        self.task=task or getattr(endpoint,"task",None)
        self.identity="partitioned-v2:"+digest([endpoint.identity,asdict(policy),self.metadata,
                                              asdict(self.task) if self.task else None])

    def fetch(self,cursor):
        from dataclasses import replace
        from harness.data.splitting import split_released, _interval
        result=self.endpoint.fetch(cursor)
        if result is None:
            return None
        batch,next_cursor=result
        with self.store.exclusive():
            state=self.store.load() or {"identity":self.identity,"seen":[],"batches":{},"roles":{},"excluded":[],"unit_roles":{},"intervals":{}}
            if state["identity"]!=self.identity:
                raise ValueError("划分目录属于其他规则或数据来源")
            signature=digest(asdict(batch))
            previous=state["batches"].get(batch.batch_id)
            if previous and previous["source"]!=signature:
                raise ValueError("已划分批次内容改变")
            if previous is None:
                new=[s for s in batch.samples if s.sample_id not in state["seen"]]
                if new and all(s.role in {"fit","serve"} for s in new):
                    eligible=tuple(s for s in new if s.role=="fit" and s.available_at==batch.ordinal)
                    # 同设备后续新窗口沿用首次划分，不能跨批重新随机分用途。
                    fixed=tuple(s for s in eligible if self.policy.mode=="group" and s.unit_id in state["unit_roles"])
                    state["roles"].update({s.sample_id:state["unit_roles"][s.unit_id] for s in fixed})
                    eligible=tuple(s for s in eligible if s not in fixed)
                    assigned,manifest=split_released(eligible,batch.ordinal,self.policy,metadata=self.metadata,task=self.task)
                    if self.policy.mode=="time":
                        # 既有用途保持；新窗口若与另一用途的历史窗口重叠，排除新窗口。
                        # 回退成fit会把旧留出窗口的信息带入训练，因此这里不改换用途。
                        kept,conflicts=[],[]
                        for sample in assigned:
                            current=_interval(sample,self.metadata)
                            overlaps=any(old["role"]!=sample.role and old["unit"]==current["unit"]
                                and current["start"]<=old["end"]+self.policy.gap
                                and old["start"]<=current["end"]+self.policy.gap
                                for old in state["intervals"].values())
                            if overlaps:
                                conflicts.append(sample.sample_id)
                            else:
                                kept.append(sample)
                        assigned=tuple(kept)
                        state["excluded"].extend(conflicts)
                        manifest["excluded_history_overlap"]=conflicts
                        manifest["available_roles_after_history_filter"]=sorted({s.role for s in assigned})
                        manifest["assignments"]={f"{s.sample_id}@{s.revision}":s.role for s in assigned}
                        manifest.pop("identity",None)
                        manifest["identity"]=digest(manifest)
                        state["intervals"].update({s.sample_id:{**_interval(s,self.metadata),"role":s.role} for s in assigned})
                    if manifest["status"]=="ready":
                        state["roles"].update({s.sample_id:s.role for s in assigned})
                        state["excluded"].extend(manifest["excluded_overlap"])
                    if self.policy.mode=="group":
                        state["unit_roles"].update({s.unit_id:s.role for s in assigned if s.unit_id is not None})
                else:
                    manifest={"status":"upstream_roles_preserved"}
                state["batches"][batch.batch_id]={"source":signature,"manifest":manifest}
                state["seen"]=sorted(set(state["seen"])|{s.sample_id for s in batch.samples})
                self.store.save(state)
            for s in batch.samples:
                # 已经服务过的输入得到标签时可作探索，不能新建采用证据。
                if s.role=="fit" and s.sample_id not in state["roles"] and state["unit_roles"].get(s.unit_id) in {"explore","adopt"}:
                    state["roles"][s.sample_id]="explore"
            self.store.save(state)
            samples=tuple(replace(s,role=state["roles"].get(s.sample_id,s.role)) for s in batch.samples
                          if s.sample_id not in state["excluded"])
            return replace(batch,samples=samples),next_cursor

    def submit(self,request_id,payload):
        return self.endpoint.submit(request_id,payload)
