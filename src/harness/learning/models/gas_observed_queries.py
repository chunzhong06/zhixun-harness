"""用当前合法探索误差生成选样条件，沿用冻结提案的训练与采用流程。"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from harness.core.contracts import digest
from harness.learning.models.gas_fit_queries import FEATURES, GasQueryTools, observation_key
from harness.learning.models.official_gas_cpu import OfficialGasTools
from harness.adapters.storage.local import atomic_json


def propose_condition(training, explore, predictions):
    """围绕重复的同方向错判提出一项可反驳的选样假设，所有清单具有相同的训练行数。"""
    counts = Counter((r.label, predictions[r.sample_id]) for r in explore if r.label != predictions[r.sample_id])
    if not counts:
        return [], {"reason": "no_errors"}
    (target, reported), count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    evidence = {"target":target, "reported":reported, "errors":count}
    if count < 2:
        return [], {**evidence, "reason":"no_repeated_error"}
    observed = np.array([r.values for r in explore if r.label == target and predictions[r.sample_id] == reported])
    source = np.array([r.values for r in training])
    target_rows = np.array([r.values for r in training if r.label == target])
    if len(target_rows) < 2:
        return [], {**evidence, "reason":"missing_training_class"}
    q1, center, q3 = np.quantile(target_rows, [.25,.5,.75], axis=0)
    offset = np.divide(np.median(observed, axis=0)-center, q3-q1,
                       out=np.zeros_like(center), where=q3>q1)
    trim = round(len(source)/8)
    # 剔除与当前错例偏移方向相反的一端。同值边界会改变预算，改查下一个测量字段。
    for column in np.argsort(-np.abs(offset), kind="stable"):
        shift = float(offset[column])
        if not shift or not 0 < trim < len(source):
            break
        ordered = np.sort(source[:, column])
        pos = trim if shift > 0 else len(source)-trim
        if ordered[pos-1] == ordered[pos]:
            continue
        op, threshold = ("le", ordered[pos-1]) if shift > 0 else ("ge",ordered[pos])
        return [{"feature":FEATURES[column], "op":op, "threshold":float(threshold)}], {
            **evidence, "standardized_displacement":shift, "excluded_rows":trim}
    return [], {**evidence, "reason":"no_eligible_boundary"}


class ObservedGasQueryTools(OfficialGasTools):
    """首轮冻结和逐轮提案共用编译器。已签收提案在进程重启后仍能恢复未完成训练。"""

    def __init__(self, path, package, batches, runner, folder, policy):
        super().__init__(path, package, batches, runner)
        if policy not in {"first", "current"}:
            raise ValueError("未登记的提案策略")
        self.policy = policy
        self.first_batch = batches[0].batch_id
        self.ledger = Path(folder) / "query-proposals.json"
        self.catalog_args = path, package, batches, runner
        self.records = {}
        empty = GasQueryTools(*self.catalog_args, {})
        self.identity = digest([empty.identity, sha256(Path(__file__).read_bytes()).hexdigest(), policy])
        self.capabilities["continuation_epochs"] = [1]
        if self.ledger.exists():
            saved = json.loads(self.ledger.read_bytes())
            if saved["identity"] != self.identity or digest(saved["records"]) != saved["records_digest"]:
                raise ValueError("提案回执与本次工具或内容不符")
            self.records = saved["records"]
        self._restore_catalog()

    def _restore_catalog(self):
        proposals = {bid:record["proposal"] for bid,record in self.records.items()}
        self.catalog = GasQueryTools(*self.catalog_args, proposals)
        # 后续批次写入新条件，不改变早期工作单的实现身份及缓存键。
        self.catalog.identity = self.identity

    def prepare_fit_manifests(self, batch, baseline, parent):
        key = observation_key(self.package, batch, baseline)
        manifest, _ = self.runner.artifacts.read(parent)
        parent_hash = "sha256:" + manifest["sha256"]
        bid = batch.batch_id
        if bid in self.records:
            saved = self.records[bid]["proposal"]
            if saved["parent_sha256"] != parent_hash or saved["observation_key"] != key:
                raise ValueError("已签收的提案属于另一父版本或观察")
            return
        if self.policy == "first" and bid != self.first_batch:
            original = self.records[self.first_batch]
            conditions, evidence = original["proposal"]["conditions"], original["evidence"]
            basis = self.first_batch
        else:
            visible = [r for r in batch.samples if r.role == "explore" and r.available_at == batch.ordinal]
            conditions, evidence = propose_condition(self.package.train, visible, baseline["predictions"])
            basis = bid
        self.records[bid] = {"basis_batch":basis, "evidence":evidence, "proposal":{
            "source":"local/"+self.policy, "parent_sha256":parent_hash,
            "observation_key":key, "conditions":conditions}}
        self._restore_catalog()
        atomic_json(self.ledger, {"identity":self.identity, "records":self.records,
                                  "records_digest":digest(self.records)})

    def fit_manifest_context(self, batch, baseline):
        return self.catalog.fit_manifest_context(batch, baseline)

    def select_fit_rows(self, name, eligible):
        return self.catalog.select_fit_rows(name, eligible)

    def _build_request(self, task, batch_id, action, rows, initialize, parent):
        return self.catalog._build_request(task,batch_id,action,rows,initialize,parent)
