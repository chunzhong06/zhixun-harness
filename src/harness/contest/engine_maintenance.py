"""以实际发放和已确认的选案决定发动机是否值得再训练。"""
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path

from harness.adapters.storage.local import atomic_json
from .round_budget import CloseResearch, RoundBudget


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class EngineRoundBudget(RoundBudget):
    def __init__(self, rnd):
        super().__init__("C", rnd.wall_seconds)
        self.rnd = rnd
        self.prior = {}
        self.identity = {}
        self.stop_reason = None
        self.outcome = None

    def restore(self, folder):
        super().restore(folder)
        with Path(self.rnd.data.train_path).open("rb") as stream:
            train = file_digest(stream, "sha256").hexdigest()
        validation = sha256()
        self.rows = 0
        with Path(self.rnd.data.validation_path).open("rb") as stream:
            for line in stream:
                validation.update(line)
                self.rows += bool(line.strip())
        self.identity = {"train": train, "validation": validation.hexdigest()}
        path = folder / "engine-maintenance.json"
        if path.is_file():
            self.prior = json.loads(path.read_bytes())
        parent = self.rnd.parent.checkpoint_sha256.removeprefix("sha256:")
        if (self.prior.get("identity") == self.identity and self.prior.get("selected_sha") == parent
                and self.prior.get("settled") is True):
            self.stop_reason = "同父版与同一发放上已实测并完成选案，继续沿用该结果"

        measured = self.rnd.parent.receipt.get("summary") or {}
        cold, warm = measured.get("first_row_seconds"), measured.get("later_p95_seconds")
        # 逐行容器缺少遥测时保留旧 C 单行时延的保守估计；只使用当轮可见信息。
        if measured.get("sample_count") != self.rows or not all(nonnegative(v) for v in (cold, warm)):
            cold, warm = 3., .006
        self.measured_stream_reserve = 45 + 1.35 * (cold + max(0, self.rows-1)*warm)

    @property
    def gateway(self):
        return max(120., 1.5 * max(self.times["model"], default=0.))

    @property
    def publish(self):
        configured = getattr(self, "estimate_predict", None)
        return max(self.measured_stream_reserve, 1.5 * max(self.times["submission"], default=0.),
                   configured() if configured else 0.)

    def allowance(self, limit, *, request=False):
        if self.stop_reason:
            raise CloseResearch(self.stop_reason)
        return super().allowance(limit, request=request)

    def record_choice(self, report, selected):
        """仅在完成真实比较后封存原发放；预算提前收尾保留下一轮的重试机会。"""
        digest = selected["checkpoint_sha256"].removeprefix("sha256:")
        parent = self.rnd.parent.checkpoint_sha256.removeprefix("sha256:")
        if self.stop_reason:
            self.outcome = self.prior
            return
        before = (self.rnd.parent.receipt.get("summary") or {}).get("metrics", {}).get("rmse")
        after = (selected.get("summary") or {}).get("metrics", {}).get("rmse")
        tested = report.get("successful_training_candidates", 0) > 0
        compared = (report.get("checkpoint_sha256") == parent
                    or any(e["stage"] == "submit_candidate" for e in self.events))
        settled = (tested and compared and selected.get("role") != "historical"
                   and (parent == digest or (nonnegative(before) and nonnegative(after) and after >= .995*before)))
        self.outcome = dict(identity=self.identity, selected_sha=digest, settled=settled)

    def save(self, folder):
        super().save(folder)
        if self.outcome is not None:
            atomic_json(folder / "engine-maintenance.json", self.outcome)
