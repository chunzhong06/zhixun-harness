"""从同一公开起点重放首轮固定选样和九轮全池续训，单独保存交接并与原组计分。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]

from engineering.cli.study.run_rul_series import RegularUpdate
from engineering.cli.study.score_local_defaults import BASE, score
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from harness.runtime.storage import LocalStore


class FirstEngineFilter(RegularUpdate):
    identity = "engine-first-f21-q0-le-then-full-v1"

    def propose(self, context):
        plan = super().propose(context)
        manifest = "f21-q0-le"
        if context["batch_id"] == "round-1" and manifest in context["fit_manifests"]["names"]:
            action = plan["actions"][0]
            action["fit_manifest"] = manifest
            action["reason"] = "首轮按事前固定的第 20 号传感器低四分位清单续训。"
        return plan


def main():
    folder = BASE / "first-engine-filter" / "engine"
    driver = {"mode": "rule"}
    with LocalStore(folder / "study-lock").exclusive():
        harness, batches = assemble_public(
            "engine", ROOT / "data/official", folder, "fixed", driver, select_training_rows=True
        )
        harness.planner = FirstEngineFilter()
        run_prepared("engine", folder, "first-engine-filter", 10, driver, harness, batches)
    score("engine", first_engine_filter=True)


if __name__ == "__main__":
    main()
