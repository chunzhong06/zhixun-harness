"""从原始 18 列逐航班核对六工况固定工具及既有 JSONL 入口。"""
import json
from score_flight_condition_response import DEST as PRIOR, BASE, LocalStore, file_hash
from verify_flight_segment_entry import verify_entry
from engineering.cli.workflow.flight_input_predict import infer_conditions

DEST = BASE / "flight-condition-entry"


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已有工况工具入口核对")
        checks = json.loads((PRIOR / "scores.json").read_bytes())
        artifact = PRIOR / "condition.joblib"
        sha = file_hash(artifact)
        if sha != checks["权重SHA256"]:
            raise ValueError("六工况权重身份改变")
        verify_entry(DEST, PRIOR, "工况分歧固定公式", artifact, sha, "condition-response", infer_conditions)


if __name__ == "__main__":
    run()
