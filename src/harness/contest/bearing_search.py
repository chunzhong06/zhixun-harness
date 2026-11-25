"""相同轴承模型在同一份训练与反馈上已完整试过候选时，可跳过后续重复开训。"""
from hashlib import file_digest
import json
from pathlib import Path


def inputs_for(rnd, version):
    def digest(path):
        with Path(path).open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest()
    return {"implementation":version, "artifact_id":rnd.parent.artifact_id,
            "parent":rnd.parent.checkpoint_sha256.removeprefix("sha256:"),
            "train":digest(rnd.data.train_path), "validation":digest(rnd.data.validation_path)}


def already_compared(restored, inputs):
    path = restored / "bearing-comparison.json"
    try:
        return json.loads(path.read_bytes()) == inputs
    except (OSError, ValueError):
        return False


def finished_without_gain(report, selected, inputs):
    """仅在完整实算且最终确实保留输入父版时，才能为下一轮登记。"""
    return (selected["artifact_id"] == inputs["artifact_id"] and selected["checkpoint_sha256"] == inputs["parent"]
            and report.get("parent_checkpoint_sha256") == inputs["parent"]
            and report.get("checkpoint_sha256") == inputs["parent"]
            and report.get("status") == "trained_not_adopted" and report.get("successful_training_candidates",0)>0
            and not report.get("error") and not report.get("invalid_actions") and bool(report.get("experiments"))
            and all(t.get("returncode") == 0 and not t.get("error") for t in report["experiments"]))
