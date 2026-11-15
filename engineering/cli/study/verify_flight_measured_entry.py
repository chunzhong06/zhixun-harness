"""发布简化后的工况采样权重，逐条核对完整原始航班预测。"""
import json
import joblib
from score_flight_condition_ablation import DEST as PRIOR, BASE, LocalStore, file_hash
from verify_flight_segment_entry import verify_entry
from engineering.cli.workflow.flight_input_predict import infer_conditions

DEST = BASE / "flight-measured-entry"


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("实测六工况权重已接回入口")
        original = PRIOR / "ablation.joblib"
        summary = json.loads((PRIOR / "scores.json").read_bytes())
        if file_hash(original) != summary["模型SHA256"]:
            raise ValueError("拆分读法的权重身份变化")
        trained = joblib.load(original)
        artifact = DEST / "measured.joblib"
        joblib.dump({"prep":trained["prep"],"points":trained["points"],"input_view":"measured",
                     "readout":trained["models"]["邻域原读数"]},artifact)
        verify_entry(DEST,PRIOR,"邻域原读数分歧加权",artifact,file_hash(artifact),"condition-response",infer_conditions)


if __name__ == "__main__":
    run()
