"""将按完整退化路径去重后的固定模型送回原始单航班命令入口。"""
import json
import joblib
from score_flight_trajectory_split import BASE, DEST as PRIOR, file_hash, LocalStore
from verify_flight_segment_entry import verify_entry
from engineering.cli.workflow.flight_input_predict import infer_conditions
from harness.models.flight_masked_readings import ANCHOR

DEST=BASE/"flight-trajectory-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("该轨迹划分已从原始入口重发")
        meta=json.loads((PRIOR/"scores.json").read_bytes())
        if file_hash(PRIOR/"source.joblib")!=meta["源模型SHA256"]:raise ValueError("源权重身份不同")
        trained=joblib.load(PRIOR/"source.joblib")
        artifact=DEST/"measured.joblib"
        joblib.dump({"prep":trained["prep"],"points":trained["points"],"input_view":"measured",
                     "readout":trained["readouts"].readers[0]},artifact)
        verify_entry(DEST,PRIOR,ANCHOR,artifact,file_hash(artifact),"condition-response",infer_conditions)
