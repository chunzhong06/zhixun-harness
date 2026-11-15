"""给已实测均分与最差设备同时改善的未校正版重走同一个发布入口。"""
import json
from verify_flight_component_entry import DEST as ENTRY, FEATURES, BASE, LocalStore, file_hash
from verify_flight_segment_entry import verify_entry
from engineering.cli.workflow.flight_input_predict import infer_component_source
from harness.models.flight_component import VIEWS

DEST=BASE/"flight-component-source-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("未校正版已核验")
        artifact=ENTRY/"release.joblib"
        sha=json.loads((ENTRY/"scores.json").read_bytes())["权重SHA256"]
        if file_hash(artifact)!=sha:raise ValueError("已发布权重不同")
        verify_entry(DEST,FEATURES,VIEWS[-1],artifact,sha,"component-source",infer_component_source)
