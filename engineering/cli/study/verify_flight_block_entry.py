"""从原生单航班输入重发每个16行平均读法的结果，并发布仅含所需模型的权重。"""
import json
import joblib
from score_flight_block import DEST as PRIOR, BASE, LocalStore, file_hash, MEAN
from verify_flight_segment_entry import verify_entry
from engineering.cli.workflow.flight_input_predict import infer_block_mean

DEST=BASE/"flight-block-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("块均值权重已有实际交付")
        meta=json.loads((PRIOR/"scores.json").read_bytes())
        trained=PRIOR/"release.joblib"
        if file_hash(trained)!=meta["发布权重SHA256"]:raise ValueError("训练权重身份变化")
        pack=joblib.load(trained);f=DEST/"release.joblib"
        joblib.dump({"extractor":pack["extractor"],"reader":pack["readers"].mean},f)
        verify_entry(DEST,PRIOR,MEAN,f,file_hash(f),"component-block-mean",infer_block_mean)
