"""真实发放开发回执选出的源机固定比例与本机再训练后的完整预测。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,LocalStore
from score_wind_donors import DEST as PRIOR
from harness.models.wind_source_control import SourceControlRelease,PRIMARY

DEST = PRIOR.parent/"wind-source-control"


def train(rows):
    saved = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"] or len(rows)!=saved["源样本"]:
            raise ValueError("源机模型或样本已变")
        f.seek(0)
        return SourceControlRelease(joblib.load(f))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("源机固定对照已发放")
        run(DEST,train,None,PRIMARY,PRIOR)
