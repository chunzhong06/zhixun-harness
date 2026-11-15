"""相同完整风机输入下，按转速上限拟合两段传感器响应并实发。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,FIELDS,LocalStore
from score_wind_persistence import DEST as PRIOR
from harness.models.wind_limit import ControlCurve,LimitRelease,PRIMARY

DEST = PRIOR.parent/"wind-limit"


def train(rows):
    saved = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"]:
            raise ValueError("回标持续性基线身份变化")
        f.seek(0)
        earlier = joblib.load(f)
    return LimitRelease(earlier,ControlCurve().fit(rows[FIELDS].to_numpy(),rows.power_pu.to_numpy()))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("转速上限装机已实发")
        run(DEST,train,None,PRIMARY,PRIOR)
