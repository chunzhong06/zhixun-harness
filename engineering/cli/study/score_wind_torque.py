"""用相同源期训练与转矩成比例的响应，复现旧对照后再按风速校准。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import DEST as PRIOR, run, LocalStore, FIELDS
from harness.models.wind_torque import TorqueMaintenance, TorqueRelease, CombinedWindRelease, PRIMARY

DEST = PRIOR.parent/"wind-torque"


def train(rows):
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=previous["权重SHA256"]:
            raise ValueError("旧源模型身份变化")
        f.seek(0)
        original = joblib.load(f)
    extra = TorqueRelease().fit(rows[FIELDS].to_numpy(),rows.power_pu.to_numpy())
    return CombinedWindRelease(original,extra)


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已完成转矩代理读法比较")
        run(DEST,train,TorqueMaintenance,PRIMARY,PRIOR)
