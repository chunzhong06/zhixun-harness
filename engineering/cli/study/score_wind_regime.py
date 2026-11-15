"""逐行发放从开发回执选出的固定配比，给现场相同的起点与全部反馈。"""
import json
from hashlib import file_digest
import joblib
from score_wind_field import run, LocalStore
from score_wind_torque import DEST as PRIOR
from harness.models.wind_regime import RegimeMaintenance,RegimeRelease,PRIMARY

DEST = PRIOR.parent/"wind-regime"


def load_source(train):
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=previous["权重SHA256"] or len(train)!=previous["源样本"]:
            raise ValueError("实际安装来源变化")
        f.seek(0)
        installed = joblib.load(f)
        return RegimeRelease(installed.original,installed.torque)


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("按区间加强的固定策略已发放")
        run(DEST,load_source,RegimeMaintenance,PRIMARY,PRIOR)
