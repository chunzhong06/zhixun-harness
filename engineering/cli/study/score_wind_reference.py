"""将最近一次误差扩展为允许重新使用的多条相近条件参考，再逐机同场比较。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,LocalStore
from score_wind_limit import DEST as PRIOR
from harness.models.wind_reference import ReferenceRelease,PRIMARY

DEST = PRIOR.parent/"wind-reference"


def train(rows):
    saved = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"] or len(rows)!=saved["源样本"]:
            raise ValueError("前轮发布和训练来源不符")
        f.seek(0)
        return ReferenceRelease(joblib.load(f))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同风况参考已完成实际发放")
        run(DEST,train,None,PRIMARY,PRIOR)
