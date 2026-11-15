"""用相同的三份源曲线比较固定源机、无标签源机匹配与实际回标选择。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,FIELDS,LocalStore
from score_wind_reference import DEST as PRIOR
from harness.models.wind_donors import SourceMachines,DonorRelease,PRIMARY

DEST = PRIOR.parent/"wind-donors"


def train(rows):
    saved = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"]:
            raise ValueError("历史参考模型身份变化")
        f.seek(0)
        base = joblib.load(f)
    return DonorRelease(base,SourceMachines().fit(rows,FIELDS))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("源机现场迁移结果已经交付")
        run(DEST,train,None,PRIMARY,PRIOR)
