"""在源机上试验同风况误差能延续多久，测试有限回标在后续实际风机上的作用。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,FIELDS,LocalStore
from score_wind_rotor import DEST as PRIOR
from harness.models.wind_persistence import ErrorPersistence,PersistenceRelease,PRIMARY

DEST = PRIOR.parent/"wind-persistence"


def train(rows):
    previous = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=previous["权重SHA256"]:
            raise ValueError("此前实际校准模型身份不符")
        f.seek(0)
        rotor = joblib.load(f)
    return PersistenceRelease(rotor,ErrorPersistence().fit(rows,FIELDS))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("误差持续性已在开发机发放")
        run(DEST,train,None,PRIMARY,PRIOR)
