"""依据无断档的源端历史训练并核查其在已安装模型的逐机功率交接上的贡献。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,FIELDS,LocalStore
from score_wind_pmlp import DEST as PRIOR
from harness.models.wind_history import HistoryRelease,Dynamics,PRIMARY

DEST = PRIOR.parent/"wind-history"


def load_previous(folder,rows):
    saved = json.loads((folder/"scores.json").read_bytes())
    with (folder/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"] or len(rows)!=saved["源样本"]:
            raise ValueError("源机训练身份或模型已改变")
        f.seek(0)
        return joblib.load(f)


def train(rows):
    return HistoryRelease(load_previous(PRIOR,rows),Dynamics().fit(rows,FIELDS))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("历史风况模型已有实际发放")
        run(DEST,train,None,PRIMARY,PRIOR)
