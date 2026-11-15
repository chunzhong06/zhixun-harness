"""在相同可用字段、相同合法回标下训练和维护同行评议的概率网络。"""
from hashlib import file_digest
import json
import joblib
from score_wind_field import run,FIELDS,LocalStore
from score_wind_source_control import DEST as PRIOR
from harness.models.wind_pmlp import ProbabilityNetwork,PmlpRelease,PRIMARY

DEST = PRIOR.parent/"wind-pmlp"


def train(rows):
    saved = json.loads((PRIOR/"scores.json").read_bytes())
    with (PRIOR/"kelmarsh.joblib").open("rb") as f:
        if file_digest(f,"sha256").hexdigest()!=saved["权重SHA256"] or len(rows)!=saved["源样本"]:
            raise ValueError("源机基线预装身份变化")
        f.seek(0)
        previous = joblib.load(f)
    network = ProbabilityNetwork().train(rows[FIELDS].to_numpy(),rows.power_pu.to_numpy(),rows.clock.to_numpy())
    print(json.dumps({"论文源期预训练":{"所选轮数":network.epochs,"训练样本":len(rows)}},ensure_ascii=False))
    return PmlpRelease(previous,network)


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同输入概率网络已有真实发放")
        run(DEST,train,None,PRIMARY,PRIOR)
