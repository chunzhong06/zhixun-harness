"""在独立进程上重建源网络与目标微调状态，检查实际均值预测及全部强对手。"""
from score_wind_pmlp import DEST as PRIOR
from verify_wind_field import verify,LocalStore
from harness.models.wind_pmlp import PAPER
from harness.models.wind_source_control import PRIMARY as PREVIOUS,CONTROL, PAPER as PREVIOUS_PAPER

DEST = PRIOR.parent/"wind-pmlp-entry"

if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("概率网络原始部署已经复核")
        verify(PRIOR,DEST,(PREVIOUS,CONTROL,PREVIOUS_PAPER,PAPER,"论文A1源网络","论文A1只微调输出层",
                           "论文按输出标准差固定合用","论文按风况固定合用"))
