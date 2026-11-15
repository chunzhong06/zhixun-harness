"""从原始三列重发独立源机、最近参考与转速上限的完整开发读法。"""
from score_wind_source_control import DEST as PRIOR
from verify_wind_field import verify,LocalStore
from harness.models.wind_donors import PREVIOUS,PREVIOUS_PAPER,ANCHOR,PRIMARY as DONORS
from harness.models.wind_limit import PRIMARY as LIMIT

DEST = PRIOR.parent/"wind-source-control-entry"

if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("最终源机与参考匹配入口已复核")
        verify(PRIOR,DEST,(PREVIOUS,PREVIOUS_PAPER,ANCHOR,LIMIT,DONORS,"冻结匹配源机初始占比",
                           "无标签匹配功率曲线","回标匹配功率曲线","全机冻结一成源校准"))
