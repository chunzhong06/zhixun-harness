"""按同区间固定配置与现场再校准的最终策略，从源模型重新发放三台的功率。"""
from verify_wind_field import verify, LocalStore
from score_wind_regime import DEST as PRIOR

DEST = PRIOR.parent/"wind-regime-entry"

if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已复测固定分风速对照")
        verify(PRIOR,DEST)
