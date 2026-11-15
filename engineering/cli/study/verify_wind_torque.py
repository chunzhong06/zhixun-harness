"""用装好的复合权重逐机复核功率训练与转矩训练的完整交付。"""
from verify_wind_field import verify, LocalStore
from score_wind_torque import DEST as PRIOR

DEST = PRIOR.parent/"wind-torque-entry"

if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("转矩版本原始部署已复核")
        verify(PRIOR,DEST)
