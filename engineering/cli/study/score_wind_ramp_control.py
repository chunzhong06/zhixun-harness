"""把可以预先写定的升降风况完整发放，再测本机因地修订它的作用。"""
from score_wind_field import run,LocalStore
from score_wind_history import load_previous
from score_wind_mapping import DEST as PRIOR
from harness.models.wind_ramp_control import RampRelease,PRIMARY

DEST = PRIOR.parent/"wind-ramp-control"

if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同一分段升降读法已有实际评分")
        run(DEST,lambda rows:RampRelease(load_previous(PRIOR,rows)),None,PRIMARY,PRIOR)
