"""改变训练内容，实发按源机正常转速与本机偏离重新标定的功率。"""
from score_wind_field import run,FIELDS,LocalStore
from score_wind_regime import DEST as PRIOR,load_source
from harness.models.wind_rotor import RotorSurface,RotorRelease,RotorMaintenance,PRIMARY

DEST = PRIOR.parent/"wind-rotor"


def train(rows):
    base = load_source(rows)
    model = RotorSurface().fit(rows[FIELDS].to_numpy(),rows.power_pu.to_numpy())
    return RotorRelease(base.original,base.torque,model)


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("转速偏离开发流已经完成")
        run(DEST,train,RotorMaintenance,PRIMARY,PRIOR)
