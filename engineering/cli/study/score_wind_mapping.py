"""比较目标机半日与一周同转速参考风速的漂移、固定修正与回标选择。"""
from score_wind_field import run,FIELDS,LocalStore
from score_wind_history import load_previous
from score_wind_dynamic_head import DEST as PRIOR
from harness.models.wind_mapping import InputMap,MappingRelease,PRIMARY

DEST = PRIOR.parent/"wind-mapping"

def train(rows):
    return MappingRelease(load_previous(PRIOR,rows),InputMap().fit(rows,FIELDS))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("运行曲线漂移已交付")
        run(DEST,train,None,PRIMARY,PRIOR)
