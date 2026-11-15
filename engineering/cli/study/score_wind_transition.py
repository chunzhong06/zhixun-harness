"""沿同一逐窗功率路径检验直接估计与从相邻功率变化递推的完整收益。"""
from score_wind_field import run,FIELDS,LocalStore
from score_wind_history import DEST as PRIOR,load_previous
from harness.models.wind_transition import TransitionRelease,PowerChange,PRIMARY

DEST = PRIOR.parent/"wind-transition"

def train(rows):
    return TransitionRelease(load_previous(PRIOR,rows),PowerChange().fit(rows,FIELDS))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("递推功率已交接")
        run(DEST,train,None,PRIMARY,PRIOR)
