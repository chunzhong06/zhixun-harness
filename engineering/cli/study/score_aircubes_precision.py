"""按源期和现场校准的实际时间差区分浓度变化读法的可信度。"""
from score_aircubes_changes import DEST as PRIOR,evaluate,extend,ADAPTIVE,PREVIOUS,JOINT,DRIFT,STACK,PRIMARY as CHANGED,LocalStore
from harness.models.aircubes_precision import PrecisionRelease,PrecisionMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-precision"


def train(tools,t,x,y):
    tools["precision"]=PrecisionRelease().fit(t,x,y)
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("前向误差条件已经交付")
        evaluate(DEST,PRIOR,train,PrecisionMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,STACK,CHANGED,PRIMARY,"后验漂移模型"},extend)
