"""在相同的装机权重和反馈预算上，用不同的漂移状态假设直接推断本机校正系数。"""
from score_aircubes_joint import DEST as PRIOR,extend,evaluate,LocalStore,ADAPTIVE,PREVIOUS,PRIMARY as JOINT
from harness.models.aircubes_bias import DriftMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-drift"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("漂移滤波已有实发")
        evaluate(DEST,PRIOR,lambda tools,t,x,y:tools,DriftMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,PRIMARY,"后验漂移模型"},extend)
