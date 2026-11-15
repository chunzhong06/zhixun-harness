"""在同一首次共址时间表上，用合法校准样本的浓度覆盖重新估计局部增益。"""
from score_aircubes_stack import DEST as PRIOR,evaluate,extend,ADAPTIVE,PREVIOUS,JOINT,DRIFT,PRIMARY as STACK,LocalStore
from harness.models.aircubes_local_gain import LocalGain,PRIMARY

DEST=PRIOR.parent/"aircubes-local-gain"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("局部增益已交付")
        evaluate(DEST,PRIOR,lambda tools,t,x,y:tools,LocalGain,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,STACK,PRIMARY,"后验漂移模型"},extend)
