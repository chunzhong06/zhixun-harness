"""只从完整可对齐的源期过去窗口学习浓度增量，部署后按两个已返区间更新。"""
from score_aircubes_stack import DEST as PRIOR,evaluate,extend,ADAPTIVE,PREVIOUS,JOINT,DRIFT,PRIMARY as STACK,LocalStore
from harness.models.aircubes_changes import ChangeRelease,ChangeMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-changes"


def train(tools,t,x,y):
    tools["changes"]=ChangeRelease().fit(t,x,y)
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("电极差分试验已经交付")
        evaluate(DEST,PRIOR,train,ChangeMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,STACK,PRIMARY,"后验漂移模型"},extend)
