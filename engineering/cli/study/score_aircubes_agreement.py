"""把两路仪器相对漂移的无标签证据接回当前最强交付，检验合法稀疏反馈的额外作用。"""
from score_aircubes_bias import DEST as PRIOR,extend,evaluate,LocalStore,ADAPTIVE,PREVIOUS,JOINT,PRIMARY as DRIFT
from harness.models.aircubes_normalize import AgreementRelease,AgreementMaintenance,PRIMARY

DEST=PRIOR.parent/"aircubes-agreement"


def install(tools,t,x,y):
    tools["agreement"]=AgreementRelease().fit(t,x,y,tools["joint"])
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("双电极相对漂移已有交付")
        evaluate(DEST,PRIOR,install,AgreementMaintenance,PRIMARY,
                 ADAPTIVE|{PREVIOUS,JOINT,DRIFT,"后验漂移模型",PRIMARY,"按已回平均误差合用偏移"},extend)
