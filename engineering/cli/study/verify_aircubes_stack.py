"""在四个独立进程逐槽复核源期前向试验的组合权重及现场训练。"""
from verify_aircubes_extended import verify,LocalStore
from score_aircubes_stack import DEST as PRIOR,PRIMARY

DEST=PRIOR.parent/"aircubes-stack-entry"


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("前向试验组合已从原始接口发放")
        verify(PRIOR,DEST,PRIMARY,("装机试验固定六工具权重","六工具固定平均",
               "源端条件与全局固定各半","现场重训条件组合/1","固定现场配对记忆0.95"))
