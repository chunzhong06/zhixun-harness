"""给同机两个NO探头和电源信号相同的输入权限，复核新增输入真正改变哪些预测。"""
import numpy as np
import pandas as pd
from score_aircubes import known
from score_aircubes_transfer import DEST as PRIOR,evaluate,LocalStore,ADAPTIVE,PRIMARY as PREVIOUS
from harness.models.aircubes_joint import AUX,PRIMARY,JointRelease,JointMaintenance

DEST=PRIOR.parent/"aircubes-joint"


def extend(site,t,x,manifest):
    frame=pd.read_csv(known(site+".csv",manifest),usecols=["timestamp",*AUX])
    if not np.array_equal(frame["timestamp"],t):raise ValueError("共址辅助通道不同步")
    extra=frame[list(AUX)].to_numpy()
    if not np.isfinite(extra[np.isfinite(x).all(1)]).all():raise ValueError("存在无辅助通道的服务时刻，需预先规定降级后再评分")
    return np.column_stack((x,extra))


def install(tools,t,x,y):
    tools["joint"]=JointRelease().fit(t,x,y)
    return tools


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("同机交叉气体的校准已发放")
        evaluate(DEST,PRIOR,install,JointMaintenance,PRIMARY,ADAPTIVE|{PRIMARY,PREVIOUS},extend)
