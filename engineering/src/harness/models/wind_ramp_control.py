"""冻结由开发数据选定的升降风况配比，实发同工具的固定对照及本机再训练。"""
import numpy as np
from harness.models.wind_history import HISTORY_EFFECTS,CONTROL,PRIMARY as HISTORY_SITE
from harness.models.wind_mapping import PRIMARY as PREVIOUS,ANCHOR,OLD_PAPER

FIXED = "按当前风速与升降方向固定历史修正"
PRIMARY = "由本机已返功率试验修订风况方向固定系数"
PAPER = "相同升降规则与连续风况论文组合"
WEIGHTS = np.array([[0.,.75],[0.,.25],[.25,.5],[.75,.25],[.75,0.],[.75,0.],
                    [.25,-.25],[0.,-.25],[0.,-.25],[.75,-.25]])


class RampRelease:
    def __init__(self,old):
        self.old = old

    def predict(self,x):
        return self.old.predict(x)

    def controller(self):
        return RampMaintenance(self.old.controller())


class RampMaintenance:
    def __init__(self,previous):
        self.old = previous
        self.last = None
        self.g = np.zeros((10,2,2))
        self.rhs = np.zeros((10,2))

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base)
        keep = (PREVIOUS,ANCHOR,OLD_PAPER,HISTORY_SITE,"三变量提升树源训练","同样工具按全机累计误差分配")
        out = {n:old[n] for n in keep}
        v = x[:,0]
        rt,rv = self.last if self.last is not None else (t[0]-1200,v[0])
        change = np.diff(v,prepend=rv)
        zone = 3*np.digitize(v,[5,10])+np.digitize(change,[-.4,.4])
        zone[np.diff(t,prepend=rt)!=600] = 9
        p = old[CONTROL]
        d = np.column_stack([old[n]-p for n in HISTORY_EFFECTS])
        fixed = p+(WEIGHTS[zone]*d).sum(1)
        out[FIXED] = fixed
        out[PAPER] = fixed+(old[OLD_PAPER]-old[ANCHOR])
        for penalty in (.0005,.003,.02):
            extra = np.linalg.solve(self.g+penalty*np.eye(2),self.rhs[:,:,None])[:,:,0].clip(-.75,.75)
            out[f"风况分格回标后修正先验{penalty:g}"] = fixed+(d*extra[zone]).sum(1)
        out[PRIMARY] = (out["风况分格回标后修正先验0.003"]+fixed)/2
        self.last = t[-1],v[-1]
        self.pending = zone[-1],d[-1],fixed[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,d,p = self.pending
        self.g[z] = .98*self.g[z]+np.outer(d,d)
        self.rhs[z] = .98*self.rhs[z]+d*(y-p)
