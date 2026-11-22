"""冻结真实开发输出选定的源机配比，再用相同起点试验是否需要因本机而改变。"""
import numpy as np
from harness.learning.models.wind.wind_donors import PRIMARY as DONORS,ANCHOR,PREVIOUS_PAPER

CONTROL = "开发冻结各风段源机占比"
PRIMARY = "在分风速固定起点上由本机回标重训源机占比"
PAPER = "论文同工具按风速合用源机"


class SourceControlRelease:
    def __init__(self,previous):
        self.previous = previous

    def predict(self,x):
        return self.previous.predict(x)

    def controller(self):
        return SourceControl(self.previous.controller())


class SourceControl:
    def __init__(self,old):
        self.old = old
        self.mass = np.zeros((3,2,2))
        self.rhs = np.zeros((3,2))
        self.w = np.array([.40,.45,0.])

    def predict_timed(self,t,x,base):
        old = self.old.predict_timed(t,x,base)
        keep = (DONORS,ANCHOR,PREVIOUS_PAPER,"由已返本机功率试验决定转速上限曲线的修正",
                "依过去参考的前向验证更新本机误差曲面","同样工具按全机累计误差分配","三变量提升树源训练",
                "无标签匹配功率曲线","回标匹配功率曲线","冻结匹配源机初始占比")
        out = {n:old[n] for n in keep}
        z = np.digitize(x[:,0],[5.,10.])
        p = old[ANCHOR]
        d = np.column_stack((old["无标签匹配功率曲线"]-p,old["回标匹配功率曲线"]-p))
        frozen = p+self.w[z]*d[:,0]
        out[CONTROL] = frozen
        out["全机冻结一成源校准"] = p+.1*d[:,0]
        out[PAPER] = old[PREVIOUS_PAPER]+self.w[z]*(old["无标签匹配功率曲线"]-old[PREVIOUS_PAPER])
        for n in (.0006,.003,.015):
            delta = np.linalg.solve(self.mass+n*np.eye(2),self.rhs[:,:,None])[:,:,0]
            weight = (delta+np.column_stack((self.w,np.zeros(3)))).clip(0,1)
            weight /= np.maximum(1.,weight.sum(axis=1))[:,None]
            out[f"按分段回标调整源占比先验{n:g}"] = p+(weight[z]*d).sum(1)
        out[PRIMARY] = (frozen+out["按分段回标调整源占比先验0.003"])/2
        self.pending = z[-1],d[-1],frozen[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,d,p = self.pending
        self.mass[z] = .98*self.mass[z]+np.outer(d,d)
        self.rhs[z] = .98*self.rhs[z]+d*(y-p)
