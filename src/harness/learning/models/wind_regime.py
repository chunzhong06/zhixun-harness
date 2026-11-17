"""同一套分风速配比给现场和固定端。现场只用各区间已收到的参考重估配比。"""
import numpy as np
from harness.learning.models.wind_torque import TorqueMaintenance, CombinedWindRelease

CONTROL = "开发冻结三段配比"
PRIMARY = "从相同三段固定配比按本机回标调整"
PAPER = "论文分段标定与转矩按风况固定合用"
TOOLS = tuple(n+"按本机风速重新标定" for n in ("原分段","直接学功率","先学转矩再还原功率"))
GRID = np.array([(a/4,b/4,1-(a+b)/4) for a in range(5) for b in range(5-a)])


class RegimeRelease(CombinedWindRelease):
    """携带同一组源模型和固定的分段安装策略，以便独立进程恢复。"""


class RegimeMaintenance:
    def __init__(self):
        self.old = TorqueMaintenance()
        # 三个固定配比来自已经打过分的三台开发设备。后续新风场须原样冻结。
        self.prior = np.array([[.25,.5,.25],[0.,0.,1.],[1.,0.,0.]])
        self.cost = np.zeros((3,len(GRID)))
        self.pending = None

    def predict(self,x,base):
        out = self.old.predict(x,base)
        bank = np.column_stack([out[n] for n in TOOLS])
        zones = np.digitize(x[:,0],[5,10])
        out[CONTROL] = (bank*self.prior[zones]).sum(1)
        out[PAPER] = np.choose(zones,[bank[:,1],bank[:,2],bank[:,0]])
        history = np.square(GRID[None,:,:]-self.prior[:,None,:]).sum(axis=2)
        for mass in (2,8,32):
            # 保留稀疏区间的固定做法。误差以额定功率比例为单位，差值按真实回标累计。
            criterion = self.cost + mass*.0004*history
            chosen = GRID[np.argmin(criterion,axis=1)]
            out[f"分段回标择权先验{mass}"] = (bank*chosen[zones]).sum(1)
        out[PRIMARY] = (out[CONTROL]+out["分段回标择权先验8"])/2
        self.pending = zones[-1],bank[-1]
        return out

    def update(self,y):
        self.old.update(y)
        z,p = self.pending
        loss = (GRID@p-y)**2
        self.cost[z] = .97*self.cost[z] + loss-loss.min()
