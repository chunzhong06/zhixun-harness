"""交替购买锚定量程的标气和检验上次误差的标气，并用逐瓶预测误差校正输出。"""
import numpy as np
from scipy.linalg import solve

from harness.models.calibration_query import CalibrationQuery, BOUNDARY
from harness.models.bottle_alignment import AlignmentDeployment, PRIMARY as ALIGNED
from harness.models.paired_design import CHOICE

ACTIVE = "保留量程再按已购标气误差选瓶"
FROZEN = "保留量程再按源端误差选瓶"
PRIMARY = "按装机已知浓度校正分界附近的误差"
PAPER = "论文配准和分界校准各半"


class BalancedQuery(CalibrationQuery):
    def next(self, rule, used, measured):
        anchor = super().next(BOUNDARY, used, measured)
        # 以原强固定设计启动；之后每四瓶最多一次定向询问，其他照顾完整配方空间。
        if len(used)<8 or len(used)%4 != 3:
            return anchor
        design = self.design[used]
        inverse = solve(design.T@design+2*np.eye(design.shape[1]), self.design.T, assume_a="pos").T
        leverage = np.sum(inverse*self.design, axis=1)
        chosen = self.y[used] > self.cut
        need_high = chosen.sum() < len(chosen)/2
        if rule == ACTIVE:
            response = self.model.predict(np.stack(measured))
        else:
            response = self.model.predict(self.source[used])
        err = response-self.y[used]
        sim = np.exp(-self.distance[:,used]/max(.01, np.median(self.distance)))
        expected = (sim@np.abs(err)+2*np.median(np.abs(err)))/(sim.sum(1)+2)
        score = (.2+self.edge)*leverage*(.5+np.minimum(3,expected/self.band))
        score[used] = -np.inf
        available = np.isfinite(score) & ((self.y>self.cut)==need_high)
        if not available.any():
            return anchor
        return int(np.argmax(np.where(available, score, -np.inf)))


class BoundaryCalibration:
    def fit(self, p, y, cut, sigma):
        # 两端的点锚定范围，中间的点单独估计局部偏差。真值均来自装机已回标的配方。
        self.cut = cut
        self.sigma = max(1., float(sigma))
        f = self.features(p)
        ridge = np.diag([3., 8., 5., 7.])
        self.coef = solve(ridge+f.T@f, f.T@(y-p), assume_a="pos")
        return self

    def features(self, estimates):
        z = np.clip((estimates-self.cut)/self.sigma,-4,4)
        around = np.exp(-.5*(z/.45)**2)
        return np.column_stack((np.ones(len(z)), z, around, z*around))

    def predict(self, estimates):
        return estimates+self.features(estimates)@self.coef


class ChallengeDeployment:
    def fit(self, src, chem, ix, measured, j):
        self.aligned = AlignmentDeployment().fit(src, chem, ix, measured, j)
        # 安装后使用源端已知配方，按整瓶同时从源端与目标端移出，再取得真实留出误差。
        from harness.models.bottle_alignment import PairedWindows
        fold = np.arange(len(ix))%4
        held = np.empty(len(ix))
        y = chem[:,j]
        for k in range(4):
            test = fold==k
            keep = np.ones(len(src), bool)
            keep[ix[test]] = False
            local = np.cumsum(keep)[ix[~test]]-1
            model = PairedWindows().fit(src[keep], y[keep], local, measured[~test])
            held[test] = model.predict(measured[test]).mean(1)
        self.cal = BoundaryCalibration().fit(held, y[ix], float(np.median(y)), np.std(y))
        return self

    def predict(self, x):
        out = self.aligned.predict(x)
        aligned = self.aligned.model.predict(x).mean(1)
        adjusted = self.cal.predict(aligned)
        out["对已配准读数只做分界校准"] = adjusted
        for w in (.25,.5,.75):
            out[f"原装机与分界校准读数占{w:g}"] = out[CHOICE] + w*(adjusted-out[CHOICE])
        out[PAPER] = (out["论文单秒续训与逐秒配准各半"]+adjusted)/2
        out[PRIMARY] = out[ALIGNED]+.5*(adjusted-aligned)
        return out
