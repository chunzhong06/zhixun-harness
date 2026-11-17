"""用成对标气校准每路加热波形，预测时只读取当前目标芯片的波形。"""
import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler

from harness.learning.models.calibration_query import CalibrationQuery
from harness.learning.models.paired_design import QueryDeployment, CHOICE


ACTIVE = "用已测芯片的波形偏差选择下一瓶"
FROZEN = "按源波形和分档边界选择校准瓶"
PRIMARY = "用逐瓶误差选择波形校准并发布"
PAPER = "论文单秒续训与逐秒配准各半"
TOOLS = ("同秒偏移", "同秒灵敏度", "相邻五秒联合配准")


class AlignmentQuery(CalibrationQuery):
    def next(self, rule, used, measured):
        if len(used) != len(measured):
            raise ValueError("所给波形与已购买配方数不一致")
        if not used:
            return int(np.argmin(abs(self.y-self.cut)))
        # 源端已知各瓶配方与波形。目标响应的差异只能由已实际返回的瓶推算。
        diversity = self.distance[:, used].min(axis=1)
        priority = (.30 + self.edge) * np.sqrt(diversity)
        if len(used) >= 6 and rule == ACTIVE:
            predicted = self.model.predict(np.stack(measured))
            residual = predicted - self.model.predict(self.source[used])
            distance = self.distance[:, used]
            scale = max(.01, float(np.median(self.distance)))
            nearby = np.exp(-distance/scale)
            shift = nearby @ residual / np.maximum(nearby.sum(1), 1e-10)
            dispersion = nearby @ (residual-np.median(residual))**2
            dispersion = np.sqrt(dispersion / np.maximum(nearby.sum(1), 1e-10))
            crossing = np.exp(-np.abs(self.y+shift-self.cut)/self.band)
            priority *= (.4+crossing) * (1+np.minimum(dispersion/self.band, 3))
        priority[used] = -np.inf
        return int(np.argmax(priority))


def neighbors(z):
    # 四路独立，各秒的均值、斜率分别配对，两端复制自身以免越过波形边界。
    cubes = z.reshape(-1, 4, 144, 2)
    padded = np.pad(cubes, ((0,0), (0,0), (2,2), (0,0)), mode="edge")
    return np.stack([padded[:,:,i:i+144,:].reshape(len(z), -1) for i in range(5)], axis=-1)


class PairedWindows:
    def fit(self, source, y, ix, measured):
        self.scale = StandardScaler().fit(source)
        z = self.scale.transform(source)
        target = self.scale.transform(measured)
        self.center = target.mean(0)
        self.offset = (z[ix]-target).mean(0)
        unaligned = target-self.center
        residual = z[ix] - target - self.offset
        self.slope = (unaligned*residual).sum(0) / (np.sum(unaligned**2, axis=0)+3)
        self.slope = np.clip(self.slope, -.9, 2.)
        design = neighbors(unaligned)
        gram = np.einsum("ndi,ndj->dij", design, design)
        rhs = np.einsum("ndi,nd->di", design, residual)
        self.local = np.linalg.solve(gram + 4*np.eye(5), rhs[:,:,None])[:,:,0]
        self.z = z.clip(-8, 8)
        self.mean = np.mean(y)
        kernel = self.kernel(self.z)
        # 校准浓度分界附近的误差，在源期仍保留完整浓度范围。
        radius = max(1., np.std(y)*.4)
        weight = 1+2*np.exp(-.5*((y-np.median(y))/radius)**2)
        self.dual = solve(kernel+np.diag(.025/weight), y-self.mean, assume_a="pos")
        return self

    def kernel(self, z):
        gram = z @ self.z.T / self.z.shape[1]
        return gram + .3 * gram**2

    def predict(self, raw):
        z = self.scale.transform(raw)
        centered = z-self.center
        translated = z+self.offset
        nearby = np.einsum("ndk,dk->nd", neighbors(centered), self.local)
        views = (translated, translated+self.slope*centered, translated+nearby)
        return np.column_stack([self.mean+self.kernel(v.clip(-8,8))@self.dual for v in views])


class AlignmentDeployment:
    def fit(self, src, chem, ix, measured, j):
        self.previous = QueryDeployment().fit(src, chem, ix, measured, j)
        y = chem[:,j]
        cut = float(np.median(y))
        fold = np.arange(len(ix))%4
        held = np.empty((len(ix), len(TOOLS)))
        for k in range(4):
            test = fold == k
            keep = np.ones(len(src), bool)
            keep[ix[test]] = False
            local = np.cumsum(keep)[ix[~test]]-1
            trained = PairedWindows().fit(src[keep], y[keep], local, measured[~test])
            held[test] = trained.predict(measured[test])
        error = (held-y[ix,None])/max(float(np.std(y)), 1.)
        weight = .35+np.exp(-np.abs(y[ix]-cut)/max(1., np.std(y)*.3))
        losses = np.average((held>cut)!=(y[ix,None]>cut), axis=0, weights=weight)
        losses += np.average(np.minimum(error**2, 4), axis=0, weights=weight)
        self.weights = np.exp(-8*(losses-losses.min()))
        self.weights /= self.weights.sum()
        self.model = PairedWindows().fit(src, y, ix, measured)
        return self

    def predict(self, x):
        out = self.previous.predict(x)
        bank = self.model.predict(x)
        for name, column in zip(TOOLS, bank.T, strict=True):
            out[name] = column
            for fraction in (.25, .5, .75):
                out[f"原装机与{name}占{fraction:g}"] = (1-fraction)*out[CHOICE]+fraction*column
        out[PAPER] = (out["单秒论文续训"]+bank[:,-1])/2
        out["固定等权波形配准与原装机各半"] = (out[CHOICE]+bank.mean(1))/2
        out[PRIMARY] = (out[CHOICE]+bank@self.weights)/2
        return out
