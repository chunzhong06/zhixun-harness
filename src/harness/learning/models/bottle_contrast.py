"""以配方成分相近、目标浓度不同的标样对学习局部校准方向。"""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from harness.learning.models.calibration_query import CalibrationQuery, BOUNDARY
from harness.learning.models.paired_design import QueryDeployment, CHOICE

ACTIVE = "保留全量程并购买能区分干扰的对照瓶"
FROZEN = "按源配方购买目标浓度相反的对照瓶"
PRIMARY = "根据成对装机试验学习本机分档方向"
PAPER = "论文续训与标气对照学习各半"


class ContrastQuery(CalibrationQuery):
    def __init__(self, source, composition, index, model):
        super().__init__(source, composition, index, model)
        self.nuisance = np.delete(self.chem, index, axis=1)
        self.standard = max(1., np.std(self.y))

    def next(self, rule, used, measured):
        if len(used)<8 or len(used)%3 != 2:
            return super().next(BOUNDARY,used,measured)
        # 在已购样本上估计哪种干扰对应当前芯片的误差；只以配方已知成分匹配未购瓶。
        z = self.nuisance[used]
        if rule == ACTIVE:
            error = (self.model.predict(np.stack(measured))-self.y[used])/self.standard
        else:
            error = (self.model.predict(self.source[used])-self.y[used])/self.standard
        gain = solve(z.T@z+8*np.eye(z.shape[1]), z.T@(error-error.mean()), assume_a="pos")
        scale = np.sqrt(1+np.minimum(np.abs(gain)*4,4))
        dist = cdist(self.nuisance*scale,z*scale,"sqeuclidean")/z.shape[1]
        across = (self.y[:,None]>self.cut)!=(self.y[used]>self.cut)
        contrast = np.minimum(1., np.abs(self.y[:,None]-self.y[used])/self.standard)
        pairs = across*contrast*np.exp(-dist)*(.3+self.edge[used])
        score = pairs.max(1)*(.3+self.edge)*np.sqrt(self.distance[:,used].min(1))
        score[used] = -np.inf
        return int(np.argmax(score))


class PairRegression:
    def fit(self, source, composition, ix, observed, j):
        y = composition[:,j]
        self.cut = np.median(y)
        self.scale = StandardScaler().fit(source)
        a = self.scale.transform(source).clip(-8,8)
        b = self.scale.transform(observed).clip(-8,8)
        # 相同标气提供偏移估计。两个器件中的训练样本分别构造差值，不制造跨器件标签对。
        self.offset = np.mean(a[ix]-b, axis=0)
        b = b+self.offset
        nuisance = StandardScaler().fit_transform(np.delete(composition,j,axis=1))
        self.x = np.vstack((a,b))
        kernel = self.kernel(self.x)
        coefficient = np.zeros_like(kernel)
        rhs = np.zeros(len(self.x))
        labels = np.r_[y, y[ix]]
        start = 0
        for z, yy, mass in ((nuisance,y,1.),(nuisance[ix],y[ix],2.)):
            n = len(yy)
            distances = cdist(z,z,"sqeuclidean")/z.shape[1]
            across = (yy[:,None]>self.cut)!=(yy>self.cut)
            middle = np.exp(-abs(yy-self.cut)/max(1.,np.std(y)*.5))
            pairs = np.exp(-distances)*(0.2+across)*(.3+middle[:,None]+middle[None,:])
            np.fill_diagonal(pairs,0.)
            # 局部成对差拟合梯度，绝对标气给出共同截距。各器件按其真实样本量计入。
            lap = np.diag(pairs.sum(1))-pairs
            local = mass*(np.diag(.5+middle)+lap/max(1.,pairs.sum(1).mean()))
            sl = slice(start,start+n)
            coefficient[sl,sl] = local
            rhs[sl] = local@(labels[sl]-self.cut)
            start += n
        self.dual = solve(coefficient@kernel+.10*np.eye(len(kernel)),rhs,assume_a="gen")
        return self

    def kernel(self, x):
        gram = x@self.x.T/self.x.shape[1]
        return 1+gram+.3*gram**2

    def predict(self, x):
        v = self.scale.transform(x).clip(-8,8)+self.offset
        return self.cut+self.kernel(v)@self.dual


class ContrastDeployment:
    def fit(self, src, chem, ix, observed, j):
        self.base = QueryDeployment().fit(src,chem,ix,observed,j)
        self.trained = PairRegression().fit(src,chem,ix,observed,j)
        fold = np.arange(len(ix))%4
        yy = chem[ix,j]
        differences, errors = [], []
        for k in range(4):
            test = fold==k
            keep = np.ones(len(src),bool)
            keep[ix[test]] = False
            local = np.cumsum(keep)[ix[~test]]-1
            reg = PairRegression().fit(src[keep],chem[keep],local,observed[~test],j)
            old = QueryDeployment().fit(src[keep],chem[keep],local,observed[~test],j)
            initial = old.predict(observed[test])[CHOICE]
            differences.extend(reg.predict(observed[test])-initial)
            errors.extend(yy[test]-initial)
        d, e = np.array(differences), np.array(errors)
        penalty = 2*np.median(d*d)+1
        self.weight = float(np.clip((d@e+.5*penalty)/(d@d+penalty),0,1))
        return self

    def predict(self, x):
        out = self.base.predict(x)
        p = out[CHOICE]
        local = self.trained.predict(x)
        out["只用成对标气学习浓度"] = local
        for w in (.25,.5,.75):
            out[f"原装机与标气对照占{w:g}"] = p+w*(local-p)
        out[PAPER] = (out["单秒论文续训"]+local)/2
        out[PRIMARY] = p+self.weight*(local-p)
        return out
