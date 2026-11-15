"""从可购买的已知配方里选下一次测量；目标芯片的波形只在购买后返回。"""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


HASH = "既有散列顺序"
COVER = "按配方分散覆盖"
SOURCE = "按源芯片波形分散覆盖"
INFORMATION = "按源信息减少平均不确定度"
BOUNDARY = "按分档边界减少不确定度"
HARD = "按源预测困难程度覆盖"
STRATA = "按目标浓度分层覆盖"
ACTIVE = "根据已测芯片的响应误差决定下一瓶"
RULES = (HASH,COVER,SOURCE,STRATA,INFORMATION,BOUNDARY,HARD,ACTIVE)


def normalized(v):
    return v/max(float(np.mean(v)),1e-12)


class CalibrationQuery:
    def __init__(self, source, composition, index, model):
        self.source = source
        self.y = composition[:,index]
        self.chem = StandardScaler().fit_transform(composition)
        z = StandardScaler().fit_transform(source)
        self.sensor = PCA(n_components=8,whiten=True,svd_solver="full").fit_transform(z)
        self.design = np.column_stack((np.ones(len(z)),self.chem,self.sensor))
        self.cdist = cdist(self.chem,self.chem,"sqeuclidean")
        self.sdist = cdist(self.sensor,self.sensor,"sqeuclidean")
        self.distance = self.cdist/self.chem.shape[1]+self.sdist/self.sensor.shape[1]
        self.cut = np.median(self.y)
        self.band = max(float(np.std(self.y))*.45,1.)
        self.edge = np.exp(-.5*((self.y-self.cut)/self.band)**2)
        self.hard = abs(self.y-model.predict(source))
        self.model = model

    def next(self, rule, used, measured):
        if len(used)!=len(measured) or len(set(used))!=len(used):
            raise ValueError("每次询问必须有一条对应的已付费测量")
        mask = np.ones(len(self.y),bool)
        mask[used] = False
        if not mask.any():raise ValueError("没有剩余标气")
        if rule==HASH:return int(np.flatnonzero(mask)[0])
        if not used:
            return int(np.argmin(abs(self.y-self.cut)))
        if rule==STRATA:
            # 完成一轮后各浓度分层样本数相同，同层优先覆盖相距较远的配方。
            bins = np.searchsorted(np.quantile(self.y,np.arange(1,10)/10),self.y)
            counts = np.bincount(bins[used],minlength=10)
            available_bins = np.unique(bins[mask])
            mask &= counts[bins]==counts[available_bins].min()
            score = self.cdist[:,used].min(1)
        elif rule in (COVER,SOURCE,HARD):
            space = self.sdist if rule==SOURCE else self.cdist
            score = space[:,used].min(1)
            if rule==HARD:score *= .5+normalized(self.hard)
        else:
            # 一瓶已知配方增加一个秩一信息项，算出它对整个配方池平均预测方差的减少量。
            batch = self.design[used]
            a = solve(batch.T@batch+2*np.eye(batch.shape[1]),self.design.T,assume_a="pos").T
            cross = a @ self.design.T
            importance = np.ones(len(self.y))
            if rule in (BOUNDARY,ACTIVE):
                importance = .2+self.edge
            if rule==ACTIVE and len(used)>=4:
                actual = self.model.predict(np.stack(measured))
                # 去除所有配方共有的偏移。真正需要再测的是该芯片偏差变化大的配方区域。
                shift = actual-self.y[used]
                variation = np.abs(shift-np.median(shift))
                distance = self.distance[:,used]
                weight = np.exp(-distance/max(np.median(distance),.01))
                expected = (weight@variation+2*np.median(variation))/(weight.sum(1)+2)
                importance *= .2+normalized(expected)
            score = (cross**2 @ importance)/(1+np.sum(a*self.design,axis=1))
        return int(np.argmax(np.where(mask,score,-np.inf)))


def query_sequence(query,rule,budget,observe):
    """observe 只收单个候选编号，未购读数不作为算法的参数。"""
    for used, measured in query_checkpoints(query, rule, (budget,), observe):
        return used, measured


def query_checkpoints(query,rule,budgets,observe):
    """达到指定预算就交还控制，调用方先发放当前模型，再请求下一瓶。"""
    if not budgets or min(budgets)<=0 or max(budgets)>len(query.y):
        raise ValueError("预算超出可用配方")
    used,measured = [],[]
    for _ in range(max(budgets)):
        idx = query.next(rule,used,measured)
        if idx in used:raise ValueError("重复消耗配方预算")
        reading = np.asarray(observe(idx),dtype=float)
        if reading.shape!=query.source.shape[1:] or not np.isfinite(reading).all():
            raise ValueError("购买的标气读数无效")
        used.append(idx)
        measured.append(reading)
        if len(used) in budgets:
            yield np.array(used),np.stack(measured)
