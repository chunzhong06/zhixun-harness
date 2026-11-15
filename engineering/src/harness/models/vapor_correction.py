"""用已交付后的参考读数估计误差能延续多久，以阈值为单位校正下一窗。"""
from collections import deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from harness.models.vapor_boundary import decision_context, measured_views, PRIMARY as MIXED
from harness.models.vapor_field import ProbeEncoder, voltage_ratio

PAPER = "论文储备池/压缩电压"
PRIMARY = "按现场已回误差选择校正强度"
FIXED = ("原样交付", "固定慢变偏差", "保留最近误差", "快速衰减", "按源场学习误差延续", "源场校正各半")


class ResidualHistory:
    def __init__(self):
        self.recent = deque(maxlen=12)

    def context(self, estimate, age):
        if self.recent:
            last, anchor, change = self.recent[-1]
            center = np.median([item[0] for item in self.recent])
        else:
            last = anchor = change = center = 0.
        return np.array([last, center, anchor, estimate, estimate-anchor,
                         age/10, change, min(len(self.recent),10)/10])

    def observe(self, estimate, raw):
        target = np.arcsinh(raw)
        error = target-estimate
        previous = self.recent[-1][0] if self.recent else error
        self.recent.append((error, estimate, error-previous))


class Corrections:
    def fit(self, rows, truth):
        # 留整场训练数据。压缩域的差值减弱高峰在学习误差延续性时的影响。
        self.model = HistGradientBoostingRegressor(max_iter=90, max_leaf_nodes=9,
            min_samples_leaf=32, learning_rate=.07, l2_regularization=3., random_state=73)
        self.model.fit(rows, np.clip(truth,-1.5,1.5))
        return self

    def offsets(self, rows):
        last, center = rows[:,0], rows[:,1]
        age = rows[:,5]*10
        learned = self.model.predict(rows)
        return np.column_stack((np.zeros(len(rows)), center, last, last*np.exp(-age/3),
                                learned, .5*learned))


def predictions(estimate, offsets):
    return np.sinh(np.clip(estimate[:,None]+offsets, -5, 5))


def losses(bank, truth):
    diff = (np.arcsinh(bank)-np.arcsinh(truth[:,None]))**2
    return ((bank>=1)!=(truth[:,None]>=1)).astype(float)+.45*np.minimum(diff,1.)


class VaporService:
    """每次只接收本半秒慢探头波形。十窗发出后才能用对应末窗的参考值。"""
    def __init__(self, tools, warm, baseline_pid, initial_pid):
        self.cut = tools["cut"]
        self.baseline_pid = baseline_pid
        self.models, self.mixer = tools["boundary"]
        self.corrections = tools["correction"]
        base = np.median(voltage_ratio(np.vstack(warm)),axis=0)
        self.encoder = ProbeEncoder(base,tools["scale"])
        for group in warm:
            self.encoder.next(group)
        self.anchor = (initial_pid-baseline_pid)/self.cut
        self.history = ResidualHistory()
        self.prior_loss = tools["initial_loss"]
        self.field_loss = np.zeros_like(self.prior_loss)
        self.mass = 0.
        self.age = 0
        self.position = 20

    def measure(self, group):
        if self.age==10:
            raise ValueError("上批回标尚未处理")
        view,_ = self.encoder.next(group)
        x = measured_views([np.atleast_2d(a) for a in view], np.array([0]))
        bank = np.column_stack([m.predict(a) for m,a in zip(self.models,x,strict=True)])
        self.age += 1
        ctx = decision_context(bank,self.anchor)
        ctx[:,-1] = self.age/10
        mixed = self.mixer.predict(bank,ctx)[0]
        paper = bank[0,5]
        estimate = np.arcsinh(paper)
        row = self.history.context(estimate,self.age)
        offsets = self.corrections.offsets(row[None,:])
        corrected = predictions(np.array([estimate]),offsets)[0]
        # 每一项固定对照用完全相同的源模型、现场读数与回标。
        cost = (4*self.prior_loss+self.field_loss)/(4+self.mass)
        weight = np.exp(-35*(cost-cost.min()))
        final = np.sinh(estimate+np.average(offsets[0],weights=weight))
        installed = np.exp(-35*(self.prior_loss-self.prior_loss.min()))
        source_mix = np.sinh(estimate+np.average(offsets[0],weights=installed))
        self.last = x,bank,ctx,estimate,corrected
        pos = self.position
        self.position += 1
        return pos, {MIXED:float(mixed),PAPER:float(paper),PRIMARY:float(final),
                     "固定源场校正权重":float(source_mix),
                     **dict(zip(FIXED,map(float,corrected),strict=True))}

    def feedback(self, pos, raw_pid):
        if self.age!=10 or pos!=self.position-1 or not np.isfinite(raw_pid):
            raise ValueError("只接收上个已交付完整批次的末窗参考电压")
        x,bank,ctx,estimate,corrected = self.last
        self.anchor = (raw_pid-self.baseline_pid)/self.cut
        target = np.array([self.anchor])
        self.history.observe(estimate,self.anchor)
        self.field_loss = .87*self.field_loss+losses(corrected[None,:],target)[0]
        self.mass = .87*self.mass+1
        self.mixer.update(bank,ctx,target)
        for model,features in zip(self.models,x,strict=True):
            model.update(features,target)
        self.age = 0
