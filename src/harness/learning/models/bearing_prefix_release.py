"""对每台设备保存独立的振动前缀；安装权重不含当前设备的寿命或结束时刻。"""
from collections import deque
import numpy as np
from sklearn.base import clone
from harness.learning.models.bearing_wear_rate import WearRate


class BearingPrefixRelease:
    def fit(self, initial, completed, template):
        self.initial_lives=np.array([u.durations["calendar"] for u in initial])
        y=np.concatenate([1-u.ages["calendar"]/u.durations["calendar"] for u in completed])
        self.phase=clone(template).fit(np.vstack([u.spectral for u in completed]),y)
        self.scale=np.median([u.durations["calendar"] for u in completed])
        self.rate=WearRate().fit(completed)
        return self

    def stream(self):
        return BearingPrefixStream(self)


class BearingPrefixStream:
    def __init__(self, model):
        self.model=model
        self.first=[]
        self.recent=deque(maxlen=5)
        self.amplitudes=deque(maxlen=3)
        self.smooth=deque(maxlen=80)
        self.elapsed=deque(maxlen=80)
        self.start=self.last=self.cadence=None
        self.active=0.

    def predict(self, spectrum, timestamps):
        raw=np.asarray(spectrum,dtype=float)
        stamps=np.asarray(timestamps,dtype=float)
        if raw.shape!=(200,) or not np.isfinite(raw).all() or np.any(raw<0):
            raise ValueError("每个测点需200个非负有限频谱幅度")
        if (stamps.ndim!=1 or not len(stamps) or not np.isfinite(stamps).all()
                or np.any(np.diff(stamps)<=0) or (self.last is not None and stamps[0]<=self.last)):
            raise ValueError("时间块必须从上一测点之后按时序延伸到当前测量")
        intervals=np.diff(stamps if self.last is None else np.r_[self.last,stamps])
        if self.start is None:self.start=float(stamps[0])
        if len(intervals) and self.cadence is None:self.cadence=float(intervals[0])
        if len(intervals):self.active+=float(np.minimum(intervals,2*self.cadence).sum())/60
        self.last=stamps[-1]
        calendar=(self.last-self.start)/60
        bands=raw.reshape(20,10).max(1)
        log=np.log1p(bands)
        if len(self.first)<8:self.first.append(log)
        self.recent.append(log)
        change=np.median(self.recent,axis=0)-np.median(self.first,axis=0)
        signal=float(np.quantile(change,.8))
        m=self.model
        lives=m.initial_lives
        clock=float((lives[lives>calendar]-calendar).mean() if np.any(lives>calendar) else lives.mean())
        phase=float(np.clip(m.phase.predict(bands[None,:])[0],0,1)*m.scale)
        anchor=phase if signal>=1 else clock
        self.amplitudes.append(np.log1p(np.sqrt(np.mean(bands**2))))
        self.smooth.append(float(np.median(self.amplitudes)))
        self.elapsed.append(self.active)
        x=np.asarray(self.elapsed)
        y=np.asarray(self.smooth)
        center=x-x.mean()
        slope=np.dot(center,y-y.mean())/max(np.dot(center,center),1e-8)
        left=m.rate.scale
        if y[-1]>=m.rate.terminal:
            left=0.
        elif slope>0:
            left=min(left,(m.rate.terminal-y[-1])/slope)
        return float((anchor+left*m.rate.ratio)/2)
