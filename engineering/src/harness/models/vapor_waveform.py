"""从原始采样的真实间隔估计电桥响应的快慢与上升、恢复方向。"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from harness.models.vapor_field import voltage_ratio
from harness.models.vapor_boundary import LevelFit

PRIMARY = "在现场用已回参考拟合原始波形与论文校正的组合"
FIXED_PAPER = "固定源场校正权重"
NAMES = ("高频波形树", "高频波形装机回归", "两种高频读法固定各半",
         "高频树与论文固定各半", "高频装机回归与论文固定各半", "两种高频读法与论文各半")


class Waveform:
    def __init__(self,baseline,scale):
        self.baseline = baseline
        self.scale = scale
        self.state = None
        self.last_time = None
        self.period = np.array([.18,.45,1.,2.5,6.,15.])

    def next(self,samples,times):
        times = np.asarray(times,dtype=float)
        if times.shape!=(len(samples),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError("每帧必须有递增的原采样时间")
        z = (voltage_ratio(samples)-self.baseline)/self.scale
        trace=[]
        for t,row in zip(times,z,strict=True):
            if self.state is None:
                self.state = np.tile(row,(len(self.period),1))
            else:
                dt=t-self.last_time
                if dt<=0:
                    raise ValueError("跨窗时间未递增")
                self.state+=(-np.expm1(-dt/self.period))[:,None]*(row-self.state)
            trace.append(self.state.copy())
            self.last_time=t
        state=np.stack(trace)
        differences = state[:,:-1]-state[:,1:]
        # 上升和恢复分开表示，使用到当前窗末为止的实测。时间仅参与积分。
        return np.r_[z.mean(0),np.ptp(z,axis=0),self.state.ravel(),
                     differences.mean(0).ravel(),differences.min(0).ravel(),differences.max(0).ravel()]


class WaveReader:
    def fit(self,x,y):
        self.tree = HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=15,
            min_samples_leaf=24,learning_rate=.07,l2_regularization=1.5,random_state=73)
        self.tree.fit(x,np.arcsinh(y))
        self.ridge = LevelFit().fit(x,y)
        return self

    def predict(self,x):
        return self.tree.predict(x), np.arcsinh(self.ridge.predict(x)[:,1])

    def update(self,x,y):
        self.ridge.update(x,y)


class WaveMix:
    def __init__(self):
        self.grid = np.linspace(0.,1.,11)
        self.err = np.zeros((3,len(self.grid)))
        self.mass = 0

    def predict(self,paper,tree,ridge):
        bases = np.array([tree,ridge,(tree+ridge)/2])
        p=np.arcsinh(paper)
        self.last = p+(bases[:,None]-p)*self.grid
        # 候选包含完全保留原固定端。弱先验偏向已有读法，回标逐项改变相对风险。
        cost=(self.err+2*self.grid**2)/(5+self.mass)
        weights=np.exp(-40*(cost-cost.min()))
        estimate=np.average(self.last,weights=weights)
        controls=[*bases,*((p+bases)/2)]
        return {**dict(zip(NAMES,np.sinh(np.clip(controls,-5,5)).tolist(),strict=True)),
                PRIMARY:float(np.sinh(np.clip(estimate,-5,5)))}

    def update(self,y):
        observed = np.arcsinh(y)
        near = 1/(1+abs(observed-np.arcsinh(1.)))
        loss = np.minimum((self.last-observed)**2,1.)*near + .15*((self.last>=np.arcsinh(1.))!=(y>=1))
        self.err = .95*self.err+loss
        self.mass = .95*self.mass+1
