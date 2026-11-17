"""对连续测量作因果编码，只有已经交付并返回的标签进入下一批训练。"""
from collections import deque
import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler

VIEWS = ("当前浓度读数","近期响应差分","论文储备池快记忆","论文储备池慢记忆","差分与论文记忆合用")
PRIMARY = "按已返回的稳定段和变化段错误更新读法"


class DynamicEncoder:
    def __init__(self,source):
        self.scale = StandardScaler().fit(source)
        rng = np.random.default_rng(71)
        n = 64
        w = rng.normal(size=(n,n))*(rng.random((n,n))<.12)
        self.w = w*(.9/np.max(abs(np.linalg.eigvals(w))))
        self.input_w = rng.normal(0,.23,size=(16,n))
        self.bias = rng.uniform(-.15,.15,size=n)
        self.reset()

    def reset(self):
        self.past = deque(maxlen=21)
        self.state = np.zeros((2,len(self.bias)))

    def encode(self,seconds):
        # 当前十秒结束后发放，所有记忆只含本窗口及更早的采样。
        z = self.scale.transform(seconds)
        if not len(z) or not np.isfinite(z).all():raise ValueError("缺少有效的十秒测量")
        state_values = []
        changes = []
        for x in z:
            if not self.past:self.past.append(x.copy())
            changes.append(np.r_[x-self.past[-1],x-self.past[max(-4,-len(self.past))],x-self.past[0]])
            self.past.append(x.copy())
            proposal = np.tanh(x@self.input_w+self.state@self.w+self.bias)
            self.state += np.array([.7,.15])[:,None]*(proposal-self.state)
            state_values.append(self.state.copy())
        avg = z.mean(0)
        changes = np.stack(changes)
        motion = np.mean(np.abs(changes[:,:16]))
        diff = np.r_[avg,changes.mean(0),changes[-1]]
        mean_state = np.mean(state_values,axis=0)
        fast,slow = [np.r_[avg,mean_state[i],self.state[i]] for i in range(2)]
        return [avg,diff,fast,slow,np.r_[diff,mean_state.ravel()]],float(motion)


class StreamRegression:
    def fit(self,features,y):
        self.scale = StandardScaler().fit(features)
        x = self.design(features)
        self.gram = x.T@x/len(x)
        self.rhs = x.T@y/len(x)
        self.count = 0
        self.site_gram = np.zeros_like(self.gram)
        self.site_rhs = np.zeros_like(self.rhs)
        self.recent_gram = self.site_gram.copy()
        self.recent_rhs = self.site_rhs.copy()
        self.recent_count = 0.
        return self

    def design(self,v):
        return np.column_stack((np.ones(len(v)),self.scale.transform(v)))

    def outputs(self,v):
        x = self.design(v)
        ridge = .005*np.eye(x.shape[1])
        def fit(gram,rhs):return x@solve(gram+ridge,rhs,assume_a="pos")
        src = fit(self.gram,self.rhs)
        if not self.count:return np.column_stack((src,src,src))
        global_m = fit(.2*self.gram+self.site_gram/self.count,.2*self.rhs+self.site_rhs/self.count)
        recent = fit(.2*self.gram+self.recent_gram/self.recent_count,.2*self.rhs+self.recent_rhs/self.recent_count)
        return np.column_stack((src,global_m,recent))

    def update(self,v,y):
        x = self.design(v)
        gram,rhs = x.T@x,x.T@y
        self.site_gram+=gram;self.site_rhs+=rhs;self.count+=len(v)
        self.recent_gram=.8*self.recent_gram+gram
        self.recent_rhs=.8*self.recent_rhs+rhs
        self.recent_count=.8*self.recent_count+len(v)


class MotionChoice:
    def __init__(self,source_motion,tools):
        self.breaks = np.quantile(source_motion,(.25,.5,.75))
        self.totals = np.zeros((4,tools))
        self.counts = np.zeros(4)

    def predict(self,motion,bank):
        global_cost = (self.totals.sum(0)+.22*12)/(self.counts.sum()+12)
        risk = (self.totals+12*global_cost)/(self.counts[:,None]+12)
        bins = np.digitize(motion,self.breaks)
        weights = np.exp(-12*risk[bins]);weights/=weights.sum(1)[:,None]
        global_w = np.exp(-12*global_cost);global_w/=global_w.sum()
        return {PRIMARY:(weights*bank).sum(1),"统一按历史误差加权":bank@global_w,
                "全部动态工具固定平均":bank.mean(1)}

    def update(self,motion,bank,y):
        loss = ((bank>=.5)!=y[:,None]).astype(float)+.15*(np.clip(bank,0,1)-y[:,None])**2
        for group in range(4):
            take = np.digitize(motion,self.breaks)==group
            self.totals[group]=.9*self.totals[group]+loss[take].sum(0)
            self.counts[group]=.9*self.counts[group]+take.sum()


class DynamicGasService:
    """按时间窗口发放并暂存该批训练输入，回执确认后才可消费本批回标。"""
    def fit(self,seconds,windows,labels,calibrate_lag=False):
        from harness.learning.models.gas_transitions import EdgeEncoder
        from harness.learning.models.delayed_teacher import DelayedTeacher
        self.encoder = DynamicEncoder(seconds)
        self.edge = EdgeEncoder(seconds) if calibrate_lag else None
        self.names = [v+"/"+m for v in VIEWS for m in ("只用源端","累计回标","近期回标")]
        starts = np.r_[0,np.flatnonzero(np.diff(windows))+1,len(windows)]
        if len(starts)-1!=len(labels) or np.any(np.diff(windows)<0) or not np.isin(labels,[1,2]).all():
            raise ValueError("源连续测量与标签错位")
        history,motion,edges = [],[],[]
        for a,b in zip(starts[:-1],starts[1:],strict=True):
            views,velocity = self.encoder.encode(seconds[a:b])
            history.append(views);motion.append(velocity)
            if self.edge is not None:edges.append(self.edge.encode(seconds[a:b]))
        self.models = [StreamRegression().fit(np.stack(v),labels==2) for v in zip(*history,strict=True)]
        self.choice = MotionChoice(motion,len(self.names))
        self.encoder.reset()
        if self.edge is not None:
            self.edge.past=[]
            self.delay=DelayedTeacher.from_source(np.stack(edges),np.stack([v[1] for v in history]),
                                                 np.stack([v[-1] for v in history]),(labels==2).astype(float))
        self.pending = None
        self.last_time = -np.inf
        return self

    def predict(self,clock,sensors,prefix):
        if self.pending is not None:raise ValueError("上一批尚未确认反馈")
        if (clock.ndim!=1 or sensors.shape!=(len(clock),16) or not len(clock)
                or not np.isfinite(clock).all() or not np.isfinite(sensors).all()
                or np.any(np.diff(clock)<0) or clock[0]<=self.last_time):
            raise ValueError("原始批次时钟或传感器字段错误")
        times = np.floor(clock).astype(int)
        cuts = np.r_[0,np.flatnonzero(np.diff(times))+1,len(times)]
        sec = np.add.reduceat(sensors,cuts[:-1],axis=0)/np.diff(cuts)[:,None]
        w = times[cuts[:-1]]//10
        splits = np.r_[0,np.flatnonzero(np.diff(w))+1,len(w)]
        if len(np.unique(w//60))!=1:raise ValueError("服务须为单一十分钟批次")
        ids = [f"{prefix}/{i}" for i in w[splits[:-1]]]
        f,motions,edges = [],[],[]
        for a,b in zip(splits[:-1],splits[1:],strict=True):
            cols,speed = self.encoder.encode(sec[a:b]);f.append(cols);motions.append(speed)
            if self.edge is not None:edges.append(self.edge.encode(sec[a:b]))
        feat = [np.stack(col) for col in zip(*f,strict=True)]
        motion = np.array(motions)
        bank = np.column_stack([m.outputs(x) for m,x in zip(self.models,feat,strict=True)])
        out = dict(zip(self.names,bank.T,strict=True));out.update(self.choice.predict(motion,bank))
        diff = np.clip(out["近期响应差分/累计回标"],0,1)
        reservoir = np.clip(out["差分与论文记忆合用/累计回标"],0,1)
        out["差分与论文固定各半"] = (diff+reservoir)/2
        if self.edge is not None:
            anchor=out["差分与论文固定各半"] if prefix=="co" else reservoir
            estimates=self.delay.predict(np.column_stack((edges,feat[1])),anchor)
            out.update(estimates)
            out.update({"论文与"+k+"各半":(reservoir+np.clip(v,0,1))/2 for k,v in estimates.items()})
        self.pending = (ids,feat,motion,bank)
        self.last_time = clock[-1]
        return ids,{k:(1+(v>=.5)).tolist() for k,v in out.items()}

    def update(self,labels):
        if self.pending is None:raise ValueError("没有待回标批次")
        ids,features,motion,bank = self.pending
        if (len(labels)>32 or not set(labels)<=set(ids) or not all(v in (1,2) for v in labels.values())):
            raise ValueError("反馈不属于该次发放或超出预算")
        if labels:
            ix = np.array([ids.index(k) for k in labels])
            y = np.array(list(labels.values()))==2
            self.choice.update(motion[ix],bank[ix],y)
            for m,x in zip(self.models,features,strict=True):m.update(x[ix],y)
        if self.edge is not None:
            indices=np.array([int(k.rsplit("/",1)[1]) for k in labels],dtype=int)
            self.delay.update(indices,np.array(list(labels.values()))==2)
        self.pending = None
