"""从已到达的载荷、转速和采样间隔累计运行量；训练只收已经结束的整台设备。"""
import numpy as np
from sklearn.preprocessing import StandardScaler

PRIMARY="按已结束整台的相似运行阶段合用累计载荷"
GLOBAL="累计载荷按历史全局表现固定合用"
RULES=("按运行时间","按累计转数","按一次载荷与转数","按三次载荷与转数")


def work_rates(settings):
    # 使用实测载荷和转速；作者提供峰值动态力与平均静态力量，二者合成为经验代理。
    load=np.hypot(settings[:,1],settings[:,3])/3000.
    speed=np.abs(settings[:,5])/2500.
    return np.column_stack((np.ones(len(speed)),speed,speed*load,speed*load**3))


def inspect(settings,active,elapsed):
    rates=work_rates(settings)
    dt=np.diff(active,prepend=active[0])
    mass=np.cumsum(dt[:,None]*rates,axis=0)
    # 外推采用当前已经观察过的负荷均值，避免刚启动时积分为零。
    pace=(mass+25*rates)/(active[:,None]+25)
    recent=elapsed-active
    context=np.column_stack((np.log1p(active/60),np.log1p(recent/60),
        np.log1p(pace[:,1:]),np.log1p(rates[:,2:])))
    return mass,pace.clip(.04),context


class LoadLife:
    def fit(self,completed):
        ends=[];ratios=[]
        for u in completed:
            age=u.ages["capped"];cal=u.ages["calendar"]
            mass,pace,_=inspect(u.settings,age,cal)
            # 采样按20行抽取，历史尾部至真实结束的短段用最后实测负荷近似。
            tail=u.durations["capped"]-age[-1]
            ends.append(mass[-1]+tail*work_rates(u.settings[-1:])[0])
            ratios.append(u.durations["calendar"]/max(u.durations["capped"],1.))
        self.ends=np.asarray(ends)
        self.gap_rate=float(np.median(ratios))
        return self

    def predict(self,settings,active,elapsed):
        mass,pace,context=inspect(settings,active,elapsed)
        predictions=[]
        for j in range(len(RULES)):
            ref=self.ends[:,j]
            # 分布尾部逐渐降权，保持已超过既往最长设备的场景仍有有限估计。
            scale=max(1.,float(np.median(ref))*.4)
            left=ref-mass[:,j,None]
            w=np.exp(-np.minimum(np.maximum(-left/scale,0),25))
            remain=(w*np.maximum(left,.12*scale)).sum(1)/w.sum(1)
            predictions.append(remain/pace[:,j])
        active_bank=np.column_stack(predictions)
        calendar_bank=active_bank*self.gap_rate
        return active_bank,calendar_bank,context


class LoadTrials:
    def fit(self,trials):
        self.trials=trials
        all_context=np.vstack([c for c,_ in trials])
        self.scale=StandardScaler().fit(all_context)
        self.global_cost=np.mean([cost.mean(0) for _,cost in trials],axis=0)
        return self

    def predict(self,context,bank):
        query=self.scale.transform(context)
        risk=[]
        from scipy.spatial.distance import cdist
        for c,squared in self.trials:
            d=cdist(query,self.scale.transform(c),"sqeuclidean")
            k=min(len(c),max(4,len(c)//8))
            ix=np.argpartition(d,k-1,axis=1)[:,:k]
            risk.append(squared[ix].mean(1))
        loss=(np.mean(risk,axis=0)+.35*self.global_cost)/1.35
        def weighted(cost):
            weights=1/np.maximum(.001,cost)
            return (bank*weights).sum(1)/weights.sum(-1)
        return {PRIMARY:weighted(loss),GLOBAL:weighted(self.global_cost),
                **dict(zip(RULES,bank.T,strict=True))}
