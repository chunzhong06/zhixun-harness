"""用当前已经走过的运行时间以及振动首次持续升高的时刻估算剩余寿命。"""
import numpy as np
from scipy.special import log_ndtr

from harness.models.bearing_prefix import THRESHOLDS, prefix_condition

SURVIVAL = "历史寿命条件均值"
ALL = "固定合用所有振动阶段"
ADAPTIVE = "按既往整台试验学习各阶段预测"


def first_crossings(condition):
    """每个门槛需连续三个已到达的读数越界。-1 表示未出现。"""
    crossings = np.full(len(THRESHOLDS),-1,dtype=int)
    for j,t in enumerate(THRESHOLDS):
        flag = condition[:,1] >= t
        hits = np.flatnonzero(flag[2:] & flag[1:-1] & flag[:-2])
        if len(hits):
            crossings[j] = hits[0]+2
    return crossings


def remaining_mean(lives, ages, sigma):
    """对数正态混合的精确条件均值。已经活到当前时刻的设备才能与本台比较。"""
    ages = np.asarray(ages,dtype=float)
    mu = np.log(np.maximum(lives,1.))
    log_age = np.log(np.maximum(ages,1e-8))[:,None]
    z = (mu-log_age)/sigma
    surv = log_ndtr(z)
    scale = np.max(surv,axis=1,keepdims=True)
    den = np.exp(surv-scale).sum(1)
    num = np.exp(mu + .5*sigma**2 + log_ndtr(z+sigma) - scale).sum(1)
    return np.maximum(0.,num / den - ages)


class LandmarkLife:
    def fit(self, archive):
        self.lives = np.array([u["active_life"] for u in archive])
        self.total_lives = np.array([u["calendar_life"] for u in archive])
        self.fractions = []
        for j in range(len(THRESHOLDS)):
            fractions = []
            for row in archive:
                ix = row["crossings"][j]
                if ix >= 0 and row["age"][ix] > 0:
                    fractions.append(row["age"][ix]/row["active_life"])
            self.fractions.append(np.array(fractions))
        self.gap_rate = np.sum(self.total_lives-self.lives)/max(np.sum(self.lives),1.)
        return self

    def predict(self, cond, age, calendar_age):
        age = np.asarray(age)
        life = remaining_mean(self.lives,age,.5)
        out = {SURVIVAL: life}
        for sigma in (.25, .5, 1.):
            out[f"固定日历生存分布{sigma:g}"] = remaining_mean(self.total_lives,calendar_age,sigma)
            out[f"固定运行生存分布{sigma:g}"] = remaining_mean(self.lives,age,sigma)
        times = first_crossings(cond)
        by_threshold, available = [], []
        for j, (index, frac) in enumerate(zip(times,self.fractions,strict=True)):
            pred = life.copy()
            valid = np.zeros(len(life),dtype=bool)
            if index >= 0 and len(frac) >= 2:
                valid[index:] = True
                # 本台抵达该振动阶段的时间已知，按每份既往档案的寿命占比外推。
                lengths = age[index]/np.maximum(frac,.01)
                pred[index:] = remaining_mean(lengths,age[index:],.5)
            out[f"振动阶段{j+1}"] = pred
            by_threshold.append(pred)
            available.append(valid)
        bank = np.column_stack(by_threshold)
        mask = np.column_stack(available)
        out[ALL] = (life + (bank*mask).sum(1))/(1+mask.sum(1))
        out["固定历史中断率"] = out[ALL]*(1+self.gap_rate)
        return out, mask.sum(1)


def record(unit):
    cond = prefix_condition(unit.spectral)
    return {"active_life":unit.durations["capped"],"calendar_life":unit.durations["calendar"],
            "age":unit.ages["capped"],"condition":cond,"crossings":first_crossings(cond)}


class LandmarkTrials:
    def fit(self, history):
        self.names = [SURVIVAL, ALL, "固定历史中断率", "固定日历生存分布0.5"]
        losses={k:[] for k in self.names}
        stage=[]
        for predictions, counts, truth in history:
            stage.extend(counts)
            for k in self.names:
                losses[k].extend((predictions[k]-truth)**2)
        stage=np.asarray(stage)
        self.actions={}
        for n in range(len(THRESHOLDS)+1):
            w=(stage==n).astype(float)+.1
            self.actions[n]=min(self.names,key=lambda k:np.average(losses[k],weights=w))
        return self

    def predict(self, bank, stage):
        return np.array([bank[self.actions[int(n)]][i] for i,n in enumerate(stage)])
