"""把一次完整 CO 暴露的电阻按当时实测的加热电压和切换时间拆开。"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from harness.learning.models.field_classification import weights


VIEWS = ("各路平均", "相位电阻", "相位比", "器件中位相位比")


def cycle_summary(t, heat, resistance):
    high = heat > .55
    changes = np.r_[0,np.flatnonzero(high[1:] != high[:-1])+1]
    elapsed = t - np.repeat(t[changes],np.diff(np.r_[changes,len(t)]))
    # 跳过截在暴露边界的首段，避免把未知的前一次切换当作本暴露内的切换。
    valid = np.arange(len(t)) >= (changes[1] if len(changes)>1 else len(t))
    masks = (high & (elapsed<2), high & (elapsed>=2), ~high & (elapsed<3),
             ~high & (elapsed>=3) & (elapsed<10), ~high & (elapsed>=10))
    logs = np.log(np.where(resistance>0, resistance, np.nan))
    means = np.nanmean(resistance,axis=0)
    med = np.nanmedian(logs,axis=0)
    vals = []
    for mask in masks:
        selected = valid & mask
        typical = np.nanmedian(logs[selected],axis=0) if selected.any() else med
        vals.append(np.where(np.isfinite(typical),typical,med))
    result = np.vstack((means,*vals))
    if not np.isfinite(result).all():
        raise ValueError("暴露的电阻摘要缺少有效测量")
    return result


def read_views(features, channels):
    values = features[:,:,channels]
    phase = values[:,1:,:]
    shift = phase - phase[:,-1:,:]
    # 减掉同一器件同次冷态的对数电阻，抵消各路乘性刻度。
    return {"各路平均":values[:,0,:], "相位电阻":phase.reshape(len(phase),-1),
            "相位比":shift[:,:-1].reshape(len(phase),-1),
            "器件中位相位比":np.median(shift[:,:-1],axis=2)}


def train_votes(src, sy, observed, returned_x, returned_y, query):
    out = {}
    for name in VIEWS:
        source = StandardScaler().fit_transform(src[name])
        scale = StandardScaler().fit(observed[name])
        x = scale.transform(query[name])
        y = np.asarray(returned_y,dtype=int)
        seen = scale.transform(returned_x[name]) if len(y) else np.empty((0,source.shape[1]))
        for choice in ("仅源","合训","仅现场"):
            a,b,m = source, sy, weights(sy,len(sy))
            if choice=="合训" and len(y):
                a,b,m = np.vstack((source,seen)), np.r_[sy,y], np.r_[m,weights(y,len(sy))]
            if choice=="仅现场" and len(np.unique(y))>=2:
                a,b,m = seen,y,weights(y,len(y))
            out[name+choice] = SVC(C=4,gamma="scale",random_state=17).fit(a,b,sample_weight=m).predict(x)
    return out
