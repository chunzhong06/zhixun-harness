"""同型号多器件共享浓度校准，按每日已知洁净空气修正完整加热曲线。"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def heating_curve(t, voltage, resistance):
    high = voltage > .55
    rising = np.flatnonzero(high[1:] & ~high[:-1]) + 1
    usable = rising[t[rising]+24.5 <= t[-1]]
    if not len(usable):
        raise ValueError("暴露没有足够的完整温度周期")
    start = t[usable[-1]]
    # 在最后一个完整高低周期上重采样；长低温周期只用前20秒以与另一类周期对齐。
    grid = start + np.arange(-2,27) + .5
    values = []
    for col in resistance.T:
        valid = np.isfinite(col) & (col>0)
        if not valid.any():
            raise ValueError("校准器没有有效电阻")
        cond = np.interp(grid,t[valid],1/col[valid])
        filtered = uniform_filter1d(median_filter(cond,size=3,mode="nearest"),size=3,mode="nearest")
        values.append(np.log(filtered[2:27]))
    return np.stack(values)


def readings(x, groups, components, query, family="pls"):
    # 每个器件作一个训练实例，每次暴露只交一个最终类别。
    z = x.reshape(-1,x.shape[-1])
    y = np.repeat(groups,x.shape[1])
    target = query.reshape(-1,query.shape[-1])
    scale = StandardScaler().fit(z)
    z, target = scale.transform(z), scale.transform(target)
    if family=="pls":
        n = min(components, z.shape[1], len(z)-1)
        model = PLSRegression(n_components=n,scale=False,max_iter=500).fit(z,y)
        raw = model.predict(target).reshape(query.shape[:2])
        return np.clip(raw.mean(1),1,3), raw.std(1)
    model = SVC(C=4,gamma="scale",class_weight="balanced",random_state=17).fit(z,y)
    predicted = model.predict(target).reshape(query.shape[:2])
    votes = np.stack([(predicted==n).sum(1) for n in (1,2,3)],axis=1)
    return 1 + votes.argmax(1), predicted.std(1)


def ordered_label(score):
    return np.clip(np.floor(score+.5),1,3).astype(int)
