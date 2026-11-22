"""用源训练去掉本底干扰方向，把两类瞬态判据折叠为可逐行部署的线性读出。"""
from hashlib import sha256

import numpy as np


def log_transients(raw):
    responses = np.abs(np.asarray(raw,dtype=float).reshape(-1,16,8)[:,:,2:])
    peak = responses.max(axis=2,keepdims=True)
    return np.log(np.where(peak>0,np.maximum(responses,peak*1e-6),1.)).reshape(len(raw),96)


def fit_readout(x,y,classes,*,covariance_estimator=None):
    """资料缺少两类语义、可比的源分层或可辨方向时，沿用已有分类方案。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import RobustScaler
    x,y = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.int64)
    if (x.ndim!=2 or x.shape!=(len(y),128) or y.ndim!=1 or not np.isfinite(x).all()
            or "Ethanol" not in classes or "Acetaldehyde" not in classes):
        return None
    labels, counts = np.unique(y,return_counts=True)
    if not np.array_equal(labels,np.arange(len(classes))) or min(counts)<8:
        return None
    sensors = x.reshape(-1,16,8)
    den = sensors[:,:,1]-1
    proxy = np.divide(sensors[:,:,0],den,out=np.zeros_like(den),where=np.abs(den)>1e-5)
    if np.any(proxy<=0) or not np.isfinite(proxy).all():
        return None
    strength = np.median(np.abs(den),axis=1)
    baseline = np.log(proxy/np.median(proxy,axis=0)).mean(axis=1)
    raw = log_transients(x)
    scale = RobustScaler().fit(raw)
    train = scale.transform(raw)
    differences = []
    for label in labels:
        rows = np.flatnonzero(y==label)
        for stratum in np.array_split(rows[np.argsort(strength[rows],kind="stable")],4):
            ordered = stratum[np.argsort(baseline[stratum],kind="stable")]
            half = len(ordered)//2
            if baseline[ordered[-1]]>baseline[ordered[0]]:
                differences.append(train[ordered[-half:]].mean(0)-train[ordered[:half]].mean(0))
    if not differences or not np.any(differences):
        return None
    axis = np.linalg.svd(np.stack(differences),full_matrices=False)[2][0]
    old, new = classes.index("Ethanol"),classes.index("Acetaldehyde")
    pair = np.isin(y,(old,new))
    projected = train[pair]-np.outer(train[pair]@axis,axis)
    # 布尔目标保证判别得分始终朝向乙醛，即便拥有相同名称的父版调换了类序。
    # 默认沿用已有估计；研发比较只替换类内协方差，输入、类别先验和本底投影相同。
    covariance = {"shrinkage": "auto"} if covariance_estimator is None else {"covariance_estimator": covariance_estimator}
    lda = LinearDiscriminantAnalysis(solver="lsqr", priors=[.5,.5], **covariance).fit(projected, y[pair]==new)
    w = lda.coef_[0]
    weight = (w-axis*(axis@w))/scale.scale_
    coefficients = np.append(weight,lda.intercept_[0]-scale.center_@weight)
    if not np.isfinite(coefficients).all() or not np.any(weight):
        return None
    spec = {"format":"gas-drift-pair-v1","rows":len(y),"reported":old,"target":new,
            "coefficients_sha256":sha256(coefficients.tobytes()).hexdigest(),
            "source_features_sha256":sha256(x.tobytes()).hexdigest(),"source_labels_sha256":sha256(y.tobytes()).hexdigest()}
    return spec, coefficients


def apply_readout(raw,probabilities,spec,coefficients):
    """只处理原报乙醇的行；对内概率按对数赔率等权合并，其他四类的概率原样交付。"""
    old,new = spec["reported"],spec["target"]
    rows = np.flatnonzero(probabilities.argmax(1)==old)
    if not len(rows):
        return probabilities
    p = probabilities[rows].copy()
    rank = log_transients(np.asarray(raw)[rows])@coefficients[:-1]+coefficients[-1]
    odds = np.log(np.maximum(p[:,new],1e-15))-np.log(np.maximum(p[:,old],1e-15))
    posterior = np.exp(-np.logaddexp(0.,-.5*(odds+rank)))
    mass = p[:,old]+p[:,new]
    p[:,old],p[:,new] = mass*(1-posterior),mass*posterior
    use = p.argmax(1)==new
    result = probabilities.copy()
    result[rows[use]] = p[use]
    return result
