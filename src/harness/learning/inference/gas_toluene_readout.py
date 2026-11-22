"""用当轮源训练拟合多类 RBF SVM，发布纯数值参数以供 SDK 单行预测。"""
from hashlib import sha256

import numpy as np


FIELDS = ("gas_svm_vectors", "gas_svm_pairs", "gas_svm_scaler")


def baseline(raw):
    sensors = np.asarray(raw, dtype=np.float64).reshape(-1,16,8)
    denominator = sensors[:,:,1] - 1
    value = np.divide(sensors[:,:,0], denominator, out=np.full_like(denominator,np.nan),
                      where=np.abs(denominator)>1e-5)
    return np.where(np.isfinite(value) & (value>0),value,np.nan)


def calibrated(raw, reference):
    observed = baseline(raw)
    gain = reference / np.where(np.isfinite(observed), observed, reference)
    result = np.asarray(raw,dtype=np.float64).reshape(-1,16,8).copy()
    result[:,:,0] *= gain
    result[:,:,2:] *= gain[:,:,None]
    return result.reshape(-1,128)


def available(x, y, classes):
    return (x.ndim==2 and x.shape==(len(y),128) and len(classes)==6 and "Toluene" in classes
            and np.array_equal(np.unique(y),np.arange(6)) and np.isfinite(x).all()
            and np.isfinite(baseline(x)).any(axis=0).all())


def fit_readout(x, y, classes):
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    x,y = np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.int64)
    if not available(x,y,classes):
        raise ValueError("当前合法训练投影缺少 SVM 所需类别或传感器本底")
    reference = np.nanmedian(baseline(x),axis=0)
    raw = calibrated(x,reference)
    scaler = RobustScaler().fit(raw)
    z = np.arcsinh(scaler.transform(raw))
    svm = SVC(C=10, gamma="scale", class_weight="balanced").fit(z,y)
    # libsvm 按类别连续存放支持向量，每对分类器的两个系数段来自不同的行。
    offsets = np.r_[0,np.cumsum(svm.n_support_)]
    coefficients = np.zeros((len(svm.support_vectors_),15))
    for k,(i,j) in enumerate(zip(*np.triu_indices(6,1),strict=True)):
        left, right = slice(offsets[i],offsets[i+1]),slice(offsets[j],offsets[j+1])
        coefficients[left,k] = svm.dual_coef_[j-1,left]
        coefficients[right,k] = svm.dual_coef_[i,right]
    arrays = dict(zip(FIELDS,(svm.support_vectors_,coefficients,np.stack((scaler.center_,scaler.scale_))),strict=True))
    spec = {"format":"gas-toluene-agreement-v1", "rows":len(x), "target":classes.index("Toluene"),
            "gamma":float(svm._gamma), "intercepts":svm.intercept_.tolist(), "baseline":reference.tolist(),
            "hashes":{k:sha256(a.tobytes()).hexdigest() for k,a in arrays.items()},
            "source_x_sha256":sha256(x.tobytes()).hexdigest(), "source_y_sha256":sha256(y.tobytes()).hexdigest()}
    np.testing.assert_array_equal(votes(x,spec,arrays),svm.predict(z))
    return spec, arrays


def validate(spec, arrays, classes):
    present = set(FIELDS).intersection(arrays)
    if spec is None and not present:
        return
    if (not isinstance(spec,dict) or spec.get("format")!="gas-toluene-agreement-v1" or present!=set(FIELDS)
            or len(classes)!=6 or "Toluene" not in classes or spec.get("target")!=classes.index("Toluene")
            or type(spec.get("rows")) is not int or spec["rows"]<=0
            or not isinstance(spec.get("gamma"),(int,float)) or not np.isfinite(spec["gamma"]) or spec["gamma"]<=0):
        raise ValueError("甲苯 SVM 描述与实际分类父版不符")
    vectors,pairs,scaler = (arrays[field] for field in FIELDS)
    base,intercepts = np.asarray(spec["baseline"]),np.asarray(spec["intercepts"])
    if (vectors.ndim!=2 or vectors.shape[1]!=128 or not 0<len(vectors)<=spec["rows"]
            or pairs.shape!=(len(vectors),15) or scaler.shape!=(2,128) or not (scaler[1]>0).all()
            or base.shape!=(16,) or not (base>0).all() or not np.isfinite(base).all()
            or intercepts.shape!=(15,) or not np.isfinite(intercepts).all()
            or any(arrays[k].dtype!=np.float64 or not np.isfinite(arrays[k]).all()
                   or sha256(arrays[k].tobytes()).hexdigest()!=spec["hashes"].get(k) for k in FIELDS)):
        raise ValueError("甲苯 SVM 的支持向量、变换或数值摘要无效")


def votes(raw, spec, arrays):
    vectors,pairs,scaler = (arrays[field] for field in FIELDS)
    left,right = np.triu_indices(6,1)
    out = np.empty(len(raw),dtype=np.int64)
    norm = np.einsum("ij,ij->i",vectors,vectors)
    for start in range(0,len(raw),128):
        x = calibrated(raw[start:start+128],np.asarray(spec["baseline"]))
        z = np.arcsinh((x-scaler[0])/scaler[1])
        squared = np.maximum((z*z).sum(1)[:,None] + norm - 2*z@vectors.T,0.)
        margins = np.exp(-spec["gamma"]*squared)@pairs + spec["intercepts"]
        winners = np.where(margins>0,left,right)
        counts = np.zeros((len(x),6),dtype=np.int64)
        np.add.at(counts,(np.arange(len(x))[:,None],winners),1)
        out[start:start+len(x)] = counts.argmax(1)
    return out


def apply_readout(raw, probabilities, spec, arrays):
    """召回 SVM 识别的甲苯；否决甲苯还要求两边同意具体的替代气体。"""
    target = spec["target"]
    svm = votes(raw,spec,arrays)
    prior = probabilities.argmax(1)
    alternate = probabilities.copy()
    alternate[:,target] = -1.
    replacement = alternate.argmax(1)
    to_target = (prior!=target) & (svm==target)
    from_target = (prior==target) & (svm==replacement)
    rows = np.flatnonzero(to_target|from_target)
    result = probabilities.copy()
    result[rows,prior[rows]],result[rows,svm[rows]] = probabilities[rows,svm[rows]],probabilities[rows,prior[rows]]
    return result
