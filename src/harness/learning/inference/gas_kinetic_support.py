"""用源训练估计乙醇支持范围与乙醛判别分数，服务侧只依赖 NumPy 和当行输入。"""
from functools import lru_cache
from hashlib import sha256

import numpy as np


FIELDS=("kinetic_support_vectors","kinetic_support_dual","kinetic_support_scaler")


@lru_cache(maxsize=1)
def response_templates():
    """计算一阶阶跃在三种官方 EMA 系数下的峰值形状，消去共同幅值。"""
    log_tau=np.linspace(np.log(.5),np.log(32768.),241)
    pole=np.exp(-1./np.exp(log_tau))[:,None]
    alpha=np.array([.001,.01,.1])
    state=np.zeros((len(pole),3))
    maximum=state.copy()
    difference=1.-pole
    for _ in range(8192):
        state=(1.-alpha)*state+alpha*difference
        maximum=np.maximum(maximum,state)
        difference=difference*pole
    shape=np.log(maximum)
    return log_tau,shape-shape.mean(1,keepdims=True)


def transform(raw):
    x=np.asarray(raw,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all():
        raise ValueError("气体响应尺度需要完整的 16 路当行测量")
    measured=np.abs(x.reshape(-1,16,8)[:,:,2:]).reshape(-1,3)
    scale=measured.max(1,keepdims=True)
    logged=np.log(np.where(scale>0,np.maximum(measured,scale*1e-8),1.))
    shape=logged-logged.mean(1,keepdims=True)
    tau,table=response_templates()
    index=np.empty(len(shape),dtype=int)
    norm=(table*table).sum(1)
    for i in range(0,len(shape),1024):
        part=shape[i:i+1024]
        index[i:i+len(part)]=np.argmin(norm-2.*part@table.T,axis=1)
    latent=tau[index].copy()
    residual=shape-table[index]
    empty=scale[:,0]==0
    latent[empty]=0.
    residual[empty]=0.
    return np.column_stack((latent.reshape(len(x),32),residual.reshape(len(x),96)))


def available(x,y,classes):
    if (x.ndim!=2 or x.shape!=(len(y),128) or not np.isfinite(x).all() or len(classes)!=6
            or "Ethanol" not in classes or "Acetaldehyde" not in classes or not np.array_equal(np.unique(y),np.arange(6))):
        return False
    return bool(np.min(np.unique(y,return_counts=True)[1])>=5)


def kernels(query,source,gamma):
    # 直接算差值可避免极远的样本与源向量相减时发生平方和相消。
    differences=query[:,None,:]-source[None,:,:]
    return -gamma*np.einsum("ijk,ijk->ij",differences,differences)


def fit(raw,y,classes):
    """在源行上交叉拟合相似度，再用全部源行保存最终 SVM 和变换刻度。"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    x=np.asarray(raw,dtype=np.float64)
    y=np.asarray(y)
    if not available(x,y,classes):
        raise ValueError("乙醛读出需要六类各至少五条合法源训练数据")
    ethanol,aldehyde=classes.index("Ethanol"),classes.index("Acetaldehyde")
    features=transform(x)

    def train(indices):
        scaler=RobustScaler().fit(features[indices])
        z=scaler.transform(features[indices])
        svc=SVC(C=3.,gamma="scale",class_weight="balanced").fit(z,y[indices]==aldehyde)
        return scaler,svc,z

    oof=np.zeros(len(y))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(features,y):
        scaler,svc,ref=train(tr)
        for start in range(0,len(va),16):
            ix=va[start:start+16]
            oof[ix]=kernels(scaler.transform(features[ix]),ref,svc._gamma).max(1)
    scaler,svc,ref=train(np.arange(len(x)))
    dual=np.zeros(len(x))
    dual[svc.support_]=svc.dual_coef_[0]
    arrays=dict(zip(FIELDS,(ref,dual,np.stack((scaler.center_,scaler.scale_))),strict=True))
    spec={"format":"gas-kinetic-ethanol-support-v1","rows":len(y),"classes":list(classes),"ethanol":ethanol,"aldehyde":aldehyde,
          "gamma":float(svc._gamma),"cutoff":float(np.quantile(oof[y==ethanol],.05)),
          "source_x_sha256":sha256(x.tobytes()).hexdigest(),"source_y_sha256":sha256(y.tobytes()).hexdigest(),
          "hashes":{k:sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    if "Ethylene" in classes:
        spec["response_pair"],extra=fit_response_pair(x,y,classes)
        spec["response_pair"]["refine_pair_odds"]=True
        arrays.update(extra)
    return spec,arrays


def validate(spec,arrays,classes):
    if (not isinstance(spec,dict) or spec.get("format")!="gas-kinetic-ethanol-support-v1" or spec.get("classes")!=classes
            or "Ethanol" not in classes or "Acetaldehyde" not in classes or len(classes)!=6
            or spec.get("ethanol")!=classes.index("Ethanol") or spec.get("aldehyde")!=classes.index("Acetaldehyde")
            or type(spec.get("rows")) is not int or spec["rows"]<30
            or not isinstance(spec.get("gamma"),(int,float)) or not np.isfinite(spec["gamma"]) or spec["gamma"]<=0
            or not isinstance(spec.get("cutoff"),(int,float)) or not np.isfinite(spec["cutoff"]) or spec["cutoff"]>0
            or not isinstance(spec.get("hashes"),dict) or set(spec["hashes"])!=set(FIELDS)
            or {k for k in arrays if k.startswith("kinetic_support_")}!=set(FIELDS)):
        raise ValueError("乙醛源支持读出的描述与父版不符")
    vectors,dual,scaler=(arrays[k] for k in FIELDS)
    if (vectors.shape!=(spec["rows"],128) or dual.shape!=(spec["rows"],) or scaler.shape!=(2,128)
            or not (scaler[1]>0).all() or not (dual>0).any() or not (dual<0).any()
            or any(arrays[k].dtype!=np.float64 or not np.isfinite(arrays[k]).all()
                   or sha256(arrays[k].tobytes()).hexdigest()!=spec["hashes"].get(k) for k in FIELDS)):
        raise ValueError("乙醛源支持读出的权重、刻度或摘要无效")
    if "response_pair" in spec or any(k.startswith("gas_pair_") for k in arrays):
        validate_response_pair(spec.get("response_pair"),arrays,classes)


def apply_readout(raw,probabilities,spec,arrays):
    """仅在原判乙醇、超出源乙醇支持且正核证据占优时交换两类概率。"""
    eligible=np.flatnonzero(probabilities.argmax(1)==spec["ethanol"])
    if not len(eligible) and "response_pair" not in spec:
        return probabilities
    vectors,dual,scaler=(arrays[k] for k in FIELDS)
    raw=np.asarray(raw,dtype=np.float64)
    result=probabilities.copy()
    for start in range(0,len(eligible),16):
        rows=eligible[start:start+16]
        z=(transform(raw[rows])-scaler[0])/scaler[1]
        logmass=kernels(z,vectors,spec["gamma"])
        change=rows[(logmass.max(1)<spec["cutoff"])&((np.exp(logmass)@dual)>0)]
        a,b=spec["ethanol"],spec["aldehyde"]
        result[change,a],result[change,b]=probabilities[change,b],probabilities[change,a]
    if "response_pair" in spec:
        result=apply_response_pair(raw,result,spec["response_pair"],arrays)
    return result


def pair_features(raw):
    measured=np.asarray(raw,dtype=np.float64).reshape(-1,16,8)
    change=measured[:,:,1]-1.
    baseline=np.log(np.maximum(np.abs(measured[:,:,0])/np.maximum(np.abs(change),1e-8),1e-8))
    strength=np.log(np.maximum(np.abs(change),1e-6))
    transients=np.log(np.maximum(np.abs(measured[:,:,2:]),1e-6))
    return baseline,strength,np.concatenate((strength[:,:,None],transients),axis=2)


def fit_response_pair(x,y,classes):
    """在同一气体、相近响应强度内拟合本底斜率，训练六类概率，仅使用乙醇与乙烯的相对证据。"""
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    baseline,strength,features=pair_features(x)
    center=np.median(baseline,axis=0)
    level=np.median(strength,axis=1)
    numerator=np.zeros((16,7))
    denominator=np.zeros(16)
    for label in np.unique(y):
        rows=np.flatnonzero(y==label)
        for block in np.array_split(rows[np.argsort(level[rows])],4):
            b=baseline[block]-baseline[block].mean(axis=0)
            v=features[block]-features[block].mean(axis=0)
            numerator+=np.mean(b[:,:,None]*v,axis=0)
            denominator+=np.mean(b*b,axis=0)
    slopes=numerator/(denominator[:,None]+.1*np.median(denominator)+1e-8)
    corrected=(features-(baseline-center)[:,:,None]*slopes).reshape(len(x),112)
    scaler=RobustScaler().fit(corrected)
    z=scaler.transform(corrected)
    svm=SVC(C=8.,probability=True,class_weight="balanced",random_state=42).fit(z,y)
    arrays={"gas_pair_center":center,"gas_pair_slopes":slopes,
            "gas_pair_scale":np.stack((scaler.center_,scaler.scale_)),
            "gas_pair_vectors":svm.support_vectors_,"gas_pair_coefs":svm.dual_coef_,
            "gas_pair_counts":svm.n_support_.astype(np.int32),
            "gas_pair_calibration":np.stack((svm.intercept_,svm.probA_,svm.probB_))}
    spec={"format":"gas-baseline-pair-v1","classes":list(classes),"gamma":float(svm._gamma),
          "hashes":{k:sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}}
    # 校准概率使用与 sklearn 训练相同的 libsvm 多类耦合；仅保存数值，不反序列化模型对象。
    np.testing.assert_allclose(pair_probabilities(z,spec,arrays),svm.predict_proba(z),rtol=1e-10,atol=1e-12)
    return spec,arrays


def validate_response_pair(spec,arrays,classes):
    vectors=arrays.get("gas_pair_vectors")
    if vectors is None or vectors.ndim!=2 or not len(vectors):
        raise ValueError("气体双类支持向量缺失")
    count=len(vectors)
    shapes={"gas_pair_center":(16,),"gas_pair_slopes":(16,7),"gas_pair_scale":(2,112),
            "gas_pair_vectors":(count,112),"gas_pair_coefs":(5,count),"gas_pair_counts":(6,),
            "gas_pair_calibration":(3,15)}
    if (not isinstance(spec,dict) or spec.get("format")!="gas-baseline-pair-v1" or spec.get("classes")!=list(classes)
            or type(spec.get("refine_pair_odds",False)) is not bool
            or len(classes)!=6 or not {"Ethanol","Ethylene"}.issubset(classes)
            or not isinstance(spec.get("gamma"),(int,float)) or not 0<spec["gamma"]<np.inf
            or set(k for k in arrays if k.startswith("gas_pair_"))!=set(shapes)
            or not isinstance(spec.get("hashes"),dict) or set(spec["hashes"])!=set(shapes)):
        raise ValueError("气体双类描述与当前模型不符")
    for k,shape in shapes.items():
        array=arrays[k]
        dtype=np.int32 if k=="gas_pair_counts" else np.float64
        if array.shape!=shape or array.dtype!=dtype or not np.isfinite(array).all() or sha256(array.tobytes()).hexdigest()!=spec["hashes"][k]:
            raise ValueError("气体双类数值参数与摘要不符")
    if np.any(arrays["gas_pair_scale"][1]<=0) or np.any(arrays["gas_pair_counts"]<=0) or arrays["gas_pair_counts"].sum()!=count:
        raise ValueError("气体双类支持向量计数或刻度有误")


def pair_probabilities(z,spec,arrays):
    from sklearn.svm import _libsvm
    vectors=arrays["gas_pair_vectors"]
    intercept,prob_a,prob_b=arrays["gas_pair_calibration"]
    return _libsvm.predict_proba(np.ascontiguousarray(z,dtype=np.float64),np.arange(len(vectors),dtype=np.int32),
                vectors,arrays["gas_pair_counts"],arrays["gas_pair_coefs"],intercept,prob_a,prob_b,
                svm_type=0,kernel="rbf",gamma=spec["gamma"])


def apply_response_pair(raw,probabilities,spec,arrays):
    pair=[spec["classes"].index(k) for k in ("Ethanol","Ethylene")]
    rows=np.flatnonzero(np.isin(probabilities.argmax(axis=1),pair))
    if not len(rows):
        return probabilities
    baseline,_,features=pair_features(np.asarray(raw)[rows])
    z=(features-(baseline-arrays["gas_pair_center"])[:,:,None]*arrays["gas_pair_slopes"]).reshape(len(rows),112)
    center,scale=arrays["gas_pair_scale"]
    p=pair_probabilities((z-center)/scale,spec,arrays)
    ix=np.ix_(rows,pair)
    old=probabilities[ix]
    relative=np.sqrt(np.maximum(old,1e-12)*p[:,pair])
    result=probabilities.copy()
    result[ix]=relative/relative.sum(axis=1,keepdims=True)*old.sum(axis=1,keepdims=True)
    if spec.get("refine_pair_odds",False):
        # 仅在第一次复核后仍由此双类胜出的行增强证据，沿用已算出的专家概率。
        kept=np.isin(result[rows].argmax(axis=1),pair)
        slots=np.ix_(rows[kept],pair)
        first=result[slots]
        refined=np.sqrt(np.maximum(first,1e-12)*p[kept][:,pair])
        result[slots]=refined/refined.sum(axis=1,keepdims=True)*first.sum(axis=1,keepdims=True)
    return result
