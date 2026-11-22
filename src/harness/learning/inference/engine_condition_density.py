"""按源训练工况保存联合高斯的条件均值，查询只使用当行测量。"""
from hashlib import sha256

import numpy as np


PREFIX = "engine_gmr_"
FIELDS = ("modes","anchors","signature","fit_ids","center","scale","limits","means","whitener","slope","log_mass")


def mode_keys(raw, columns, decimals):
    return np.column_stack([np.round(raw[:,i],d) for i,d in zip(columns,decimals,strict=True)])


def available(x, y, condition):
    cols=condition.get("indices", [])
    if (len(cols)!=3 or len(set(cols))!=3 or x.ndim!=2 or x.shape[1]!=24 or y.shape!=(len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not all(0<=i<x.shape[1] for i in cols) or len(condition.get("decimals",[]))!=3):
        return False
    modes=mode_keys(x,cols,condition["decimals"])
    _,groups,counts=np.unique(modes,axis=0,return_counts=True,return_inverse=True)
    return any(np.ptp(y[groups==group])>0 for group in np.flatnonzero(counts>=6*(x.shape[1]+2)))


def fit_readout(x, y, condition):
    from sklearn.mixture import GaussianMixture
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    if not available(x,y,condition):
        raise ValueError("合法训练投影缺少可估计的发动机工况")
    columns,decimals=condition["indices"],condition["decimals"]
    sensors=np.array([i for i in range(x.shape[1]) if i not in columns])
    modes,inv=np.unique(mode_keys(x,columns,decimals),axis=0,return_inverse=True)
    medians,stable=[],[]
    values={k:[] for k in FIELDS[3:]}
    for i in range(len(modes)):
        subset=inv==i
        readings=x[subset][:,sensors]
        median=np.median(readings,axis=0)
        medians.append(median)
        stable.append((readings==median).mean(0)>=.99)
        if len(readings)<6*(x.shape[1]+2) or np.ptp(y[subset])==0:
            continue
        joint=np.column_stack((x[subset],y[subset]))
        center,scale=joint.mean(0),joint.std(0)
        scale=np.where(scale>1e-9*np.maximum(1.,np.abs(center)),scale,1.)
        mixture=GaussianMixture(n_components=6,reg_covar=.005,random_state=42,n_init=1,max_iter=50,tol=.001)
        mixture.fit((joint-center)/scale)
        factor=np.linalg.cholesky(mixture.covariances_[:,:-1,:-1])
        # 白化、线性条件均值和边缘密度常数预先求好；推理端仅依赖 NumPy。
        values["fit_ids"].append(i)
        values["center"].append(center)
        values["scale"].append(scale)
        values["limits"].append([y[subset].min(),y[subset].max()])
        values["means"].append(mixture.means_)
        values["whitener"].append(np.linalg.inv(factor).transpose(0,2,1))
        values["slope"].append(np.linalg.solve(factor,mixture.covariances_[:,:-1,-1,None])[:,:,0])
        values["log_mass"].append(np.log(mixture.weights_)-np.log(np.diagonal(factor,axis1=1,axis2=2)).sum(1))
    if not values["fit_ids"]:
        raise ValueError("各工况的合法寿命标签均不足以拟合联合分布")
    medians=np.asarray(medians)
    keep=np.all(stable,axis=0)&(np.ptp(medians,axis=0)>0)
    values.update(modes=modes,anchors=sensors[keep],signature=medians[:,keep])
    arrays={PREFIX+k:np.asarray(v) for k,v in values.items()}
    spec={"format":"engine-condition-gmr-v1","rows":len(x),"weight":.25,"columns":list(columns),"decimals":list(decimals),
          "source_x_sha256":sha256(x.tobytes()).hexdigest(),"source_y_sha256":sha256(y.tobytes()).hexdigest(),
          "hashes":{k:sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    return spec, arrays


class ConditionDensity:
    def __init__(self, spec, arrays, width, condition):
        fields={k for k in arrays if k.startswith(PREFIX)}
        if (not isinstance(spec,dict) or spec.get("format")!="engine-condition-gmr-v1" or spec.get("weight")!=.25
                or width!=24 or spec.get("columns")!=condition["indices"] or spec.get("decimals")!=condition["decimals"]
                or len(condition["indices"])!=3 or len(set(condition["indices"]))!=3 or type(spec.get("rows")) is not int
                or spec["rows"]<1 or fields!={PREFIX+k for k in FIELDS}):
            raise ValueError("发动机条件密度描述与父模型不符")
        self.columns,self.decimals=spec["columns"],spec["decimals"]
        for k in FIELDS:
            v=arrays[PREFIX+k]
            if v.dtype.kind not in 'fi' or not np.isfinite(v).all() or sha256(v.tobytes()).hexdigest()!=spec["hashes"].get(PREFIX+k):
                raise ValueError("发动机条件密度数值状态损坏")
            setattr(self,k,v)
        m,n=len(self.modes),len(self.fit_ids)
        expected={"modes":(m,3),"signature":(m,len(self.anchors)),"fit_ids":(n,),"center":(n,width+1),
                  "scale":(n,width+1),"limits":(n,2),"means":(n,6,width+1),"whitener":(n,6,width,width),
                  "slope":(n,6,width),"log_mass":(n,6)}
        if (not 0<n<=m or len(np.unique(self.modes,axis=0))!=m or np.any(self.scale<=0)
                or self.anchors.ndim!=1 or self.anchors.dtype.kind!='i' or len(set(self.anchors))!=len(self.anchors)
                or any(i not in range(width) or i in self.columns for i in self.anchors)
                or self.fit_ids.dtype.kind!='i' or len(set(self.fit_ids))!=n or np.any(self.fit_ids<0) or np.any(self.fit_ids>=m)
                or any(getattr(self,k).shape!=shape for k,shape in expected.items()) or np.any(self.limits[:,0]>self.limits[:,1])):
            raise ValueError("发动机条件密度的工况、传感器或协方差维度无效")

    def map(self, x):
        if len(self.anchors)<2:
            return x
        keys=mode_keys(x,self.columns,self.decimals)
        match=keys[:,None,:]==self.modes[None,:,:]
        unknown=~match.all(2).any(1)
        if not unknown.any():
            return x
        rows=np.flatnonzero(unknown)
        support=(x[rows][:,None,self.anchors]==self.signature[None,:,:]).all(2)
        choices=(match[rows].sum(2)==2)&support
        unique=(support.sum(1)==1)&(choices.sum(1)==1)
        if not unique.any():
            return x
        rows=rows[unique]
        selected=choices[unique].argmax(1)
        wrong=np.argmin(match[rows,selected],axis=1)
        result=x.copy()
        result[rows,np.asarray(self.columns)[wrong]]=self.modes[selected,wrong]
        return result

    def mix(self, raw, parent, derivative):
        keys=mode_keys(raw,self.columns,self.decimals)
        out=np.asarray(parent,dtype=np.float64).copy()
        slope=derivative.copy()
        for i,mode in enumerate(self.fit_ids):
            hit=(keys==self.modes[mode]).all(1)
            if not hit.any():
                continue
            z=(raw[hit]-self.center[i,:-1])/self.scale[i,:-1]
            evidence,conditional=[],[]
            for j in range(6):
                centered=z-self.means[i,j,:-1]
                u=centered@self.whitener[i,j]
                evidence.append(self.log_mass[i,j]-.5*(u*u).sum(1))
                estimate=self.center[i,-1]+self.scale[i,-1]*(self.means[i,j,-1]+u@self.slope[i,j])
                conditional.append(np.clip(estimate,*self.limits[i]))
            logp=np.column_stack(evidence)
            p=np.exp(logp-logp.max(axis=1,keepdims=True))
            expert=(p*np.column_stack(conditional)).sum(1)/p.sum(1)
            out[hit]=.75*out[hit]+.25*expert
            slope[hit]*=.75
        return out,slope
