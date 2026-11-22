"""从气体父模型的输入尺度和本轮训练测量估计新设备的支持范围。"""
from base64 import b64decode, b64encode
from hashlib import sha256
import json

import numpy as np

if __package__:
    from .gas_device_context import background
    from .gas_views import represent
else:
    from gas_device_context import background
    from gas_views import represent


MAX_SOURCE = 128
MAX_BANKS = 6


def distances(query, recent, parent, *, leave_self_out=False):
    query_bg, source_bg = background(query), background(recent)
    original_bg = background(parent[None, :])[0]
    valid_source = np.isfinite(source_bg).all(axis=1)
    local = np.full(len(query), np.inf)
    original = np.full(len(query), np.inf)
    if not np.isfinite(original_bg).all() or not valid_source.any():
        return local, original
    for i in np.flatnonzero(np.isfinite(query_bg).all(axis=1)):
        delta = np.full(len(recent), np.inf)
        delta[valid_source] = np.median(np.abs(source_bg[valid_source]-query_bg[i]), axis=1)
        if leave_self_out:
            delta[i] = np.inf
        local[i] = delta.min()
        original[i] = np.median(np.abs(original_bg-query_bg[i]))
    return local, original


def source_radius(x, parent, *, extend_to_midpoint=False):
    local, original = distances(x, x, parent, leave_self_out=True)
    supported = np.isfinite(local) & (local < original)
    if supported.sum() <= len(x)/2:
        return None
    radius = float(local[supported].max())
    if extend_to_midpoint:
        radius = max(radius, .5*float(np.median(original[supported])))
    return radius


def supported(query, recent, parent, radius):
    local, original = distances(query, recent, parent)
    return (local < original) & (local <= radius + 1e-12)


def redistribute(parent, classes, conditional, mask):
    """保留未发放类别的概率，只重新分配本轮可学习的总概率质量。"""
    output = parent.copy()
    positions = np.ix_(mask, classes)
    output[positions] = output[positions].sum(axis=1, keepdims=True) * conditional
    return output


def midpoint_radius(x, parent):
    local, between = distances(x, x, parent, leave_self_out=True)
    covered = np.isfinite(local) & (local < between)
    if covered.sum() <= len(x)/2:
        return None
    # 防止一条孤立的训练记录把整个旧台架纳入新分类器的支持范围。
    within = float(np.quantile(local[covered], .9))
    return max(within, float(np.sqrt(within*np.median(between[covered]))))


def parent_mixture(parent, classes, local, mask, parent_weight=.25):
    """作者组合沿用的父概率份额也给缺席类留下非零概率。"""
    output = parent.copy()
    output[mask] *= parent_weight
    output[np.ix_(mask, classes)] += (1.-parent_weight)*local
    return output


def prototype_parameters(train, y):
    from sklearn.preprocessing import StandardScaler

    scale = StandardScaler().fit(train)
    train = scale.transform(train)
    labels = np.unique(y)
    centers = np.asarray([train[y==c].mean(0) for c in labels])
    spread = np.mean([np.square(train[y==c]-m).mean() for c,m in zip(labels,centers,strict=True)])
    distances2 = np.square(train[:,None,:]-centers[None,:,:]).mean(2)
    index = np.searchsorted(labels, y)
    counts = np.bincount(index)[index]
    within = distances2[np.arange(len(y)), index]*(counts/np.maximum(counts-1,1))**2
    distances2[np.arange(len(y)), index] = np.inf
    rival = distances2.min(axis=1)
    return labels, {"mean":scale.mean_.tolist(), "scale":scale.scale_.tolist(),
                    "centers":centers.tolist(), "temperature":max(2*float(spread),1e-8),
                    "known_distance":float(np.median(within)), "missing_distance":float(np.median(rival))}


def prototype_output(query, parameters):
    z = (query-np.asarray(parameters["mean"]))/np.asarray(parameters["scale"])
    centers = np.asarray(parameters["centers"])
    squared = np.square(z[:,None,:]-centers[None,:,:]).mean(2)
    scores = np.exp(-(squared-squared.min(axis=1, keepdims=True))/parameters["temperature"])
    return scores/scores.sum(axis=1, keepdims=True), squared.min(axis=1)


def prototype_probabilities(query, parameters):
    return prototype_output(query,parameters)[0]


def distance_weight(nearest, parameters):
    known, missing = parameters["known_distance"], parameters["missing_distance"]
    if missing <= known:
        return np.full(len(nearest), .75)
    return .75*np.clip((missing-nearest)/(missing-known),0.,1.)


def feature_view(x, view):
    if view == "log":
        return np.sign(x)*np.log1p(np.abs(x))
    if view == "shape":
        return represent(x,"shape",None,None)
    raise ValueError("未知的小样本气体表示")


def source_rows(entry):
    y = np.asarray(entry["labels"],dtype=np.int64)
    raw = b64decode(entry["source"], validate=True)
    if not 2 <= len(y) <= MAX_SOURCE or len(raw) != len(y)*128*8:
        raise ValueError("气体源上下文维度异常")
    x = np.frombuffer(raw,dtype="<f8").reshape(len(y),128)
    if not np.isfinite(x).all() or sha256(raw+y.astype("<i8").tobytes()).hexdigest()!=entry["source_sha256"]:
        raise ValueError("气体源上下文身份变化")
    return x,y


def merge_rows(old_x, old_y, x, y):
    """重发的相同训练测量不重复记账，达到容量时各类先保留两个样本。"""
    xx = np.concatenate((old_x,x)).astype("<f8",copy=False)
    yy = np.concatenate((old_y,y)).astype("<i8",copy=False)
    fingerprints = [sha256(row.tobytes()+label.tobytes()).digest() for row,label in zip(xx,yy,strict=True)]
    unique = {}
    for index, identity in enumerate(fingerprints):
        unique.setdefault(identity,index)
    indices = list(unique.values())
    if len(indices) > MAX_SOURCE:
        order = sorted(indices,key=lambda i:fingerprints[i])
        keep = {i for label in np.unique(yy) for i in [j for j in order if yy[j]==label][:2]}
        keep.update([i for i in order if i not in keep][:MAX_SOURCE-len(keep)])
        indices = sorted(keep)
    return xx[indices], yy[indices]


def prepare(model, x, y):
    paths=[json.dumps([["key","features"],["index",i]],separators=(",",":")) for i in range(128)]
    classes=model.meta["classes"]
    if model.meta.get("profile")!="A" or model.meta["paths"]!=paths or not 2<=len(classes)<=16:
        return None
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y)
    if x.shape!=(len(y),128) or not len(y) or not np.isfinite(x).all() or y.dtype.kind not in "iu" or not np.isin(y,np.arange(len(classes))).all():
        return None
    previous=model.meta.get("gas_partial_context",{}).get("banks",[])
    if not previous and np.unique(y).size==len(classes):
        return None
    selected, best_count = None, len(x)//2
    for i,entry in enumerate(previous):
        old_x,_=source_rows(entry)
        count=int(supported(x,old_x,model.mean,entry["radius"]).sum())
        if count>best_count:
            selected,best_count=i,count
    old_x,old_y = source_rows(previous[selected]) if selected is not None else (np.empty((0,128)),np.empty(0,dtype=int))
    tx,ty = merge_rows(old_x,old_y,x,y)
    labels, counts=np.unique(ty,return_counts=True)
    if len(labels)<2 or counts.min()<2:
        return None
    cap=midpoint_radius(tx,model.mean)
    if cap is None:
        return None
    identity=sha256(tx.tobytes()+ty.tobytes()).hexdigest()
    remaining=[e for i,e in enumerate(previous) if i!=selected]
    # 每个上下文至多128行，保留最近六个独立支持范围，限制原生 NPZ 元信息体积。
    return tx,ty,float(cap),identity,remaining[-(MAX_BANKS-1):],previous[selected] if selected is not None else None


def fit(prepared, view, protect_missing):
    tx,ty,cap,digest,remaining,previous=prepared
    if (previous and previous["source_sha256"]==digest and previous["view"]==view
            and previous["protect_missing"]==protect_missing):
        return None
    classes, params=prototype_parameters(feature_view(tx,view),ty)
    entry={"source":b64encode(tx.tobytes()).decode("ascii"),"labels":ty.tolist(),"source_sha256":digest,
           "classes":classes.tolist(),"radius":cap,"view":view,"protect_missing":protect_missing,"prototype":params}
    return {"format":"gas-partial-context-v1","banks":[*remaining,entry]}


class PartialContext:
    def __init__(self, spec, parent):
        self.parent=parent
        self.classes=len(parent.meta["classes"])
        if spec.get("format")!="gas-partial-context-v1" or not 1<=len(spec["banks"])<=MAX_BANKS or parent.meta.get("profile")!="A":
            raise ValueError("气体小样本检查点格式无效")
        self.banks=[]
        for entry in spec["banks"]:
            x,y=source_rows(entry)
            labels,counts=np.unique(y,return_counts=True)
            params=entry["prototype"]
            arrays=[np.asarray(params[k],dtype=float) for k in ("mean","scale","centers")]
            scalars=[params[k] for k in ("temperature","known_distance","missing_distance")]
            if (entry["view"] not in ("log","shape") or type(entry["protect_missing"]) is not bool
                    or not np.array_equal(labels,entry["classes"]) or not np.isin(labels,np.arange(self.classes)).all()
                    or len(labels)<2 or counts.min()<2 or not np.isfinite(entry["radius"]) or entry["radius"]<0
                    or [a.shape for a in arrays]!=[(128,),(128,),(len(labels),128)]
                    or not all(np.isfinite(a).all() for a in arrays) or np.any(arrays[1]<=0)
                    or not np.isfinite(scalars).all() or min(scalars)<0 or scalars[0]==0):
                raise ValueError("气体小样本表示、类别或概率参数无效")
            self.banks.append((x,labels,entry))

    def local(self, raw, parent):
        raw=np.asarray(raw,dtype=np.float64)
        best=np.full(len(raw),np.inf)
        weight=np.zeros(len(raw))
        conditional=np.zeros_like(parent)
        for training,labels,entry in self.banks:
            distance,original=distances(raw,training,self.parent.mean)
            mask=(distance<original)&(distance<=entry["radius"]+1e-12)&(distance<best)
            if not mask.any():
                continue
            prob,nearest=prototype_output(feature_view(raw[mask],entry["view"]),entry["prototype"])
            best[mask]=distance[mask]
            conditional[mask]=0.
            conditional[np.ix_(mask,labels)]=prob
            weight[mask]=.75
            if entry["protect_missing"]:
                selected=np.flatnonzero(mask)
                missing=~np.isin(parent[mask].argmax(1),labels)
                weight[selected[missing]]=distance_weight(nearest[missing],entry["prototype"])
        return weight[:,None],conditional

    def apply(self, raw, parent):
        w,local=self.local(raw,parent)
        return (1.-w)*parent+w*local

    def continuation(self, raw, logits, existing):
        parent=np.exp(logits-logits.max(axis=1,keepdims=True))
        parent/=parent.sum(axis=1,keepdims=True)
        w,local=self.local(raw,parent)
        mask=w[:,0]>0
        if not mask.any():
            return logits,existing
        mix=(1.-w[mask])*parent[mask]+w[mask]*local[mask]
        safe=np.maximum(1e-30,mix)
        value=logits.copy()
        value[mask]=np.log(safe)
        eye=np.eye(self.classes)
        jac=np.broadcast_to(eye,(len(raw),self.classes,self.classes)).copy()
        factor=np.where(mix>1e-30,(1.-w[mask])*parent[mask]/safe,0.)
        jac[mask]=factor[:,:,None]*(eye-parent[mask,None,:])
        if existing is not None:
            jac=jac@existing if np.ndim(existing)==3 else jac*existing[:,None,:]
        return value,jac


class GasContextModel:
    def __init__(self, model):
        self.context_base=model
        self.gas_context=PartialContext(model.meta["gas_partial_context"],model)

    def __getattr__(self,name):
        return getattr(self.context_base,name)

    def predict(self,raw):
        return self.gas_context.apply(raw,self.context_base.predict(raw))

    def predict_batch(self,raw):
        base=self.context_base
        return self.gas_context.apply(raw,getattr(base,"predict_batch",base.predict)(raw))
