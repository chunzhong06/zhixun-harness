"""用源工况内的父版梯度估计距离；发布后只以查询当前行检索源寿命。"""
from hashlib import sha256

import numpy as np

if __package__:
    from .engine_condition_density import mode_keys
else:
    from engine_condition_density import mode_keys

PREFIX="engine_neighbor_"
FIELDS=("center","scale","projection","anchors","labels")


def source_modes(x,condition):
    return mode_keys(x,condition["indices"],condition["decimals"])


def usable_modes(x,y,condition):
    if (x.ndim!=2 or x.shape!=(len(y),24) or len(y)<64 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or len(condition.get("indices",[]))!=3):
        return []
    cols=np.setdiff1d(np.arange(24),condition["indices"])
    keys=source_modes(x,condition)
    tokens,idx,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    selected=[]
    for i in np.flatnonzero(counts>=64):
        rows=np.flatnonzero(idx==i)
        values=x[rows][:,cols]
        mean,scale=values.mean(0),values.std(0)
        keep=scale>1e-6*np.maximum(1.,np.abs(mean))
        if keep.any() and np.ptp(y[rows])>0:
            selected.append((tokens[i],rows,cols[keep],mean[keep],scale[keep]))
    # 扩展到多工况输入时，优先保留源样本较充足的工况。
    if len(selected)>384:
        selected=sorted(sorted(selected,key=lambda v:-len(v[1]))[:384],key=lambda v:tuple(v[0]))
    return selected


def fit(model,x,y):
    modes=usable_modes(x,y,model.state["condition"])
    if not modes or getattr(model,"_density",None) is None:
        raise ValueError("同工况寿命检索需要条件密度父版及足量有变化的源训练行")
    rng=np.random.default_rng(42)
    arrays={}
    records=[]
    limit=min(4096,24576//len(modes))
    for i,(token,rows,cols,center,scale) in enumerate(modes):
        anchors=np.sort(rng.choice(rows,min(limit,len(rows)),replace=False))
        examples=x[rng.choice(rows,min(384,len(rows)),replace=False)]
        gradients=np.empty((len(examples),len(cols)))
        for j,col in enumerate(cols):
            a,b=examples.copy(),examples.copy()
            a[:,col]+=.01*scale[j]
            b[:,col]-=.01*scale[j]
            gradients[:,j]=(model.predict(a)-model.predict(b))/.02
        eig,vec=np.linalg.eigh(gradients.T@gradients/len(examples))
        eig=np.maximum(eig,0.)
        eig=eig*len(cols)/eig.sum() if eig.sum()>0 else np.ones(len(cols))
        projection=vec*np.sqrt(.1+.9*eig)
        neighbors=((x[anchors][:,cols]-center)/scale)@projection
        for key,value in zip(FIELDS,(center,scale,projection,neighbors,y[anchors]),strict=True):
            arrays[f"{PREFIX}{i}_{key}"]=np.ascontiguousarray(value,dtype=np.float64)
        records.append({"mode":token.tolist(),"columns":cols.tolist(),"count":len(anchors)})
    return {"format":"engine-gradient-neighbors-v1","rows":len(x),"modes":records,
            "source_x_sha256":sha256(x.tobytes()).hexdigest(),"source_y_sha256":sha256(y.tobytes()).hexdigest(),
            "hashes":{k:sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}},arrays


class LifeNeighbors:
    def __init__(self,spec,arrays,condition):
        if (not isinstance(spec,dict) or spec.get("format")!="engine-gradient-neighbors-v1"
                or type(spec.get("rows")) is not int or spec["rows"]<64 or not isinstance(spec.get("hashes"),dict)
                or not isinstance(spec.get("modes"),list) or not 1<=len(spec["modes"])<=384):
            raise ValueError("发动机同工况检索描述无效")
        self.condition=condition
        self.blocks=[]
        modes=set()
        expected=set()
        total=0
        for i,record in enumerate(spec["modes"]):
            token,cols,n=record["mode"],record["columns"],record["count"]
            if (len(token)!=3 or not np.isfinite(token).all() or tuple(token) in modes
                    or not 1<=len(cols)<=21 or any(type(j) is not int or j not in range(24) or j in condition["indices"] for j in cols)
                    or len(set(cols))!=len(cols) or type(n) is not int or not 64<=n<=4096):
                raise ValueError("发动机检索工况或传感器列无效")
            modes.add(tuple(token))
            total+=n
            keys=[f"{PREFIX}{i}_{key}" for key in FIELDS]
            expected.update(keys)
            d=len(cols)
            for key,shape in zip(keys,((d,),(d,),(d,d),(n,d),(n,)),strict=True):
                v=arrays.get(key)
                if (v is None or v.shape!=shape or v.dtype!=np.float64 or not np.isfinite(v).all()
                        or sha256(v.tobytes()).hexdigest()!=spec["hashes"].get(key)):
                    raise ValueError("发动机检索数组形状、数据或摘要不符")
            center,scale,projection,anchors,labels=(arrays[k] for k in keys)
            if np.any(scale<=0):
                raise ValueError("发动机近邻缩放必须为正数")
            self.blocks.append((np.asarray(token),cols,center,scale,projection,anchors,(anchors*anchors).sum(1),labels))
        if (total>min(24576,spec["rows"]) or expected!={k for k in arrays if k.startswith(PREFIX)}
                or expected!=set(spec["hashes"])):
            raise ValueError("发动机检索的源行或数组总数不符")

    def mix(self,raw,parent,derivative):
        value=parent.copy()
        slope=derivative.copy()
        keys=source_modes(raw,self.condition)
        for token,cols,center,scale,mat,anchors,square,labels in self.blocks:
            hit=np.flatnonzero((keys==token).all(1))
            for start in range(0,len(hit),32):
                ix=hit[start:start+32]
                z=((raw[ix][:,cols]-center)/scale)@mat
                # 查询的平方范数对排序是常数。选出近邻后直接作差以保留小距离的精度。
                order=square-2*z@anchors.T
                if len(anchors)==64:
                    near=np.broadcast_to(np.arange(64),(len(ix),64))
                else:
                    ranked=np.argpartition(order,(63,64),axis=1)
                    near=ranked[:,:64].copy()
                    boundary=np.take_along_axis(order,ranked[:,63:65],axis=1)
                    tolerance=128*np.finfo(np.float64).eps*(1.+square.max()+(z*z).sum(1))
                    for r in np.flatnonzero(boundary[:,1]-boundary[:,0]<=tolerance):
                        # 等距的第 64、65 位用源行次序破同分，单条/成批采用相同样本。
                        exact=((anchors-z[r])**2).sum(1)
                        near[r]=np.argsort(exact,kind='stable')[:64]
                dist=((anchors[near]-z[:,None,:])**2).sum(2)
                weights=np.exp(-2.*dist/np.maximum(dist.max(1,keepdims=True),1e-24))
                local=(weights*labels[near]).sum(1)/weights.sum(1)
                value[ix]=.75*parent[ix]+.25*local
                slope[ix]*=.75
        return value,slope
