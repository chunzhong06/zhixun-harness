"""用当轮源训练的振动能量与冲击严重度，在源上尾区域限制剩余寿命。"""
import json

import numpy as np


ENERGY = ("rms","mean_abs","peak_to_peak","band1_energy_0_2k","band2_energy_2k_6k","band3_energy_6k_12k")
IMPULSE = ("kurtosis","peak","crest_factor","impulse_factor")
LEVELS = (.90,.95,.975,.99,.995)


def column_indices(paths):
    lookup = {}
    for index, encoded in enumerate(paths):
        try:
            parts=json.loads(encoded)
        except (ValueError,TypeError):
            return None
        if (isinstance(parts,list) and len(parts)==3 and parts[0]==["key","features"]
                and all(isinstance(p,list) and len(p)==2 and p[0]=="key" and isinstance(p[1],str) for p in parts[1:])):
            key = parts[1][1],parts[2][1]
            if key in lookup:
                return None
            lookup[key] = index
    ordered=[(axis,name) for axis in ("horizontal","vertical") for name in (*ENERGY,*IMPULSE)]
    if any(key not in lookup for key in ordered):
        return None
    return np.asarray([lookup[key] for key in ordered]).reshape(2,-1)


def available(paths,y):
    return len(y)>=80 and np.isfinite(y).all() and np.ptp(y)>0 and np.min(y)>=0 and column_indices(paths) is not None


def severity(raw,columns,center,scale):
    z=(np.log1p(np.abs(np.asarray(raw,dtype=np.float64)[:,columns]))-center)/scale
    active=np.clip((z-1.)/3.,0.,1.)
    components=[]
    for group in (active[:,:,:len(ENERGY)],active[:,:,len(ENERGY):]):
        values=group.reshape(len(raw),-1)
        count=(values.shape[1]+1)//2
        components.append(np.partition(values,-count,axis=1)[:,-count:].mean(1))
    return (components[0]+components[1])/2


def fit(raw,y,paths):
    y=np.asarray(y,dtype=np.float64)
    if not available(paths,y) or len(raw)!=len(y):
        raise ValueError("严重度读出需要可用的双轴特征及足量非恒定源寿命标签")
    columns=column_indices(paths)
    x=np.asarray(raw,dtype=np.float32).astype(np.float64)
    reference=np.log1p(np.abs(x[y>=np.quantile(y,.75)][:,columns]))
    center=np.median(reference,axis=0)
    scale=np.maximum(1.4826*np.median(abs(reference-center),axis=0),1e-6)
    signal=severity(x,columns,center,scale)
    marks,limits=[],[]
    for cutoff in np.unique(np.quantile(signal,LEVELS)):
        subset=y[signal>=cutoff]
        if len(subset)>=8:
            marks.append(float(cutoff))
            limits.append(float(np.quantile(subset,.90)))
    if not marks:
        raise ValueError("源严重度上尾样本不足")
    return {"format":"bearing-energy-impulse-cap-v1","rows":len(y),"center":center.tolist(),"scale":scale.tolist(),
            "thresholds":marks,"limits":limits}


class BearingHealthReadout:
    def __init__(self,spec,paths):
        self.columns=column_indices(paths)
        if (self.columns is None or not isinstance(spec,dict) or spec.get("format")!="bearing-energy-impulse-cap-v1"
                or type(spec.get("rows")) is not int or spec["rows"]<80):
            raise ValueError("轴承能量冲击读出描述与父版不符")
        self.center=np.asarray(spec["center"],dtype=float)
        self.scale=np.asarray(spec["scale"],dtype=float)
        self.marks=np.asarray(spec["thresholds"],dtype=float)
        self.limits=np.asarray(spec["limits"],dtype=float)
        if (self.center.shape!=self.columns.shape or self.scale.shape!=self.center.shape or (self.scale<=0).any()
                or self.marks.ndim!=1 or not 0<len(self.marks)<=len(LEVELS) or self.marks.shape!=self.limits.shape
                or (np.diff(self.marks)<=0).any() or (self.marks<0).any() or (self.marks>1).any() or (self.limits<0).any()
                or not all(np.isfinite(a).all() for a in (self.center,self.scale,self.marks,self.limits))):
            raise ValueError("轴承严重度刻度或寿命上界无效")

    def upper(self,raw):
        value=severity(raw,self.columns,self.center,self.scale)
        step=np.searchsorted(self.marks,value,side="right")-1
        return np.where(step>=0,self.limits[np.maximum(step,0)],np.inf)
