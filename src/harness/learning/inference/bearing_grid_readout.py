"""以双轴功率谱的点数识别采集窗口，用已发放的新设备标签校准实际父版寿命。"""
from hashlib import sha256
import json

import numpy as np

POOL_METHOD = "scaled-ridge-mean-pool-v1"
METHOD = "scaled-ridge-mean-pool-terminal-v2"


def terminal_module():
    if __package__:
        from . import bearing_terminal_readout
    else:
        import bearing_terminal_readout
    return bearing_terminal_readout


def health_module():
    if __package__:
        from . import bearing_health_readout
    else:
        import bearing_health_readout
    return bearing_health_readout


def spectral_columns(paths):
    fields = ("band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k", "spectral_rms")
    lookup = {path: i for i, path in enumerate(paths)}
    pairs = []
    for axis in ("horizontal", "vertical"):
        indices = []
        for field in fields:
            encoded = json.dumps([["key","features"],["key",axis],["key",field]],separators=(",",":"))
            if encoded not in lookup:
                return None
            indices.append(lookup[encoded])
        pairs.append(indices)
    return np.array(pairs)


def grid(raw, columns):
    signals = np.asarray(raw, dtype=np.float64)[:, columns]
    energy, denominator = signals[:, :, :3].sum(2), signals[:, :, 3]**2
    counts = np.divide(energy, denominator, out=np.zeros_like(energy), where=denominator>0)
    rounded = np.rint(counts)
    valid = (np.isfinite(counts).all(1) & (counts>1).all(1) & (counts<2**53).all(1)
             & (np.abs(counts-rounded)<=.25).all(1))
    return np.where(valid[:,None],rounded,0).astype(np.int64)


def checkpoint_bins(mean, scale, paths):
    """单一谱窗口满足 E[总能量] = 点数 × E[谱 RMS²]。只接受双轴均能还原的整数。"""
    columns = spectral_columns(paths)
    if columns is None:
        return None
    rms = columns[:, 3]
    if np.any(scale[rms] == 1.):  # 原生检查点会把零标准差替换成 1，已无法还原二阶矩。
        return None
    second = np.square(mean[rms]) + np.square(scale[rms])
    energy = mean[columns[:, :3]].sum(axis=1)
    counts = np.divide(energy, second, out=np.zeros_like(energy), where=second > 0)
    nearest = np.rint(counts)
    if (not np.isfinite(counts).all() or (nearest <= 1).any() or (nearest >= 2**53).any()
            or (np.abs(counts-nearest) > 1e-6).any()):
        return None
    return tuple(map(int, nearest))


def new_groups(x, y, prior, paths, previous=None, *, source_bins=None):
    columns = spectral_columns(paths)
    if columns is None or len(x)!=len(y):
        return []
    if prior is None and source_bins is None:
        return []
    historic = {source_bins} if prior is None else set(map(tuple,grid(prior,columns)))
    current = grid(x,columns)
    saved = {tuple(item["bins"]):item["source_sha256"] for item in (previous or {}).get("models",[])
             if item.get("method")==METHOD}
    groups = []
    for bins in np.unique(current,axis=0):
        key = tuple(map(int,bins))
        if 0 in key or key in historic:
            continue
        use = (current==bins).all(1)
        if use.sum()<3 or np.unique(y[use]).size<2:
            continue
        digest = sha256(np.ascontiguousarray(x[use],dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(y[use],dtype=np.float64).tobytes())
        identity = digest.hexdigest()
        if saved.get(key)!=identity:
            groups.append((key,use,identity))
    return groups


def design(x, parent_prediction, factor=1.):
    return np.column_stack((np.sign(x)*np.log1p(abs(x)/factor),parent_prediction))


def fit(model, x, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler

    previous = getattr(model, "state", {}).get("bearing_grid_readout")
    source = getattr(model, "auxiliary_arrays", {}).get("readout_source_x")
    paths = model.meta["paths"]
    source_bins = checkpoint_bins(model.mean, model.scale, paths) if source is None else None
    groups = new_groups(x, y, source, paths, previous, source_bins=source_bins)
    if not groups:
        return None
    # 训练缓存来自已发放的源行。替换同一口径时，使用校准前的实际父输出避免重复校准。
    history = {tuple(item["bins"]):item for item in (previous or {}).get("models",[])}
    for bins, use, identity in groups:
        tx, ty = x[use], y[use]
        predictor = model.training_before_grid if hasattr(model, "training_before_grid") else model.predict
        prior = predictor(tx)
        factor = np.median(np.abs(tx),axis=0)
        factor = np.where(factor>0,factor,1.)
        z = design(tx,prior,factor)
        scaler = RobustScaler().fit(z)
        estimator = Ridge(alpha=1.0).fit(scaler.transform(z),ty)
        weights = estimator.coef_ / scaler.scale_
        intercept = float(estimator.intercept_ - np.dot(weights,scaler.center_))
        entry = {"bins":list(bins),"rows":len(tx),"source_sha256":identity,"method":METHOD,
                 "weights":weights.tolist(),"intercept":intercept,"factor":factor.tolist(),
                 "target_range":[float(ty.min()),float(ty.max())],"target_mean":float(ty.mean())}
        health = health_module()
        if health.available(model.meta["paths"],ty):
            entry["health_cap"] = health.fit(tx,ty,model.meta["paths"])
        terminal = terminal_module().fit(tx,ty,factor)
        if terminal is not None:
            entry["terminal"] = terminal
        history[bins] = entry
    return {"format":"bearing-spectrum-grid-ridge-v1","models":[history[k] for k in sorted(history)]}


class GridReadout:
    def __init__(self, spec, paths):
        self.columns = spectral_columns(paths)
        if self.columns is None or not isinstance(spec,dict) or spec.get("format")!="bearing-spectrum-grid-ridge-v1":
            raise ValueError("寿命校准的功率谱字段或描述无效")
        self.models = {}
        for item in spec["models"]:
            bins = tuple(item["bins"])
            weights = np.asarray(item["weights"],dtype=np.float64)
            if (len(bins)!=2 or any(type(n) is not int or n<=1 for n in bins) or bins in self.models
                    or weights.shape!=(len(paths)+1,) or not np.isfinite(weights).all() or not np.isfinite(item["intercept"])
                    or type(item["rows"]) is not int or item["rows"]<3 or len(item["source_sha256"])!=64):
                raise ValueError("寿命校准的拟合参数无效")
            method = item.get("method")
            if method is None:  # 已经发布的首版成品保留当时的预测语义。
                self.models[bins] = (weights,float(item["intercept"]),1.,None,None,None,None)
                continue
            factor = np.asarray(item["factor"], dtype=np.float64)
            bounds = np.asarray(item["target_range"], dtype=np.float64)
            mean = float(item["target_mean"])
            if (method not in (POOL_METHOD,METHOD) or factor.shape!=(len(paths),) or not np.isfinite(factor).all()
                    or (factor<=0).any() or bounds.shape!=(2,) or not np.isfinite(bounds).all()
                    or not 0<=bounds[0]<=mean<=bounds[1]):
                raise ValueError("采集窗口的特征量纲或寿命范围无效")
            cap = health_module().BearingHealthReadout(item["health_cap"],paths) if "health_cap" in item else None
            terminal = terminal_module().TerminalReadout(item["terminal"],factor) if method==METHOD and "terminal" in item else None
            self.models[bins] = (weights,float(item["intercept"]),factor,bounds,mean,cap,terminal)
        if not self.models:
            raise ValueError("寿命校准缺少合法已发放的采集口径")

    def apply(self, raw, parent, *, native_logits=False):
        candidates = grid(raw,self.columns)
        output = np.asarray(parent,dtype=np.float64).copy()
        derivative = np.ones(len(raw))
        for bins,(weights,intercept,factor,bounds,mean,cap,terminal) in self.models.items():
            use = (candidates==bins).all(1)
            if use.any():
                original = output[use]
                # 原生回归预测截断负寿命；原采集窗口继续使用未截断的训练 logit。
                prior = np.maximum(0., original) if native_logits else original
                value = design(raw[use],prior,factor) @ weights + intercept
                if bounds is None:
                    output[use] = np.maximum(0.,value)
                    derivative[use] = (value>0)*weights[-1]
                    if native_logits:
                        derivative[use] *= original > 0.
                    continue
                upper = bounds[1] if cap is None else np.minimum(bounds[1],cap.upper(raw[use]))
                output[use] = .5*(mean+np.minimum(np.clip(value,*bounds),upper))
                derivative[use] = .5*weights[-1]*((value>bounds[0]) & (value<upper))
                if terminal is not None:
                    output[use], slope = terminal.apply(raw[use],output[use])
                    derivative[use] *= slope
                if native_logits:
                    derivative[use] *= original > 0.
        return output, derivative
