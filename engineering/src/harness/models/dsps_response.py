"""从单次暴露的干空气、响应和稳态温度循环读取一种器件的气体响应。"""
import numpy as np


LOADS = np.array([.10,.25,.40,.87,1.30,3.01,6.19,9.09,21,30.01,40.20,51.10,68.10,82.5,90.90,105.])


def curve_scale(v, loads):
    # 让不同负载互相核对。若 v = A/(R + L)，v*L 对 v 的斜率是 -R。
    centered = v - v.mean(axis=-1, keepdims=True)
    product = v * loads
    r = -np.sum(centered * product, axis=-1) / np.maximum(np.sum(centered**2, axis=-1), 1e-12)
    return np.clip(r, .01, 1e5)


def physical_views(bl, tr, st):
    arrays = [np.asarray(a, dtype=float) for a in (bl,tr,st)]
    if [a.shape[:2] for a in arrays] != [(100,8),(500,8),(500,8)] or any(
            a.ndim != 3 or a.shape[2] != 16 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("单器件输入需要8种传感器、16个负载和各阶段完整温度周期")
    b,t,s = [a.reshape(-1,100,8,16) for a in arrays]
    sv = s.mean(axis=0) - b[0]
    b_log,t_log,s_log = [np.log(curve_scale(v,LOADS)) for v in (b,t,s)]
    steady = s_log.mean(axis=0) - b_log[0]
    early = t_log - b_log
    def bins(v):
        return v.reshape(10,10,8).mean(axis=1).T
    pooled = bins(steady)
    normalized = pooled / max(.02,np.linalg.norm(pooled)/np.sqrt(pooled.size))
    trajectory = np.stack([bins(v) for v in early])
    # 暴露过程中从最初响应到后期响应的变化，与稳态响应同时出现才有意义。
    relative = (trajectory - pooled) / np.maximum(.1,np.abs(pooled))
    multi = sv.reshape(10,10,8,16).mean(axis=1).transpose(1,0,2)
    return {"单负载稳态": bins(sv[:,:,7]).ravel(), "全负载稳态": multi.ravel(),
            "负载曲线稳态": pooled.ravel(), "去总幅度的温度曲线": normalized.ravel(),
            "暴露速度": relative.ravel(),
            "稳态与速度": np.r_[normalized.ravel(),relative.ravel()],
            "干空气对照": bins(b_log[0]).ravel()}
