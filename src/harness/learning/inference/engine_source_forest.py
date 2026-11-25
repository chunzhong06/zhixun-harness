"""用数值数组恢复发动机条件分布。推理只需 NumPy 和当前一行的 24 项读数。"""
import json

import numpy as np


KIND = 'engine-source-distribution'
ARMS = ('conditional_mean', 'conditional_risk', 'balanced', 'bounded_parent')


def regime_keys(x):
    return np.column_stack((np.rint(x[:, 21]), np.round(x[:, 22], 2), np.rint(x[:, 23])))


def normalized(x, lookup, centers, scales):
    k = regime_keys(x)
    if len(x) == 1:
        group = lookup.get(tuple(k[0]), 0)
    else:
        unique, inverse = np.unique(k, axis=0, return_inverse=True)
        group = np.array([lookup.get(tuple(key), 0) for key in unique])[inverse]
    z = np.empty(x.shape, dtype=np.float32)
    z[:, :21] = (x[:, :21] - centers[group]) / scales[group]
    z[:, 21:] = x[:, 21:]
    return z


def combine(arm, parent, mean, risk):
    if arm == 'bounded_parent':
        return np.clip(parent, np.minimum(mean, risk), np.maximum(mean, risk))
    head = {'conditional_mean': mean, 'conditional_risk': risk, 'balanced': (mean+risk)*.5}[arm]
    # 源分布提供校正，真实接续的父网保留单行预测贡献。
    return .125 * parent + .875 * head


class SourceForestModel:
    def __init__(self, base, arrays):
        self.base, self.meta, self.auxiliary_arrays = base, base.meta, arrays
        self.state = json.loads(str(arrays['complete_state']))
        if self.state.get('kind') != KIND or self.state.get('arm') not in ARMS or self.meta.get('profile') != 'C':
            raise ValueError('发动机分布检查点与任务不匹配')
        self.lookup = {tuple(k): i+1 for i,k in enumerate(arrays['forest_keys'])}
        self.c, self.s = arrays['forest_center'], arrays['forest_scale']
        self.roots, self.nodes = arrays['forest_roots'], arrays['forest_nodes']
        self.limits, self.values = arrays['forest_limits'], arrays['forest_values']
        self.depth = self.state['depth']
        n = len(self.nodes)
        if (self.c.shape != (len(self.lookup)+1,21) or self.s.shape != self.c.shape or
                not np.all(np.isfinite(self.c)) or not np.all(np.isfinite(self.s) & (self.s>0)) or
                self.nodes.shape != (n,3) or self.nodes.dtype.kind not in 'iu' or
                self.roots.ndim != 1 or not 0<len(self.roots)<=128 or self.roots.dtype.kind not in 'iu' or
                not np.all((self.roots>=0)&(self.roots<n)) or self.values.shape!=(n,3) or
                len(self.limits)!=n or not np.all(np.isfinite(self.limits)) or
                not np.all(np.isfinite(self.values)) or not np.all(self.values[:,1:]>0) or
                not np.all((self.nodes[:,:2]>=0)&(self.nodes[:,:2]<n)) or
                not np.all((self.nodes[:,2]>=0)&(self.nodes[:,2]<24)) or type(self.depth)!=int or not 0<self.depth<=24):
            raise ValueError('发动机树参数不合法')

    def moments(self, x):
        x = np.asarray(x, dtype=float)
        z = normalized(x, self.lookup, self.c, self.s)
        out = []
        for start in range(0,len(z),512):
            block = z[start:start+512]
            rows = np.arange(len(block))[:,None]
            nodes = np.broadcast_to(self.roots,(len(block),len(self.roots)))
            for _ in range(self.depth):
                feature = self.nodes[nodes,2]
                right = (block[rows,feature] > self.limits[nodes]).astype(np.intp)
                nodes = self.nodes[nodes,right]
            out.append(self.values[nodes].mean(axis=1))
        mean, high, low = np.concatenate(out).T
        return mean, np.maximum(0., (np.log(high)-np.log(low)+np.log(10/13))/(1/10+1/13))

    def predict(self, raw):
        return combine(self.state['arm'], self.base.predict(raw), *self.moments(raw))
