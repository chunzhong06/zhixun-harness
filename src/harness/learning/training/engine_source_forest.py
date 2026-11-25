"""在合法源机组上诚实分割：一半学习树，另一半按设备等权估计叶中寿命。"""
import json
import re

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from harness.learning.inference.engine_source_forest import KIND, normalized, regime_keys


def device_rows(ids, y):
    matches = [re.fullmatch(r'(.+)_c([0-9]+)', str(key)) for key in ids]
    if not all(matches):
        return None
    names, inverse = np.unique([m[1] for m in matches], return_inverse=True)
    cycles = np.asarray([int(m[2]) for m in matches])
    ends = cycles+y
    lo, hi = np.full(len(names), np.inf), np.full(len(names), -np.inf)
    np.minimum.at(lo,inverse,ends)
    np.maximum.at(hi,inverse,ends)
    if (len(names)<4 or np.any(hi-lo>1e-6) or np.any(y<0) or np.max(y)>5000 or
            len(np.unique(np.column_stack((inverse,cycles)),axis=0))!=len(ids)):
        return None
    keep = np.flatnonzero(cycles % 3 == 1)
    groups, labels, counts = np.unique(inverse[keep],return_inverse=True,return_counts=True)
    if len(keep)<192 or len(groups)<4:
        return None
    return keep, labels, counts


def fit_arrays(x, y, grouping, source_digest):
    keep, inverse, counts = grouping
    x, y = x[keep], y[keep]
    keys, group, freq = np.unique(regime_keys(x),axis=0,return_inverse=True,return_counts=True)
    chosen = np.flatnonzero(freq>=20)
    subsets = [x[:,:21]]+[x[group==i,:21] for i in chosen]
    centers = np.stack([s.mean(0) for s in subsets])
    scales = np.stack([s.std(0) for s in subsets])
    scales[scales<=1e-8] = 1.
    codes = keys[chosen]
    z = normalized(x,{tuple(k):i+1 for i,k in enumerate(codes)},centers,scales)
    order = np.random.default_rng(42).permutation(len(counts))
    first = np.isin(inverse,order[:len(order)//2])
    roots, nodes, limits, values = [], [], [], []
    offset, depth = 0, 0
    for split in (first,~first):
        if min(split.sum(), (~split).sum())<24:
            raise ValueError('源机组整组划分后训练量不足')
        forest = ExtraTreesRegressor(n_estimators=32,max_depth=14,min_samples_leaf=12,
                                     max_features=.85,n_jobs=4,random_state=42)
        forest.fit(z[split],y[split],sample_weight=1/counts[inverse[split]])
        target, weights = y[~split],1/counts[inverse[~split]]
        moments = np.column_stack((np.ones(len(target)),target,np.exp(target/13),np.exp(-target/10)))
        for tree in forest.estimators_:
            t = tree.tree_
            leaf = tree.apply(z[~split])
            mass = np.column_stack([np.bincount(leaf,weights*col,minlength=t.node_count) for col in moments.T])
            samples = np.bincount(leaf,minlength=t.node_count)
            branches = np.flatnonzero(t.children_left>=0)
            for j in branches[::-1]:
                a,b = t.children_left[j],t.children_right[j]
                mass[j],samples[j] = mass[a]+mass[b],samples[a]+samples[b]
            for j in branches:
                for child in (t.children_left[j],t.children_right[j]):
                    if samples[child]<12:
                        mass[child] = mass[j]
            ix = np.arange(t.node_count)
            nodes.append(np.column_stack((offset+np.where(t.children_left<0,ix,t.children_left),
                                          offset+np.where(t.children_right<0,ix,t.children_right),np.maximum(0,t.feature))))
            roots.append(offset)
            offset += t.node_count
            depth = max(depth,t.max_depth)
            limits.append(t.threshold)
            values.append(mass[:,1:]/mass[:,:1])
    state = dict(kind=KIND, format='harness.complete-checkpoint.v1', arm='conditional_mean',
                 depth=int(depth), source=source_digest)
    return dict(complete_state=np.array(json.dumps(state,sort_keys=True)),forest_keys=codes,
                forest_center=centers,forest_scale=scales,forest_roots=np.array(roots,dtype=np.int32),
                forest_nodes=np.concatenate(nodes).astype(np.int32),forest_limits=np.concatenate(limits),
                forest_values=np.concatenate(values))
