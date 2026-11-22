"""拟合当轮的短寿命概率，发布时只保存树的分裂条件与叶概率。"""
import numpy as np


def fit(x, y, factor):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import RobustScaler

    cutoff = float(np.quantile(y,.2))
    short = y<=cutoff
    if min(short.sum(),(~short).sum())<6:
        return None
    z = np.sign(x)*np.log1p(abs(x)/factor)
    low,high = np.quantile(z,[.01,.99],axis=0)
    clipped = np.clip(z,low,high)
    scaler = RobustScaler().fit(clipped)
    # 较大训练批次限制树的叶数，避免交付产物随训练行数线性膨胀。
    forest = ExtraTreesClassifier(n_estimators=96,min_samples_leaf=3,random_state=42,n_jobs=1,
                                  max_leaf_nodes=64 if len(y)>384 else None)
    forest.fit(scaler.transform(clipped),short)
    trees = []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        counts = tree.value[:,0,:]
        probability = counts[:,1]/counts.sum(axis=1)
        trees.append({"left":tree.children_left.tolist(),"right":tree.children_right.tolist(),
                      "feature":tree.feature.tolist(),"threshold":tree.threshold.tolist(),"prob":probability.tolist()})
    return {"format":"bearing-short-life-extra-trees-v1","cutoff":cutoff,"prevalence":float(short.mean()),
            "short_mean":float(y[short].mean()),"low":low.tolist(),"high":high.tolist(),
            "center":scaler.center_.tolist(),"scale":scaler.scale_.tolist(),"trees":trees}


class TerminalReadout:
    def __init__(self, spec, factor):
        self.factor = factor
        self.low,self.high,self.center,self.scale = (np.asarray(spec[k],dtype=float) for k in ("low","high","center","scale"))
        self.prevalence,self.short_mean = float(spec["prevalence"]),float(spec["short_mean"])
        if (spec.get("format")!="bearing-short-life-extra-trees-v1"
                or not all(a.shape==factor.shape and np.isfinite(a).all() for a in (self.low,self.high,self.center,self.scale))
                or (self.low>self.high).any() or (self.scale<=0).any() or not 0<self.prevalence<1
                or not np.isfinite(spec["cutoff"]) or not 0<=self.short_mean<=spec["cutoff"] or len(spec["trees"])!=96):
            raise ValueError("轴承短寿命概率参数无效")
        self.trees = []
        for tree in spec["trees"]:
            left,right,feature = (np.asarray(tree[k]) for k in ("left","right","feature"))
            threshold,prob = (np.asarray(tree[k],dtype=float) for k in ("threshold","prob"))
            nodes = np.arange(len(left))
            if (not 1<=len(nodes)<=255 or any(a.shape!=nodes.shape for a in (left,right,feature,threshold,prob))
                    or any(a.dtype.kind not in "iu" for a in (left,right,feature))
                    or not np.isfinite(threshold).all() or not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any()):
                raise ValueError("短寿命概率树结构无效")
            internal = left!=-1
            if ((right[~internal]!=-1).any() or (feature[internal]<0).any() or (feature[internal]>=len(factor)).any()
                    or any(((child[internal]<=nodes[internal])|(child[internal]>=len(nodes))).any() for child in (left,right))):
                raise ValueError("短寿命概率树的子节点无效")
            self.trees.append((left,right,feature,threshold,prob))

    def probability(self, raw):
        z = np.sign(raw)*np.log1p(np.abs(raw)/self.factor)
        # sklearn 的树比较 float32 输入和双精度分裂阈值。
        x = ((np.clip(z,self.low,self.high)-self.center)/self.scale).astype(np.float32)
        total = np.zeros(len(raw))
        for left,right,feature,threshold,prob in self.trees:
            node = np.zeros(len(raw),dtype=int)
            rows = np.flatnonzero(left[node]!=-1)
            while len(rows):
                active = node[rows]
                go_left = x[rows,feature[active]]<=threshold[active]
                node[rows] = np.where(go_left,left[active],right[active])
                rows = rows[left[node[rows]]!=-1]
            total += prob[node]
        return total/len(self.trees)

    def apply(self, raw, original):
        weight = np.clip((self.probability(raw)-self.prevalence)/(1.-self.prevalence),0.,1.)
        active = original>self.short_mean
        return original-weight*np.maximum(original-self.short_mean,0.), 1.-weight*active
