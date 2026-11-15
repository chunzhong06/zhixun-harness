"""只从单一合法源批估计各类变化方向，供同预算的续训对照使用。"""
import numpy as np


def box_distance(point, direction):
    # 官包在标准化后截断至 [-10,10]。零分量不会限制位移。
    nonzero = direction != 0.
    boundary = np.copysign(10.,direction[nonzero])
    return float(np.min((boundary-point[nonzero])/direction[nonzero]))


class ClassDirections:
    def __init__(self, rows):
        if any(r.role!="fit" or r.label is None for r in rows):
            raise ValueError("同类方向只能使用训练行")
        self.x = np.asarray([r.values for r in rows])
        self.y = np.asarray([r.label-1 for r in rows])
        self.last_diagnostics = {}

    def inputs(self, model, seed):
        z = model.normalized(self.x)
        axes, deviations, fractions = {}, {}, {}
        for g in np.unique(self.y):
            group = z[self.y==g]
            _,s,vt = np.linalg.svd(group-group.mean(0),full_matrices=False)
            # 统一奇异向量的符号，种子控制的正负号不会随线性代数库翻转。
            axis = vt[0]
            axes[g] = axis*np.sign(axis[np.argmax(np.abs(axis))])
            deviations[g] = s[0]/np.sqrt(len(group)-1)
            fractions[str(g+1)] = float(s[0]**2/np.sum(s**2))
        rng=np.random.default_rng(np.random.SeedSequence([seed,model.meta["epochs"]]))
        out={k:self.x.copy() for k in ("principal","random")}
        lengths=[]
        for i in rng.permutation(len(z))[:len(z)//2]:
            d=axes[self.y[i]]*rng.choice([-1.,1.])
            u=rng.normal(size=len(d))
            u/=np.linalg.norm(u)
            length=min(.5*deviations[self.y[i]],box_distance(z[i],d),box_distance(z[i],u))
            if length<=1e-12:
                continue
            for name,vector in (("principal",d),("random",u)):
                out[name][i]=(z[i]+length*vector)*model.scale+model.mean
            lengths.append(length)
        norm=[np.linalg.norm(model.normalized(a)-z,axis=1) for a in out.values()]
        if not np.allclose(*norm,rtol=1e-9,atol=1e-9):
            raise ValueError("两种方向在原模型所见坐标中的强度不同")
        self.last_diagnostics[str(seed)]={
            "variance_fraction":fractions,"moved":len(lengths),"mean_length":float(np.mean(lengths)) if lengths else 0.,
        }
        return out,len(lengths)
