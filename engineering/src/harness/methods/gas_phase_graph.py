"""在当前可见的现场样本上沿相似电阻传播，边界只含过去批次的真实回标。"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

LEVELS = (.2,1.,4.)
PRIMARY = "现场用回标检验传播强度"
PAPER = "调和图传播"


def propagate(features, prior, anchors, labels):
    n,classes=prior.shape
    if not len(anchors):
        return {k:prior.copy() for k in (*LEVELS,PAPER)}
    z=StandardScaler().fit_transform(features)
    count=min(13,n)
    distances,ids=NearestNeighbors(n_neighbors=count).fit(z).kneighbors(z)
    sigma=np.maximum(distances[:,-1],.001)
    val=np.exp(-(distances**2)/(sigma[:,None]*sigma[ids]))
    w=csr_matrix((val.ravel(),(np.repeat(np.arange(n),count),ids.ravel())),shape=(n,n))
    w=w.maximum(w.T)
    w.setdiag(0)
    w.eliminate_zeros()
    degree=np.asarray(w.sum(1)).ravel()
    lap=diags(degree)-w
    labeled=np.zeros(n,dtype=bool)
    labeled[anchors]=True
    unknown=np.flatnonzero(~labeled)
    # 未标部分解带标边界的图方程；同一传播强度可作固定规则或由历史回标选用。
    block=lap[unknown][:,unknown].tocsc()
    truth=np.eye(classes)[labels-1]
    fixed_term=w[unknown][:,anchors]@truth
    output={}
    for key,lam in [(a,a) for a in LEVELS]+[(PAPER,1e-7)]:
        tether=lam*degree[unknown].clip(1e-8)
        f=spsolve(block+diags(tether,format="csc"),fixed_term+tether[:,None]*prior[unknown])
        result=prior.copy()
        result[anchors]=truth
        result[unknown]=f
        output[key]=result
    return output
