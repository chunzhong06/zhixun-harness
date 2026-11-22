"""从同一趟飞行的相近工况配对加、减油门测点，估算部件响应中的方向分量。"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from harness.learning.models.flight.flight_condition_response import ConditionReadout
from harness.learning.models.flight.flight_masked_readings import combine

FIXED="升降油门配对差与原部件合训"
CORRECTED="统一消除本趟测得的方向响应"
PRIMARY="按本趟配对留出试验决定是否消除方向响应"
CONTROL="同一批配对的均值与原部件合训"


def matches(z,change):
    up=np.flatnonzero(change>.25)
    down=np.flatnonzero(change<-.25)
    if not len(up) or not len(down):return np.empty((0,2),int)
    dist,idx=cKDTree(z[down]).query(z[up],k=min(3,len(down)))
    dist=dist.reshape(len(up),-1)
    idx=idx.reshape(len(up),-1)
    choices=np.argwhere(dist<=.5)
    used_up=set();used_down=set();pairs=[]
    for i,j in choices[np.argsort(dist[dist<=.5],kind="stable")]:
        a,b=up[i],down[idx[i,j]]
        if a not in used_up and b not in used_down:
            used_up.add(a);used_down.add(b);pairs.append((a,b))
    return np.asarray(sorted(pairs),dtype=int).reshape(-1,2)


def slope(delta,gap):
    # 通过原点的成对回归，每个时刻仅使用一次；小样本向零变化收缩。
    return gap@delta/(gap@gap+4.)


def summarize(matrix,indices,points):
    q1,mid,q3=np.percentile(matrix[indices],[25,50,75],axis=1)
    return np.column_stack((mid,q3-q1,points))


def read_throttle(extractor,rows):
    z,all_values=extractor.measurements(rows)
    values=all_values[:,np.r_[0:14,26:38]]
    change=np.r_[0,np.diff(rows[:,16])].clip(-10,10)
    pairs=matches(z,change)
    points=extractor.points
    nearest=np.argsort(cdist(points,z),axis=1)[:,:max(2,int(np.ceil(len(z)/8)))]
    base=summarize(values,nearest,points)
    direction=np.zeros((len(points),values.shape[1]))
    level=direction.copy()
    coef=np.zeros(values.shape[1])
    keep=np.zeros(values.shape[1],dtype=bool)
    if len(pairs)>=4:
        a,b=pairs.T
        delta=values[a]-values[b];gap=change[a]-change[b]
        center=(z[a]+z[b])/2
        near=np.argsort(cdist(points,center),axis=1)[:,:max(3,int(np.ceil(len(center)/6)))]
        direction=np.median((delta/gap[:,None])[near],axis=1)
        level=np.median(((values[a]+values[b])/2)[near],axis=1)
        coef=slope(delta,gap)
        # 按配对序列的前后两半互相验证方向校正，测试半的读数不用于拟合它的系数。
        ix=np.arange(len(gap))
        folds=ix<len(gap)//2
        held=np.empty_like(delta)
        for test in (folds,~folds):
            held[test]=delta[test]-gap[test,None]*slope(delta[~test],gap[~test])
        keep=np.mean(held**2,axis=0)<.95*np.mean(delta**2,axis=0)
    corrected=summarize(values-change[:,None]*coef,nearest,points)
    selected=summarize(values-change[:,None]*(coef*keep),nearest,points)
    return (base,np.column_stack((base,direction)),np.column_stack((base,level)),corrected,selected), (len(pairs),int(keep.sum()))


class ThrottleReaders:
    def fit(self,views,y):
        # 共用一个消除方向响应的模型，现场接受或拒绝各通道的系数，不额外训练寿命模型。
        self.readers=[ConditionReadout().fit(x,y) for x in views[:4]]
        return self

    def predict(self,views):
        base,contrast,mean,corrected,selected=[combine(*m.readings(x)) for x,m in zip(views,[*self.readers,self.readers[-1]],strict=True)]
        return {"不分方向部件模型":base,FIXED:contrast,CONTROL:mean,CORRECTED:corrected,
                "现场接受的通道校正":selected,PRIMARY:(contrast+selected)/2,
                "方向差与统一校正各半":(contrast+corrected)/2,
                "方向差与原工具各半":(contrast+base)/2}
