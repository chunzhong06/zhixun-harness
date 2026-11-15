"""用有限回标了解两种测量读法的分歧，并保留覆盖整个服务批的询问名额。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from harness.methods.feedback_query_risk import graded_coverage

POLICIES = ("固定分歧","按历史分歧分配","固定只覆盖")


class DisagreementFeedback:
    def __init__(self):
        self.evidence = []
        self.answers = []

    @staticmethod
    def context(anchor, paper, raw, spread, alternate):
        # 同方向分歧再区分论文连续读数到类别边界的距离、器件间分歧与其他模型的票数。
        return np.column_stack((anchor,paper,raw,spread,alternate))

    def rates(self, x):
        default = np.full(len(x),-.3)
        if not self.evidence:
            return default, np.zeros(len(x))
        prev=np.vstack(self.evidence)
        y=np.concatenate(self.answers)
        same=(prev[None,:,0]==x[:,None,0]) & (prev[None,:,1]==x[:,None,1])
        near=np.exp(-2*abs(x[:,None,2]-prev[None,:,2])-abs(x[:,None,3]-prev[None,:,3])
                    -.5*abs(x[:,None,4]-prev[None,:,4]))
        weight=same*near
        gains=(y==prev[:,1]).astype(float)-(y==prev[:,0]).astype(float)
        return (weight@gains-1.2)/(4+weight.sum(1)),weight.sum(1)

    def publish(self, x):
        a=x[:,0].astype(int); p=x[:,1].astype(int)
        gain,count=self.rates(x)
        supported=(a!=p)&(gain>0)&(count>=2)
        return np.where(supported,p,a)

    def query(self, policy, ids, values, x, budget):
        n=len(ids)
        if budget>=n:return list(range(n))
        if budget<=0:return []
        order=np.argsort(ids,kind="stable")
        z=StandardScaler().fit_transform(values[order]).clip(-5,5)
        d=cdist(z,z,"sqeuclidean")/max(1,z.shape[1])
        nz=d[d>0]; h=np.median(nz) if len(nz) else 1.
        similarity=np.exp(-d/max(h,1e-7))
        # 同一距离、同八个名额。固定分歧和现场分歧都可见相同的两种预测。
        ctx=x[order]; disagree=(ctx[:,0]!=ctx[:,1])
        if policy=="固定只覆盖":
            demand=np.ones(n)
        elif policy=="固定分歧":
            demand=1+4*disagree
        elif policy=="按历史分歧分配":
            gain,count=self.rates(ctx)
            value=np.maximum(0,.45+gain)+.6/np.sqrt(1+count)
            demand=1+5*disagree*value
        else:raise ValueError("未知分歧查询策略")
        return order[graded_coverage(similarity,demand,budget)].tolist()

    def receive(self,x,y):
        self.evidence.append(x.copy());self.answers.append(y.copy())
