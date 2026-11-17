"""按已返回的事前预测估计全批混淆矩阵，选择下一批发布办法。"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.research.methods.feedback_query_risk import select
from harness.research.methods.gas_returned_errors import CORRECTIONS, TOOLS

FIXED = "相近读数残差"
ARMS = tuple(dict.fromkeys((*TOOLS, *CORRECTIONS, "固定残差三式均值", "原票与读数残差各半")))
POLICIES = ("均匀覆盖", "混合抽查", "随机", "BADGE")
PRIMARY = "按入选概率估计F1"


def sample(policy, ids, x, source_x, p, budget, seed):
    n = len(x)
    count = min(n,budget)
    if count == 0:
        return [], np.empty(0)
    if policy != "混合抽查":
        view = source_x if policy == "BADGE" else StandardScaler().fit_transform(x).clip(-5,5)
        ix,_ = select(policy, ids, view, p, count, seed, [], [], [])
        # 确定性覆盖和BADGE不提供可用于整批无偏估计的包含概率。
        weight = n/count if policy == "随机" else 1.
        return ix, np.full(count, weight)
    cover_n = count // 2
    ix,_ = select("均匀覆盖",ids,StandardScaler().fit_transform(x).clip(-5,5),p,cover_n,seed,[],[],[])
    selected = set(ix)
    remaining = [int(j) for j in np.argsort(ids,kind="stable") if j not in selected]
    random = np.random.default_rng(seed).choice(remaining,size=count-cover_n,replace=False)
    # 在给定本批的条件下，固定覆盖部分概率为1，其他样本等概率无放回抽取。
    return [*ix,*random.tolist()], np.r_[np.ones(cover_n), np.full(count-cover_n,len(remaining)/(count-cover_n))]


class OfflineScoreEstimate:
    def __init__(self, classes):
        self.classes = classes
        self.confusion = np.zeros((2,len(ARMS),classes,classes))
        self.rows = 0

    def update(self, predictions, y, inverse):
        answers = np.asarray(y)-1
        for i,name in enumerate(ARMS):
            p = predictions[name]-1
            for mat, weight in zip(self.confusion[:,i],(np.ones(len(y)),inverse),strict=True):
                np.add.at(mat,(answers,p),weight)
        self.rows += len(y)

    def estimates(self, index):
        cm = self.confusion[index].copy()
        anchor = ARMS.index(FIXED)
        if self.rows:
            # 将权重按已抽到的样本数归一，避免抽样权重使少数标签看似有全批的信息量。
            cm *= self.rows / cm[anchor].sum()
            cm += 16 * cm[anchor] / self.rows
        diag = np.diagonal(cm,axis1=1,axis2=2)
        denominator = cm.sum(1)+cm.sum(2)
        f1 = np.divide(2*diag,denominator,out=np.zeros_like(diag),where=denominator>0).mean(1)
        recall = np.divide(diag,cm.sum(2),out=np.zeros_like(diag),where=cm.sum(2)>0).mean(1)
        return f1, recall

    def choices(self):
        plain, recall = self.estimates(0)
        corrected, _ = self.estimates(1)
        proposals = {PRIMARY:corrected, "回标直接算F1":plain, "回标按类别平均正确率":recall}
        default = ARMS.index(FIXED)
        return {k: (ARMS[int(np.argmax(v))] if v.max()>v[default]+.005 else FIXED)
                for k,v in proposals.items()}, corrected
