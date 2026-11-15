"""用已交付、已回标的预测训练残差修正，下一批作答时决定需要保留多少修正。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

TOOLS = ("原刻度", "独立刻度", "只用现场", "固定合训核", "固定现场核", "现场秩等权与合训核", "两域秩固定各半与合训核")
ANCHOR = TOOLS[-1]
CORRECTIONS = ("不改原票", "全局残差", "相近读数残差", "相同预测组合残差", "相近读数直接训练")
PRIMARY = "按已回标的逐批收益选修正"


def base_probability(votes, classes):
    encoded = np.eye(classes)[votes - 1]
    return .55 * encoded[:, -1] + .45 * encoded.mean(1)


class ReturnedErrors:
    def __init__(self, classes):
        self.classes = classes
        self.x, self.votes, self.y, self.scores = [], [], [], []

    def predict(self, x, votes):
        base = base_probability(votes, self.classes)
        batch = np.repeat(base[:, None, :], len(CORRECTIONS), axis=1)
        if self.y:
            past = np.vstack(self.x)
            pv = np.vstack(self.votes)
            y = np.concatenate(self.y)
            target = np.eye(self.classes)[y - 1]
            residual = target - base_probability(pv, self.classes)
            scale = StandardScaler().fit(np.vstack((x, past)))
            ds = cdist(scale.transform(x), scale.transform(past), metric="sqeuclidean") / x.shape[1]
            similar_vote = np.mean(votes[:, None, :] == pv[None, :, :], axis=2)
            local = np.exp(-ds / .6) * (.2 + .8 * similar_vote)
            same = similar_vote ** 5
            # 少量近邻向未修正的原预测收缩；训练只使用当时已收到的合法标签。
            batch[:, 1] += residual.sum(0) / (len(y) + 8)
            batch[:, 2] += (local @ residual) / (local.sum(1)[:, None] + 2)
            batch[:, 3] += (same @ residual) / (same.sum(1)[:, None] + 3)
            batch[:, 4] = (local @ target + 2 * base) / (local.sum(1)[:, None] + 2)
        batch = np.maximum(batch, .01)
        batch /= batch.sum(2, keepdims=True)
        chosen = 0
        if self.y:
            seen = np.vstack(self.scores)
            yy = np.concatenate(self.y)
            # 回标时比较各修正当时真实的误差。类别等权，证据少的类别向原票收缩。
            losses = np.mean((seen - np.eye(self.classes)[yy - 1, None, :]) ** 2, axis=2)
            loss = np.mean([(losses[yy == t].sum(0) + 8 * losses[yy == t, 0].mean()) /
                            (sum(yy == t) + 8) for t in np.unique(yy)], axis=0)
            chosen = int(np.argmin(loss))
        p = {name: batch[:, i] for i, name in enumerate(CORRECTIONS)}
        p["固定残差三式均值"] = batch[:, 1:4].mean(1)
        p["原票与读数残差各半"] = (batch[:, 0] + batch[:, 2]) / 2
        p[PRIMARY] = batch[:, chosen]
        return {k: (v.argmax(1) + 1).tolist() for k,v in p.items()}, batch, chosen

    def receive(self, x, votes, labels, before):
        self.x.append(x)
        self.votes.append(votes)
        self.y.append(labels)
        self.scores.append(before)
