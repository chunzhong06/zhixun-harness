"""在已知释放上构造衰减读数，并用留出的真实标气决定哪些预测值得改判。"""
import numpy as np
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


GRID = np.arange(30, 3001) / 10
VIEWS = ("原释放训练", "加入向本次空气衰减的释放", "加入向同背景标气衰减的释放")
PRIMARY = "靠装机弱释放的纠错收益决定是否改判"
ANCHOR = "原去噪固定"


def sample_cycle(rows):
    rows = np.asarray(rows)
    if (rows.ndim != 2 or rows.shape[1] != 11 or not np.isfinite(rows).all()
            or np.any(np.diff(rows[:, 0]) < 0) or rows[0, 0] > 10 or rows[-1, 0] < 285
            or np.any((rows[:, 3:] <= 0) | (rows[:, 3:] >= 3110))):
        raise ValueError("风洞原始时间范围或传感器读数异常")
    clock = rows[:, 0]
    right = np.searchsorted(clock, GRID).clip(0, len(clock)-1)
    left = np.maximum(0, right-1)
    index = np.where(np.abs(clock[left]-GRID) <= np.abs(clock[right]-GRID), left, right)
    raw = rows[index, 3:]
    return np.log(raw / (3110-raw))


def air_signal(x):
    pre = x[:, GRID < 60]
    center = np.median(pre, axis=1, keepdims=True)
    noise = pre - center
    offsets = (np.arange(len(GRID)) - len(pre[0])) % len(pre[0])
    return x-center, noise[:, offsets], center


def features(x):
    response, _, _ = air_signal(x)
    frames = response[:, (GRID >= 60) & (GRID < 240)].reshape(len(x), 18, 100, 8).mean(2)
    # 保留八路共同的时间顺序。比例特征与按绝对幅度读数一起交给两边的训练器。
    shape = frames / (.02 + np.linalg.norm(frames, axis=2, keepdims=True))
    rest = response[:, (GRID >= 240) & (GRID < 285)].mean(1)
    return np.column_stack((frames.reshape(len(x),3,6,8).mean(2).reshape(len(x),-1),
                            np.quantile(frames,.85,axis=1),shape.mean(1),rest))


def context(x):
    response, _, _ = air_signal(x)
    active = response[:, (GRID >= 60) & (GRID < 240)]
    shape = active.mean(1)
    size = np.linalg.norm(shape, axis=1, keepdims=True)
    air = np.std(response[:, GRID < 60], axis=1).mean(1)
    return np.column_stack((np.log(.01+size[:,0]), np.log(.001+air), shape/(.02+size)))


def augment(x, y, groups, view):
    """背景参照只从当前训练折取；同配方留出的真实释放从所有合成父本中排除。"""
    if view == VIEWS[0]:
        return x, y
    signal, air, center = air_signal(x)
    ref = air.copy()
    if view == VIEWS[2]:
        # 对有匹配背景的正样本，保留背景气体的响应；纯乙烯及无匹配背景的样本用本次空气。
        for j in np.flatnonzero(y == 2):
            match = (y == 1) & np.all(groups[:,1:] == groups[j,1:], axis=1)
            if match.any():
                ref[j] = signal[match].mean(0)
    parts = [x]
    for strength in (.2, .5, .8):
        blend = center + strength*signal + (1-strength)*ref
        blend[:, GRID < 60] = x[:, GRID < 60]
        parts.append(blend)
    return np.concatenate(parts), np.tile(y, len(parts))


class WeakTraining:
    def fit(self, x, y, groups, view):
        expanded, truth = augment(x, y, groups, view)
        feat = features(expanded)
        self.scale = StandardScaler().fit(feat)
        train = self.scale.transform(feat)
        _, ids = np.unique(groups, axis=0, return_inverse=True)
        original = train[:len(y)]
        means = np.stack([original[ids == i].mean(0) for i in range(ids.max()+1)])
        noise = original - means[ids]
        fallback = original - original.mean(0)
        self.whitener = np.linalg.cholesky(OAS(assume_centered=True).fit(noise if np.any(noise) else fallback).precision_)
        self.clf = LogisticRegression(class_weight="balanced", C=1, max_iter=1500).fit(train @ self.whitener, truth)
        return self

    def predict(self, x):
        return self.clf.predict_proba(self.scale.transform(features(x)) @ self.whitener)[:, 1]


class WeakInstallation:
    def fit(self, x, y, groups, anchor_history):
        _, code = np.unique(groups,axis=0,return_inverse=True)
        self.tools = [WeakTraining().fit(x,y,groups,v) for v in VIEWS]
        # 保留同一背景的其他浓度，整种配方均未参与其验证预测或合成训练。
        held = np.empty((len(y),len(VIEWS)))
        for fold in range(5):
            test = code % 5 == fold
            for j, view in enumerate(VIEWS):
                model = WeakTraining().fit(x[~test], y[~test], groups[~test], view)
                held[test,j] = model.predict(x[test])
        self.anchor_history = anchor_history
        self.candidates = held
        self.truth = y
        self.scale = StandardScaler().fit(context(x))
        self.calibration = self.scale.transform(context(x))
        return self

    def predict(self, x, anchor):
        p = np.column_stack([m.predict(x) for m in self.tools])
        distance = self.scale.transform(context(x))[:,None] - self.calibration[None]
        # 只改判在相似装机释放中能净纠错的分歧。类别不平衡，以类内出现率加权。
        near = np.exp(-.5*np.mean(distance**2,axis=2))
        near *= np.array([1/np.mean(self.truth == val) for val in self.truth])[None]
        base_correct = self.anchor_history == self.truth
        delta = ((1+(self.candidates>=.5)) == self.truth[:,None]).astype(float) - base_correct[:,None]
        gain = near @ delta
        winner = gain.argmax(1)
        current = 1 + (p[np.arange(len(p)),winner] >= .5)
        pred = {name: 1+(p[:,j]>=.5) for j,name in enumerate(VIEWS)}
        pred[PRIMARY] = np.where(gain.max(1)>0,current,anchor)
        pred["三种训练固定平均"] = 1+(p.mean(1)>=.5)
        return pred
