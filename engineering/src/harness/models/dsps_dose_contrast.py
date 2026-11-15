"""安装时用同浓度的不同气体学习类别差异。新暴露在未知浓度下作答。"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness.models.dsps_cycle_release import CalibratedCycleRelease, DirectCalibration, kernel


class DoseContrasts:
    def fit(self, x, y, levels):
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.reference = z
        self.levels = levels
        self.grid = np.unique(levels)
        self.centers = np.stack([z[levels == level].mean(0) for level in self.grid])
        self.classes = np.unique(y)
        self.prototypes = np.stack([
            [z[(levels == level) & (y == label)].mean(0) for label in self.classes]
            for level in self.grid
        ])
        self.residual = kernel(z - self.centers[np.searchsorted(self.grid, levels)], y)
        return self

    def scores(self, x):
        z = self.scaler.transform(x)
        # 只用本条目标读数定位浓度。源端浓度只在安装时可见。
        distances = cdist(z, self.reference, metric="sqeuclidean")
        near = np.stack([distances[:, self.levels == v].min(1) for v in self.grid], axis=1)
        level = near.argmin(1)
        residual = self.residual.decision_function(z - self.centers[level])
        delta = z[:, None, :] - self.prototypes[level]
        template = -np.mean(delta ** 2, axis=2)
        template -= template.mean(1, keepdims=True)
        return residual, template, self.grid[level]


PAPER = "直接标准化"
RESIDUAL = "扣除浓度共有响应"
TEMPLATE = "同浓度类别参照"
JOINT = "三路固定合训"
EQUAL = "三路固定等权"
PRIMARY = "按装机留组表现选读法"


def decode(scores, y):
    return LogisticRegression(C=1, max_iter=500).fit(scores, y)


class InstalledDoseRelease:
    def fit(self, source, target, y, groups, levels):
        self.grid = np.unique(levels)
        a = CalibratedCycleRelease.steady(source)
        b = CalibratedCycleRelease.steady(target)
        def fit_readers(mask):
            return (DirectCalibration().fit(a[mask], b[mask], y[mask]),
                    DoseContrasts().fit(b[mask], y[mask], levels[mask]))
        def read_pair(readers, x):
            direct, contrast = readers
            residual, template, level = contrast.scores(x)
            return np.column_stack((direct.scores(x), residual, template)), level
        held = np.empty((len(y), 9))
        estimated = np.empty_like(levels)
        for group in np.unique(groups):
            test = groups == group
            held[test], estimated[test] = read_pair(fit_readers(~test), b[test])
        self.readers = fit_readers(np.ones(len(y), dtype=bool))
        self.arms = {PAPER: slice(0,3), RESIDUAL: slice(3,6), TEMPLATE: slice(6,9), JOINT: slice(0,9)}
        self.decoders = {k: decode(held[:, sl], y) for k, sl in self.arms.items()}
        # 这里第二层分类器单独留组，但它使用的历史分数来自同一套交叉预测。
        # 两次装机重复时，另一组的分数生成器曾训练于本组；这不是整套工具的独立试用。
        correct = np.empty((len(y), len(self.arms)), dtype=float)
        for col, sl in enumerate(self.arms.values()):
            for group in np.unique(groups):
                test = groups == group
                m = decode(held[~test, sl], y[~test])
                correct[test, col] = m.predict(held[test, sl]) == y[test]
        self.selections = {}
        for level in self.grid:
            w = np.exp(-.65 * (estimated - level)**2) + .07
            self.selections[int(level)] = list(self.arms)[np.average(correct, weights=w, axis=0).argmax()]
        return self

    def probabilities(self, features):
        b = CalibratedCycleRelease.steady(features)
        direct, contrast = self.readers
        residual, template, levels = contrast.scores(b)
        bank = np.column_stack((direct.scores(b), residual, template))
        probs = {k: m.predict_proba(bank[:, self.arms[k]]) for k,m in self.decoders.items()}
        probs[EQUAL] = np.mean([probs[k] for k in (PAPER, RESIDUAL, TEMPLATE)], axis=0)
        return probs, levels

    def predict(self, features):
        probs, levels = self.probabilities(features)
        pred = {k: self.decoders[PAPER].classes_[p.argmax(1)] for k,p in probs.items()}
        pred[PRIMARY] = np.array([pred[self.selections[int(level)]][i] for i, level in enumerate(levels)])
        return pred, levels


TRIAL = "用装机暴露独立试用后纠正等权结果"
SIMPLE = "用装机试用分选择固定配比"
TOOLS = (PAPER, RESIDUAL, TEMPLATE)


def fixed_mixtures():
    weights = {EQUAL: np.full(3, 1/3)}
    for i in range(5):
        for j in range(5-i):
            v = (i, j, 4-i-j)
            weights["固定配比" + "/".join(map(str,v))] = np.asarray(v)/4
    return weights


class InstallationTrials:
    """用装机资料模拟多次试用。整条被考核的暴露从两级训练中同时移出。"""
    def fit(self, source, target, y, groups, levels):
        pairs = np.unique(groups)
        grid = np.unique(levels)
        if len(grid) != 6 or not 2 <= len(pairs) <= len(grid):
            raise ValueError("安装试用需要六个浓度和至少两个采集重复")
        self.mixtures = fixed_mixtures()
        self.bank = InstalledDoseRelease().fit(source,target,y,groups,levels)
        trial_probs = np.empty((len(y), 3, 3))
        self.trial_levels = np.empty_like(levels)
        assigned = np.zeros(len(y), dtype=int)
        for i in range(len(grid)):
            test = np.zeros(len(y), dtype=bool)
            for offset, group in enumerate(pairs):
                test |= (groups == group) & (levels == grid[(i+offset) % len(grid)])
            train = ~test
            trial = InstalledDoseRelease().fit(source[train],target[train],y[train],groups[train],levels[train])
            prob, self.trial_levels[test] = trial.probabilities(target[test])
            trial_probs[test] = np.stack([prob[k] for k in TOOLS],axis=1)
            assigned[test] += 1
        if not np.all(assigned == 1):
            raise ValueError("装机试用未恰好覆盖每一次物理暴露")
        self.trial_truth = y.copy()
        self.trial_pred = 1 + trial_probs.argmax(2)
        self.trial_equal = 1 + trial_probs.mean(1).argmax(1)
        # 固定配比的选型也获得同一批试用结果，以平均真类概率打破并列。
        def score(weight):
            p = np.einsum("ntc,t->nc",trial_probs,weight)
            return (np.mean(p.argmax(1)+1 == y), p[np.arange(len(y)),y-1].mean())
        self.best = max(self.mixtures,key=lambda k:score(self.mixtures[k]))
        return self

    def predict(self, features):
        probs, levels = self.bank.probabilities(features)
        p = np.stack([probs[k] for k in TOOLS],axis=1)
        base = {k:1+np.einsum("ntc,t->nc",p,v).argmax(1) for k,v in self.mixtures.items()}
        base[PAPER] = 1+probs[PAPER].argmax(1)
        base[JOINT] = 1+probs[JOINT].argmax(1)
        base[SIMPLE] = base[self.best]
        field = base[EQUAL].copy()
        votes = 1+p.argmax(2)
        for i, (answer, level) in enumerate(zip(base[EQUAL],levels,strict=True)):
            # 相同答案、相似浓度且某工具确实持异议时，检验过去换答案的净收益。
            best_net = 1.
            for tool, alternative in enumerate(votes[i]):
                if alternative == answer:
                    continue
                matches = (self.trial_equal==answer) & (self.trial_pred[:,tool]==alternative)
                evidence = np.exp(-.35*(self.trial_levels-level)**2) * matches
                gains = (self.trial_truth==alternative).astype(float) - (self.trial_truth==answer)
                net = np.dot(evidence,gains)
                if net > best_net and evidence.sum() >= 2:
                    field[i], best_net = alternative, net
        base[TRIAL] = field
        return base, levels
