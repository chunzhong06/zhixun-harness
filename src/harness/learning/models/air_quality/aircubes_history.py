"""在现场参考稀疏时，用电极的连续变化补充当前读数，并检验重训的实际收益。"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.learning.models.air_quality.aircubes_precision import PrecisionMaintenance, PRIMARY as PREVIOUS
from harness.learning.models.air_quality.aircubes_changes import INCUMBENT


BENCHMARK = "同工具论文固定差分量0.2"
PRIMARY = "按实际回标检验是否重训连续电极响应"
VIEWS = ("仅当前读数", "半小时及两小时电极历史", "加入六小时电极历史")


class CausalHistory:
    def __init__(self):
        self.state = None
        self.stamp = None

    def design(self, clocks, signals):
        # 中断后重新以第一条有效测量启动，避免把空档误读成一次浓度跃迁。
        raw = signals[:, [0, 1, 5, 6, 2, 3]]
        differences = np.zeros((len(raw), 3, raw.shape[1]))
        decay = np.exp(-10 / np.array([30., 120., 360.]))[:, None]
        for i, (stamp, reading) in enumerate(zip(clocks, raw, strict=True)):
            if self.stamp is None or stamp - self.stamp != 600:
                self.state = np.broadcast_to(reading, (3, len(reading))).copy()
            else:
                self.state = decay * self.state + (1 - decay) * reading
            differences[i] = reading - self.state
            self.stamp = stamp
        a, b, temp, rh, drh, n1, n2, power = signals.T
        core = np.column_stack((np.ones(len(a)), a, b, temp, rh, drh, n1, n2, power,
                                a*temp, b*temp, n1*temp, n2*temp, a*rh, b*rh,
                                n1*a, n2*b, a*a, b*b, temp*temp, rh*rh))
        short = differences[:, :2].reshape(len(a), -1)
        # 分开保留上升与下降的幅度，两种方向可以有不同的灵敏度。
        signed = differences[:, 0, :4]
        fast = np.column_stack((core, short, np.abs(signed), signed*temp[:, None]))
        return core, fast, np.column_stack((fast, differences[:, 2]))


class HistoryRelease:
    def fit(self, clocks, raw, truth):
        self.scale = StandardScaler().fit(raw)
        features = CausalHistory().design(clocks, self.scale.transform(raw).clip(-7, 7))
        self.models = []
        for view in features:
            gram = view.T @ view / len(view) + .01 * np.eye(view.shape[1])
            initial = np.linalg.solve(gram, view.T @ truth / len(view))
            self.models.append((gram, initial))
        return self


class HistoryMaintenance:
    def __init__(self, tools):
        self.old = PrecisionMaintenance(tools)
        self.model = tools["history"]
        self.sensor = CausalHistory()
        self.grams = [np.zeros_like(g) for g, _ in self.model.models]
        self.errors = [np.zeros_like(b) for _, b in self.model.models]
        self.last = None
        self.past_losses = np.zeros(4)
        self.mass = 0.

    def predict(self, clocks, raw, paper):
        previous = self.old.predict(clocks, raw, paper)
        names = (PREVIOUS, BENCHMARK, INCUMBENT, "论文组合与单步浓度变化占0.25",
                 "源端统一差分量", "四模型等权")
        out = {n: previous[n] for n in names if n in previous}
        anchor = out[BENCHMARK]
        features = self.sensor.design(clocks, self.model.scale.transform(raw).clip(-7, 7))
        self.pending = []
        candidates = [anchor]
        for i, (view, (prior, start)) in enumerate(zip(features, self.model.models, strict=True)):
            update = np.linalg.solve(12 * prior + self.grams[i], self.errors[i])
            installed = view @ start
            trained = installed + view @ update
            self.pending.append((view[-1].copy(), installed[-1], trained[-1]))
            if self.last is not None:
                stamp, errors = self.last
                trained += .5*np.exp(-(clocks-stamp)/21600)*errors[i]
            trained = np.maximum(0., trained)
            out[VIEWS[i] + "冻结源模型"] = np.maximum(0., installed)
            out[VIEWS[i] + "回标重训"] = trained
            for fraction in (.25, .5, .75):
                out[f"论文组合与{VIEWS[i]}占{fraction:g}"] = anchor + fraction*(trained-anchor)
            candidates.append(out[f"论文组合与{VIEWS[i]}占0.5"])
        bank = np.column_stack(candidates)
        # 只用此前真实发放、之后收到了参考的预测评估接下来应采用的读法。
        relative = self.past_losses / (15 + self.mass)
        weights = np.exp(-np.clip(relative, -10, 10))
        out[PRIMARY] = bank @ (weights / weights.sum())
        out["保持装机时的连续响应等权选择"] = bank.mean(axis=1)
        self.stamp = clocks[-1]
        self.issued = bank[-1].copy()
        return out

    def update(self, truth):
        self.old.update(truth)
        residuals = []
        for i, (feature, source, actual) in enumerate(self.pending):
            self.grams[i] = .90*self.grams[i] + np.outer(feature, feature)
            self.errors[i] = .90*self.errors[i] + feature*(truth-source)
            residuals.append(np.clip(truth-actual, -12, 12))
        self.last = self.stamp, np.array(residuals)
        losses = np.minimum((truth-self.issued)**2, 625.)
        self.past_losses = .96*self.past_losses + losses-losses[0]
        self.mass = .96*self.mass + 1
