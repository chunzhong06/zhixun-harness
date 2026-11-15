"""用源端仿真故障标注训练寿命模型。部署时仅访问当前航班的三段传感器摘要。"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

FAULTS = ("高压涡轮", "两级涡轮", "风扇", "高压压气机")
PRIMARY = "按当前故障概率与统一模型组合"
ANCHOR = "直接用树方差倒数"


def fit_tree(x, y):
    return ExtraTreesRegressor(n_estimators=96, min_samples_leaf=5, n_jobs=1, random_state=17).fit(x, y)


def segment_prediction(tree, views):
    flat = views.reshape(-1, views.shape[-1])
    outputs = np.stack([t.predict(flat).reshape(views.shape[:2]) for t in tree.estimators_])
    weight = (1 + outputs.std(0)) ** -2
    return (weight * outputs.mean(0)).sum(1) / weight.sum(1)


class FaultExperts:
    def fit(self, views, labels, fault):
        x = views.reshape(-1, views.shape[-1])
        mode = np.repeat(fault, 3)
        y = np.repeat(labels, 3)
        if set(fault) != set(range(len(FAULTS))):
            raise ValueError("故障专家的源端训练类型不完整")
        self.frequency = np.bincount(fault) / len(fault)
        self.gate = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=8, random_state=17,
                                         class_weight="balanced", n_jobs=1).fit(x, mode)
        self.experts = [fit_tree(x[mode == i], y[mode == i]) for i in range(len(FAULTS))]
        return self

    def outputs(self, views, anchor):
        flat = views.reshape(-1, views.shape[-1])
        prob = self.gate.predict_proba(flat).reshape(len(views), 3, -1).mean(axis=1)
        estimates = np.stack([segment_prediction(tree, views) for tree in self.experts], axis=1)
        mixed = np.sum(prob * estimates, axis=1)
        entropy = -np.sum(prob * np.log(prob.clip(1e-10)), axis=1)
        trust = np.clip(1 - entropy / np.log(len(FAULTS)), 0, 1)
        fixed = estimates @ self.frequency
        out = {name + "专长": estimates[:, i] for i, name in enumerate(FAULTS)}
        out.update({"专长固定均值": estimates.mean(1), "专长按源比例": fixed,
                    "统一和源比例各半": (anchor + fixed) / 2, "按当前故障概率": mixed,
                    PRIMARY: anchor + trust * (mixed - anchor)})
        return out, prob
