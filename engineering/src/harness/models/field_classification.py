"""现场分类工具在发放后才取得标签；同一份预测和回标支撑固定、首选和连续选择。"""
from hashlib import sha256
from pathlib import Path
from time import process_time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from harness.models.gas_feedback_bank import fit_discriminant
from harness.models.observer_errors import ObserverErrors


TOOLS = ("observer", "lda", "mixed", "recent-tree")
MODES = (*TOOLS, "once", "ongoing", "recent-backtest")


def implementation():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def weights(y, mass):
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return mass / (len(counts)*counts[inverse])


def fixed_snapshot(tool, fitted):
    if tool == "lda":
        center, scale, coefficient, intercept, classes = fitted
        return {"tool": tool, "center": center.tolist(), "scale": scale.tolist(),
                "coef": coefficient.T.tolist(), "intercept": intercept.tolist(), "classes": classes.tolist()}
    if tool == "mixed":
        scaler, reg = fitted
        return {"tool": tool, "center": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": reg.coef_.tolist(), "intercept": reg.intercept_.tolist(), "classes": reg.classes_.tolist()}
    if tool == "recent-tree":
        tree = fitted.estimators_[0].tree_
        return {"tool": tool, "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                "features": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                "values": tree.value[:,0,:].tolist(), "classes": fitted.classes_.tolist()}
    raise ValueError("无法封存未登记的分类器")


def model_scores(record, values, class_count):
    """JSON 中只恢复数值参数；输出列按 1..C 排列，单类回标树仍交全宽结果。"""
    x = np.asarray(values,dtype=float)
    classes = np.asarray(record["classes"],dtype=int)
    if record["tool"] == "recent-tree":
        tree_input = x.astype(np.float32)
        nodes = np.zeros(len(x),dtype=int)
        left, right = np.asarray(record["left"]), np.asarray(record["right"])
        features, threshold = np.asarray(record["features"]), np.asarray(record["threshold"])
        for _ in range(len(left)):
            live = np.flatnonzero(left[nodes] >= 0)
            if not len(live):
                break
            n = nodes[live]
            nodes[live] = np.where(tree_input[live,features[n]] <= threshold[n],left[n],right[n])
        p = np.asarray(record["values"])[nodes]
        p = p / p.sum(axis=1,keepdims=True)
    else:
        z = (x-np.asarray(record["center"]))/np.asarray(record["scale"])
        logits = z @ np.asarray(record["coef"]).T + np.asarray(record["intercept"])
        if record["tool"] == "mixed" and len(classes) == 2:
            logits = np.column_stack([np.zeros(len(x)), logits[:,0]])
        p = np.exp(logits-logits.max(axis=1,keepdims=True))
        p /= p.sum(axis=1,keepdims=True)
    full = np.zeros((len(x),class_count))
    full[:,classes-1] = p
    return full


def macro(matrix):
    den = matrix.sum(0) + matrix.sum(1)
    return np.divide(2*matrix.diagonal(),den,out=np.zeros(len(den)),where=den>0).mean()


class FieldClassifier:
    """接收方先确认交付，上层才调用 update。可从保存状态重建全部现场候选和决策。"""
    def __init__(self, x, y, bank_names, classes, mode, state=None):
        if mode not in MODES:
            raise ValueError("未知现场决策模式")
        self.source = np.asarray(x,dtype=float)
        self.labels = np.asarray(y,dtype=int)
        self.classes = classes
        self.mode = mode
        if (self.source.ndim!=2 or len(x)!=len(y) or not np.isfinite(self.source).all()
                or set(self.labels) != set(range(1,classes+1))):
            raise ValueError("源模型需要完整类别和有限的训练观测")
        # 复用研究配方按列整理后的数值布局；迭代优化器在接近类别边界时受求和顺序影响。
        column_data = np.asfortranarray(self.source)
        self.scaler = StandardScaler().fit(column_data)
        self.scaled = self.scaler.transform(column_data)
        self.lda = fixed_snapshot("lda",fit_discriminant(self.source,self.labels))
        saved = state or {}
        self.observer = ObserverErrors(bank_names,classes,saved.get("observer_counts"))
        self.history = saved.get("history",[])
        self.answers = saved.get("answers",[])
        self.last_size = saved.get("last_size",0)
        self.rounds = saved.get("feedback_rounds",0)
        self.active = saved.get("active","observer")
        self.loss = np.asarray(saved.get("counts",np.zeros((len(TOOLS),classes,classes))),dtype=int)
        if (len(self.history)!=len(self.answers) or self.loss.shape!=(len(TOOLS),classes,classes)
                or np.any(self.loss.sum(axis=(1,2)) != len(self.answers))
                or self.observer.counts[0].sum()!=len(self.answers) or self.active not in TOOLS):
            raise ValueError("现场恢复记录与反馈计数不一致")
        self.fits = 0
        self.cpu = 0.
        self._trained = None

    def state(self):
        return {"observer_counts":self.observer.counts.tolist(), "history":self.history,
                "answers":self.answers,"last_size":self.last_size,"feedback_rounds":self.rounds,
                "active":self.active,"counts":self.loss.tolist()}

    def models(self):
        if self._trained is None:
            start = process_time()
            features = self.scaled
            labels = self.labels
            balance = weights(labels,len(labels))
            recent = None
            if self.history:
                x, y = np.asarray(self.history), np.asarray(self.answers)
                features = np.vstack((features,self.scaler.transform(np.asfortranarray(x))))
                labels = np.r_[labels,y]
                balance = np.r_[balance,weights(y,len(self.labels))]
                tree = RandomForestClassifier(n_estimators=1,random_state=0,n_jobs=1)
                tree.fit(x[-self.last_size:], y[-self.last_size:])
                recent = fixed_snapshot("recent-tree",tree)
                self.fits += 1
            reg = LogisticRegression(C=1.,random_state=17,max_iter=1000).fit(features,labels,sample_weight=balance)
            self.fits += 1
            self._trained = {"mixed":fixed_snapshot("mixed",(self.scaler,reg)), "recent-tree":recent, "lda":self.lda}
            self.cpu += process_time()-start
        return self._trained

    def current(self):
        return self.mode if self.mode in TOOLS else self.active

    def snapshot(self):
        selected = self.current()
        return None if selected == "observer" else self.models()[selected]

    def predict(self, x, votes):
        observed = self.observer.predict(votes,"uniform")
        p = {"observer":observed}
        for name, record in self.models().items():
            p[name] = observed if record is None else 1+model_scores(record,x,self.classes).argmax(axis=1)
        return p

    def update(self, x, votes, correct, before):
        labels = np.asarray(correct,dtype=int)
        if not len(labels):
            return
        if len(x)!=len(labels) or any(len(before[a])!=len(labels) for a in TOOLS):
            raise ValueError("回标与当时发出的候选预测长度不同")
        update = np.zeros_like(self.loss)
        for j, a in enumerate(TOOLS):
            np.add.at(update[j], (labels-1,np.asarray(before[a],dtype=int)-1), 1)
        self.loss += update
        self.observer.update(labels,votes)
        if self.mode in ("ongoing","recent-backtest") or (self.mode=="once" and not self.rounds):
            tables = update if self.mode=="recent-backtest" else self.loss
            scores = [macro(m) for m in tables]
            index = int(np.argmax(scores))
            if scores[index] >= scores[TOOLS.index(self.active)] + .01:
                self.active = TOOLS[index]
        self.history.extend(np.asarray(x,dtype=float).tolist())
        self.answers.extend(labels.tolist())
        self.last_size = len(labels)
        self.rounds += 1
        self._trained = None
