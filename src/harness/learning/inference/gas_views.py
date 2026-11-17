"""气体测量的两种表示与固定读出。拟合只接收源训练行，预测逐行独立。"""
import numpy as np
from base64 import b64decode, b64encode


def represent(x, view, mean, scale):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
        raise ValueError("气体表示需要每行128个有限测量值")
    if view == "raw":
        return np.clip((x - mean) / scale, -10, 10)
    if view != "shape":
        raise ValueError("未知的测量表示")
    sensors = x.reshape(-1, 16, 8)
    centered = sensors - sensors.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))
    return (centered / np.maximum(rms, 1e-12)).reshape(-1, 128)


def fit_view(x, y, method, mean, scale):
    view, readout = method.split("-")
    if readout not in ("knn", "lda", "agree", "blend"):
        raise ValueError("未知的气体修正方法")
    train = represent(x, view, mean, scale)
    labels = np.asarray(y, dtype=int)
    if set(labels) != set(range(6)) or len(labels) != len(train):
        raise ValueError("修正模型需要六类完整且一致的训练样本")
    model = {"method": method}
    if readout in ("knn", "agree", "blend"):
        model.update(train=train.tolist(), labels=labels.tolist())
    if readout in ("lda", "agree"):
        centers = np.stack([train[labels == i].mean(0) for i in range(6)])
        residual = train - centers[labels]
        cov = residual.T @ residual / (len(train)-6)
        ridge = max(np.trace(cov) / len(cov), 1e-12)
        weights = np.linalg.solve(.8 * cov + .2 * ridge * np.eye(len(cov)), centers.T)
        model.update(weights=weights.tolist(), bias=(-.5 * (centers * weights.T).sum(1)).tolist())
    return model


def predict_view(model, x, mean, scale, parent_probs):
    view, readout = model["method"].split("-")
    features = represent(x, view, mean, scale)
    if readout in ("lda", "agree"):
        logits = features @ np.asarray(model["weights"]) + np.asarray(model["bias"])
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        lda = probs / probs.sum(axis=1, keepdims=True)
        if readout == "lda":
            return lda
    train = np.asarray(model["train"])
    labels = np.asarray(model["labels"])
    norms = model.get("train_norms")
    if norms is None:
        norms = (train**2).sum(axis=1)
    # 分块限制工作内存。稳定排序使距离相同时按官方训练清单的原顺序取五个。
    votes = np.empty((len(features), 6))
    for start in range(0, len(features), 64):
        part = features[start:start+64]
        dist = norms - 2 * part @ train.T
        top = labels[np.argsort(dist, axis=1, kind="stable")[:, :5]]
        votes[start:start+len(part)] = np.eye(6)[top].mean(axis=1)
    if readout == "knn":
        return votes
    if readout == "blend":
        return .5 * (votes + parent_probs)
    # 两种判别依据同意且近邻有足够支持，才允许改变该条预测。
    use = (votes.max(axis=1) >= .8) & (votes.argmax(axis=1) == lda.argmax(axis=1))
    return np.where(use[:, None], votes, parent_probs)


def build_pair_memory(spec, x, y, mean, scale):
    """嵌入固定训练参照；新机器预测时仅需完整 NPZ 及项目的读取器。"""
    memory = {"version": 1, **spec}
    if spec["method"] == "prior":
        memory["counts"] = np.bincount(y, minlength=6).tolist()
    else:
        bank = fit_view(x, y, spec["method"], mean, scale)
        data = np.asarray(bank["train"], dtype="<f8")
        memory.update(labels=bank["labels"], features64=b64encode(data.tobytes()).decode("ascii"))
    return memory


def append_pair(memory, spec, policy):
    """同一不可变参照库只保存一次。旧的单次修正原样保留为首条规则。"""
    if (policy not in ("cascade", "protected") or memory["version"] != 1
            or memory["method"] != spec["method"] or spec["method"] != "shape-blend"):
        raise ValueError("追加修正需要兼容的单次融合父版和明确的执行顺序")
    first = {"reported": memory["reported"], "target": memory["target"]}
    second = {"reported": spec["reported"], "target": spec["target"]}
    if first == second:
        raise ValueError("同一改判方向已经在用")
    return {**memory, "version": 2, "policy": policy, "rules": [first, second]}


def pair_model(base):
    """兼容原检查点；续训的优化器与特征尺度仍来自同一父版。"""
    class PairModel(base):
        def _prediction_bank(self, memory):
            if getattr(self, "_bank_source", None) is not memory:
                labels = np.asarray(memory["labels"])
                train = np.frombuffer(b64decode(memory["features64"], validate=True), dtype="<f8").reshape(len(labels), 128)
                self._bank = {"method": memory["method"], "train": train, "labels": labels,
                              "train_norms": (train**2).sum(axis=1)}
                self._bank_source = memory
            return self._bank

        def predict(self, x):
            return self._predict_pairs(x, None)

        def predict_with_repairs(self, x):
            """服务回执随实际执行路径生成；无修正规则、未触发规则时也有原判。"""
            traces = []
            scores = self._predict_pairs(x, traces)
            for trace, winner in zip(traces, scores.argmax(axis=1)):
                trace["after"] = int(winner) + 1
            return scores, traces

        def _predict_pairs(self, x, traces):
            scores = super().predict(x)
            if traces is not None:
                traces.extend({"before": int(label) + 1, "steps": []} for label in scores.argmax(axis=1))
            memory = self.meta.get("gas_pair_patch")
            if memory is None:
                return scores
            if memory["version"] == 1:
                rules = [memory]
                protected = False
            elif memory["version"] == 2 and memory["policy"] in ("cascade", "protected") and len(memory["rules"]) == 2:
                rules = memory["rules"]
                protected = memory["policy"] == "protected"
            else:
                raise ValueError("未知气体修正版本")
            # 同一入口的受保护规则只会采用同一份建议中的一个类别。
            # 两个出口可共查一次；入口相接或允许继续修改时，保留逐步更新。
            if protected and rules[0]["reported"] == rules[1]["reported"]:
                groups = [(rules[0]["reported"], (rules[0]["target"], rules[1]["target"]))]
            else:
                groups = [(rule["reported"], (rule["target"],)) for rule in rules]
            locked = np.zeros(len(x), dtype=bool) if protected and len(groups) > 1 else None
            for reported, targets in groups:
                eligible = scores.argmax(axis=1) == reported - 1
                if locked is not None:
                    eligible &= ~locked
                rows = np.flatnonzero(eligible)
                if not len(rows):
                    continue
                current = scores[rows]
                if memory["method"] == "prior":
                    proposal = current / np.asarray(memory["counts"])
                    proposal /= proposal.sum(axis=1, keepdims=True)
                else:
                    proposal = predict_view(self._prediction_bank(memory), x[rows], self.mean, self.scale, current)
                winner = proposal.argmax(axis=1) + 1
                allowed = winner == targets[0]
                if len(targets) == 2:
                    allowed |= winner == targets[1]
                changed = rows[allowed]
                if traces is not None:
                    for index, target in zip(changed, winner[allowed]):
                        traces[index]["steps"].append({"reported": reported, "target": int(target),
                                                       "method": memory["method"]})
                scores[changed] = proposal[allowed]
                if locked is not None:
                    locked[changed] = True
            return scores
    return PairModel
